from ast import Dict
import token
import torch
import train.Yujie.yj_navsim_train as config
import torch.nn as nn
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import torchvision.models as models
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import timm


class UtilitiesFunc:
    """
    Utility functions for image preprocessing and ResNet backbone.
    """
    @staticmethod
    def preprocess_image_seq(images):
        """
        Preprocesses image sequences.

        Args:
            images (torch.Tensor): Input image sequences.

        Returns:
            torch.Tensor: Preprocessed image sequences.
        """
        # 0 - 7: cameras [CAM_F0, CAM_L0, CAM_R0, CAM_L1, CAM_R1, CAM_L2, CAM_R2, CAM_B0]
        # The input image shape is: B * seq_len * 8 * 3 * 360 * 360
        refactored_images = []
        for sample_index in range(images.shape[0]):
            sample = images[sample_index]
            seq_images = []
            for seq_idx in range(images.shape[1]):
                sequence = sample[seq_idx]
                center = torch.zeros((3, 360, 360), dtype=torch.float32, device=config.device)
                cam_f0 = sequence[0]
                cam_l0 = sequence[1][:, :, 100:-80]  # 裁剪左边图片
                cam_r0 = sequence[2][:, :, 80:-100]  # 裁剪右边图片
                #cam_l1 = torch.rot90(sequence[3], k=1, dims=(1, 2))  # 逆时针90
                #cam_r1 = torch.rot90(sequence[4], k=-1, dims=(1, 2))  # 顺时针90
                #cam_l2 = torch.rot90(sequence[5], k=2, dims=(1, 2))  # 逆时针180
                #cam_r2 = torch.rot90(sequence[6], k=2, dims=(1, 2))  # 逆时针180
                #cam_b0 = torch.rot90(sequence[7], k=2, dims=(1, 2))  # 逆时针180

                # 将 9 个数组图像连接成一个3x3的大图像
                row1 = torch.cat([cam_l0, cam_f0, cam_r0], dim=2)  # 沿着宽度方向连接 360 * 720
                #row2 = torch.cat([cam_l1, center, cam_r1], dim=2)  # 沿着宽度方向连接
                #row3 = torch.cat([cam_l2, cam_b0, cam_r2], dim=2)  # 沿着宽度方向连接
                #big_image = torch.cat([row1, row2, row3], dim=1)  # 沿着高度方向
                big_image = row1
                seq_images.append(big_image)
            refactored_images.append(torch.stack(seq_images))
        # B * seq_len * 3 * 1080 * 1080
        refactored_images = torch.stack(refactored_images)
        # (B * seq_len) * 3 * 1080 * 1080
        refactored_images = refactored_images.view(-1, refactored_images.size(2), refactored_images.size(3),
                                                    refactored_images.size(4)).to(config.device)

        # combine sequence with channels, (B * seq_len) * 3 * 1080 * 1080
        return images.shape[0], refactored_images

    @staticmethod
    def preprocess_lidar_seq(lidars):
        # The input lidar shape is: B * seq_len * 1(2) * 256 * 256
        batch_size = lidars.size(0)
        # combine sequence with channels, (B * seq_len) * 1(2) * 256 * 256
        lidars_combined = lidars.view(-1, lidars.size(2), lidars.size(3), lidars.size(4))
        return batch_size, lidars_combined

    @staticmethod
    def recover_output_features(batch_size, features):
        """
        Recovers output features.

        Args:
            batch_size (int): Batch size.
            features (torch.Tensor): Image or lidar features.

        Returns:
            torch.Tensor: Recovered output features.
        """
        return features.view(batch_size, -1, features.size(-1))  # B * seq_len * 512
    
    @staticmethod
    def visual_lidar_bev(lidar_bev_, batch_i_, num_frame, below_above):
        # 0: below, 1: above
        plt.imshow(lidar_bev_[batch_i_][num_frame][below_above].cpu(), cmap='gray')
        plt.title('Bird\'s Eye View Image')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar(label='Point Count')
        plt.show()

    @staticmethod
    def visual_images(image, batch_i_, num_frame):
        plt.imshow(image[batch_i_][num_frame].cpu())
        plt.axis('off')
        plt.show()

class TransfuserBackbone(nn.Module):
    def __init__(self):
        super(TransfuserBackbone, self).__init__()
        # self.modeltype = "transfuser"

        self.image_encoder = timm.create_model(
            config.image_architecture, pretrained=True, features_only=True
        )

        if config.use_ground_plane:
            in_channels = 2
        else:
            in_channels = 1

        self.avgpool_img = nn.AdaptiveAvgPool2d(
            (config.img_vert_anchors, config.img_horz_anchors)
        )

        self.lidar_encoder = timm.create_model(
            config.lidar_architecture,
            pretrained=False,
            in_chans=in_channels,
            features_only=True,
        )
        self.global_pool_lidar = nn.AdaptiveAvgPool2d(output_size=1)
        self.avgpool_lidar = nn.AdaptiveAvgPool2d(
            (config.lidar_vert_anchors, config.lidar_horz_anchors)
        )
        
        self.global_pool_img = nn.AdaptiveAvgPool2d(output_size=1)
        start_index = 0
        self.transformers = nn.ModuleList(
            [
                Transformer(
                    n_embd=self.image_encoder.feature_info.info[start_index + i]["num_chs"],
                )
                for i in range(4)
            ]
        )

        self.lidar_channel_to_img = nn.ModuleList(
            [
                nn.Conv2d(
                    self.lidar_encoder.feature_info.info[start_index + i]["num_chs"],
                    self.image_encoder.feature_info.info[start_index + i]["num_chs"],
                    kernel_size=1,
                )
                for i in range(4)
            ]
        )

        self.img_channel_to_lidar = nn.ModuleList(
            [
                nn.Conv2d(
                    self.image_encoder.feature_info.info[start_index + i]["num_chs"],
                    self.lidar_encoder.feature_info.info[start_index + i]["num_chs"],
                    kernel_size=1,
                )
                for i in range(4)
            ]
        )

        self.num_image_features = self.image_encoder.feature_info.info[start_index + 3]["num_chs"]

        self.decoder_layer = TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
        )
        self.transformer_decoder = TransformerDecoder(
            self.decoder_layer, num_layers=config.tf_num_layers
        )

        if config.add_features:
            self.lidar_to_img_features_end = nn.Linear(
                self.lidar_encoder.feature_info.info[start_index + 3]["num_chs"],
                self.image_encoder.feature_info.info[start_index + 3]["num_chs"],
            )
            # feature的数量， 如果加起来，那么就是跟image的feature数量一样
            self.num_features = self.image_encoder.feature_info.info[start_index + 3]["num_chs"]
        else:
            # feature的数量， 如果cat起来，那么就是image和lidar的feature数量之和
            self.num_features = (
                    self.image_encoder.feature_info.info[start_index + 3]["num_chs"]
                    + self.lidar_encoder.feature_info.info[start_index + 3]["num_chs"]
            )

    def forward_layer_block(self, layers, return_layers, features):
        """
        Run one forward pass to a block of layers from a TIMM neural network and returns the result.
        Advances the whole network by just one block
        :param layers: Iterator starting at the current layer block
        :param return_layers: TIMM dictionary describing at which intermediate layers features are returned.
        :param features: Input features
        :return: Processed features
        """
        for name, module in layers:
            features = module(features)
            if name in return_layers:
                break
        return features
    
    def fuse_features(self, image_features, lidar_features, velocity, layer_idx):
        # 先进行average pooling, image变成[B*4, C, 8, 8]，lidar变成[B, C, 8, 8]
        image_embd_layer = self.avgpool_img(image_features)
        lidar_embd_layer = self.avgpool_lidar(lidar_features)
        # 使image和lidar的channel数相同
        lidar_embd_layer = self.lidar_channel_to_img[layer_idx](lidar_embd_layer)

        # 进行transformer的操作， 这里是将image和lidar进行concat，然后进行transformer的self attention操作
        image_features_layer, lidar_features_layer = self.transformers[layer_idx](
            image_embd_layer, lidar_embd_layer, velocity
        )
        lidar_features_layer = self.img_channel_to_lidar[layer_idx](lidar_features_layer)

        # 将averge pooling之后的image feature map进行差值 对于lidar也是一样的
        image_features_layer = F.interpolate(
            image_features_layer,
            size=(image_features.shape[2], image_features.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        lidar_features_layer = F.interpolate(
            lidar_features_layer,
            size=(lidar_features.shape[2], lidar_features.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        # 进行一个残差的操作
        image_features = image_features + image_features_layer
        lidar_features = lidar_features + lidar_features_layer

        return image_features, lidar_features
    
    def forward(self, image_features, lidar_features, velocity):
        # Generate an iterator for all the layers in the network that one can loop through.
        image_layers = iter(self.image_encoder.items())
        lidar_layers = iter(self.lidar_encoder.items())
        # Loop through the 4 blocks of the network.

        for i in range(4):
            # 先进行resnet的每一层，从[704, 160, 3]到[176, 40, 72]到[88, 20, 144]到[44, 10, 288]
            image_features = self.forward_layer_block(
                image_layers, self.image_encoder.return_layers, image_features
            )
            # 同样进行resnet每一层，从[256, 256, 3]到[64, 64, 72]到[32, 32, 144]到[16, 16, 288]
            lidar_features = self.forward_layer_block(
                lidar_layers, self.lidar_encoder.return_layers, lidar_features
            )
            # 进行特征融合
            image_features, lidar_features = self.fuse_features(image_features, lidar_features, velocity, i)

        # 全局平均池化， image变成[B*seq_len, C, 1, 1]，lidar变成[B*seq_len, C, 1, 1]
        image_features = self.global_pool_img(image_features)
        image_features = torch.flatten(image_features, 1)
        lidar_features = self.global_pool_lidar(lidar_features)
        lidar_features = torch.flatten(lidar_features, 1)

        # 选择是加起来还是cat起来
        if config.add_features:
            lidar_features = self.lidar_to_img_features_end(lidar_features)
            fused_features = image_features + lidar_features
        else:
            fused_features = torch.cat((image_features, lidar_features), dim=1)

        return fused_features

class Transformer(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.n_embd = n_embd
        self.seq_len = config.n_input_frames
        self.lidar_time_frames = config.n_input_lidar_frames
        self.image_time_frames = config.n_input_image_frames
        # positional embedding parameter (learnable), image + lidar
        self.pos_emb = nn.Parameter(
            torch.zeros(
                1,
                self.image_time_frames * config.img_vert_anchors * config.img_horz_anchors
                + self.lidar_time_frames * config.lidar_vert_anchors * config.lidar_horz_anchors,
                self.n_embd,
            )
        )

        # velocity embedding
        if(config.use_velocity == True):
            self.vel_emb = nn.Linear(self.seq_len * config.n_feature_dim, self.n_embd)  # x_y velocity [B, seq_len * 2]

        self.block = TransformerEncoderLayer(
            d_model=n_embd,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout
        )
        self.blocks = TransformerEncoder(self.block, num_layers=config.n_layers)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.ln_f = nn.LayerNorm(n_embd)

    def forward(self, image_tensor, lidar_tensor, velocity):
        """
        Forward pass of the GPT model.
        Args:
            image_tensor (tensor): B*seq_len, C, H, W
            lidar_tensor (tensor): B*lidar_time_frame, C, H, W
        Returns:
            image_tensor_out (tensor): B, H, W, C
            lidar_tensor_out (tensor): B, H, W, C

        """
        bz_seq_len = lidar_tensor.shape[0] # batch size * seq_len
        bz = bz_seq_len // self.lidar_time_frames
        lidar_h, lidar_w = lidar_tensor.shape[2:4]
        img_h, img_w = image_tensor.shape[2:4]

        # image_tensor (tensor) : B, X, n_embd
        image_tensor = image_tensor.permute(0, 2, 3, 1).contiguous().view(bz, -1, self.n_embd)
        # lidar_tensor (tensor) : B, X, n_embd
        lidar_tensor = lidar_tensor.permute(0, 2, 3, 1).contiguous().view(bz, -1, self.n_embd)
            
        token_embeddings = torch.cat((image_tensor, lidar_tensor), dim=1) # 这里将lidar和image的特征拼接在一起
        
        # project velocity to n_embed
        if(config.use_velocity==True):
            velocity_embeddings = self.vel_emb(velocity) # (B, C)
            # add (learnable) positional embedding and velocity embedding for all tokens
            x = self.drop(self.pos_emb + token_embeddings + velocity_embeddings.unsqueeze(1)) #(B, an * T, C)
        else:
            x = self.drop(self.pos_emb + token_embeddings)

        x = self.blocks(x)  # (B, an * T, C)
        x = self.ln_f(x)  # (B, an * T, C)

        image_tensor_out = (
            x[:, : self.image_time_frames * config.img_vert_anchors * config.img_horz_anchors, :]
            .reshape(bz * self.image_time_frames, img_h, img_w, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        lidar_tensor_out = (
            x[
            :,
            self.lidar_time_frames * config.lidar_vert_anchors * config.lidar_horz_anchors:,
            :,
            ]
            .reshape(bz * self.lidar_time_frames, lidar_h, lidar_w, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        return image_tensor_out, lidar_tensor_out

class TransfuserModel(nn.Module):
    def __init__(self):
        super(TransfuserModel, self).__init__()
        self.model_type = "Transfuser"
        self.backbone = TransfuserBackbone()
        self.decoder_layer = TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
        )
        self.transformer_decoder = TransformerDecoder(
            self.decoder_layer, num_layers=config.tf_num_layers
        )
        self.mhd = nn.MultiheadAttention(
            embed_dim=config.tf_d_model,
            num_heads=config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.seq_len = config.n_input_frames
        self.feature_len = config.n_input_image_frames  
        #self.status_encoding = nn.Linear(config.n_feature_dim, config.tf_d_model)
        self.gru_decoder = nn.GRU(input_size=config.tf_d_model, hidden_size=config.tf_d_model*2, num_layers=5, batch_first=True)
        self.fc2 = nn.Linear(config.tf_d_model*2*self.seq_len, config.n_output_frames*config.n_output_dim)
        
        if config.add_features == True:
            self.join = nn.Sequential(
                                nn.Linear(256, 128),
                                nn.ReLU(inplace=True),
                                nn.Linear(128, 64),
                                nn.ReLU(inplace=True),
                            )
        else:
            self.join = nn.Sequential(
                                nn.Linear(512, 256),
                                nn.ReLU(inplace=True),
                                nn.Linear(256, 128),
                                nn.ReLU(inplace=True),
                                nn.Linear(128, 64),
                                nn.ReLU(inplace=True),
                            )
        self.decoder = nn.GRUCell(input_size=3, hidden_size=64)
        self.output = nn.Linear(64, 3)  # predict x, y


    def forward(self, ego_states, images, lidar):
        # ego_states = ego_states[:, 2:4, :]
        # images = images[:, 2:4, :, :, :, :]
        # lidar = lidar[:, 2:4, :, :, :]
        batch_size = ego_states.shape[0]
        status_features = ego_states
    
        _, camera_features = UtilitiesFunc.preprocess_image_seq(images)
        _, lidar_features = UtilitiesFunc.preprocess_lidar_seq(lidar)

        # torch.cat([ego_pose, velocity, acceleration, driving_command], dim=-1) [B, seq_len, 11]
        # only use velocity
        # status_features = status_features[:, :, 0:7]
        status_features = status_features.contiguous().view(batch_size, -1)

        fused_features = self.backbone(camera_features, lidar_features, status_features)
        fused_features = fused_features.view(batch_size, self.feature_len, -1) # cat: [B, feature_len, 512] add: [B, feature_len, 256]
        
        z = self.join(fused_features)  # [B, feature_len, 64]
        # use the last frame
        z = z[:, -1, :]

        output_wp = list()
        # initial input variable to GRU
        x = torch.zeros(size=(z.shape[0], 3), dtype=z.dtype).to(z.device)
        
        # autoregressive generation of output waypoints
        for _ in range(config.n_output_frames):
            x_in = x
            z = self.decoder(x_in, z)
            dx = self.output(z)
            x = dx + x
            output_wp.append(x)
        
        output = torch.stack(output_wp, dim=1)
        
        # status_features = self.status_encoding(status_features) # [B, seq_len, 512]
        
        # query = status_features
        # key = fused_features
        # value = fused_features

        # memory, _ = self.mhd(query, key, value)
        # output, h_n = self.gru_decoder(memory)
        # bs = output.size()[0]
        # output = output.reshape(bs, -1)
        # output = self.fc2(output)
        # output = output.reshape(bs, -1, config.n_output_dim)

        #keyval = torch.cat([fused_features, status_encoding[:, None]], dim=1)
        #keyval += self.keyval_embedding.weight[None, ...]

        #query = self.query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        #query_out = self.transformer_decoder(query, keyval)

        return output


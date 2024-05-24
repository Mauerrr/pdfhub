import math
import torch
from torch import nn
import torch.nn.functional as F
import timm
import copy
from torch.nn import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer


class TransfuserBackbone(nn.Module):
    def __init__(self, config):
        super(TransfuserBackbone, self).__init__()
        self.modeltype = "transfuser"
        self.config = config

        self.image_encoder = timm.create_model(
            config.image_architecture, pretrained=True, features_only=True
        )

        if config.use_ground_plane:
            in_channels = 2 * config.lidar_seq_len
        else:
            in_channels = config.lidar_seq_len

        self.avgpool_img = nn.AdaptiveAvgPool2d(
            (self.config.img_vert_anchors, self.config.img_horz_anchors)
        )

        self.lidar_encoder = timm.create_model(
            config.lidar_architecture,
            pretrained=False,
            in_chans=in_channels,
            features_only=True,
        )
        self.global_pool_lidar = nn.AdaptiveAvgPool2d(output_size=1)
        self.avgpool_lidar = nn.AdaptiveAvgPool2d(
            (self.config.lidar_vert_anchors, self.config.lidar_horz_anchors)
        )
        lidar_time_frames = 1

        self.global_pool_img = nn.AdaptiveAvgPool2d(output_size=1)
        start_index = 0
        self.transformers = nn.ModuleList(
            [
                Transformer(
                    n_embd=self.image_encoder.feature_info.info[start_index + i]["num_chs"],
                    config=config,
                    lidar_time_frames=lidar_time_frames,
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

        if self.config.add_features:
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
    
    def fuse_features(self, image_features, lidar_features, layer_idx):
        # 先进行average pooling, image变成[B*4, C, 8, 8]，lidar变成[B, C, 8, 8]
        image_embd_layer = self.avgpool_img(image_features)
        lidar_embd_layer = self.avgpool_lidar(lidar_features)
        # 使image和lidar的channel数相同
        lidar_embd_layer = self.lidar_channel_to_img[layer_idx](lidar_embd_layer)

        # 进行transformer的操作， 这里是将image和lidar进行concat，然后进行transformer的self attention操作
        image_features_layer, lidar_features_layer = self.transformers[layer_idx](
            image_embd_layer, lidar_embd_layer
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
    
    def forward(self, image, lidar):
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
            image_features, lidar_features = self.fuse_features(image_features, lidar_features, i)

        # 全局平均池化， image变成[B*seq_len, C, 1, 1]，lidar变成[B*seq_len, C, 1, 1]
        image_features = self.global_pool_img(image_features)
        image_features = torch.flatten(image_features, 1)
        lidar_features = self.global_pool_lidar(lidar_features)
        lidar_features = torch.flatten(lidar_features, 1)

        # 选择是加起来还是cat起来
        if self.config.add_features:
            lidar_features = self.lidar_to_img_features_end(lidar_features)
            fused_features = image_features + lidar_features
        else:
            fused_features = torch.cat((image_features, lidar_features), dim=1)

        return fused_features

class Transformer(nn.Module):
    def __init__(self, n_embd, config, lidar_time_frames):
        super().__init__()
        self.n_embd = n_embd
        self.seq_len = 4
        self.config = config
        self.lidar_time_frames = lidar_time_frames

        # positional embedding parameter (learnable), image + lidar
        self.pos_emb = nn.Parameter(
            torch.zeros(
                1,
                self.seq_len * self.config.img_vert_anchors * self.config.img_horz_anchors
                + lidar_time_frames
                * self.config.lidar_vert_anchors
                * self.config.lidar_horz_anchors,
                self.n_embd,
            )
        )

        self.block = TransformerEncoderLayer(
            d_model=n_embd,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout
        )
        self.blocks = TransformerEncoder(self.block, num_layers=config.n_layers)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.ln_f = nn.LayerNorm(n_embd)

    def forward(self, image_tensor, lidar_tensor):
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

        x = self.drop(self.pos_emb + token_embeddings)
        x = self.blocks(x)  # (B, an * T, C)
        x = self.ln_f(x)  # (B, an * T, C)

        image_tensor_out = (
            x[:, : self.seq_len * self.config.img_vert_anchors * self.config.img_horz_anchors, :]
            .view(bz * self.seq_len, img_h, img_w, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        lidar_tensor_out = (
            x[
            :,
            self.seq_len * self.config.img_vert_anchors * self.config.img_horz_anchors:,
            :,
            ]
            .view(bz * self.lidar_time_frames, lidar_h, lidar_w, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        return image_tensor_out, lidar_tensor_out

class TransfuserModel(nn.Module):
    def __init__(self, config):
        super(TransfuserModel, self).__init__()
        self.config = config
        self.backbone = TransfuserBackbone(config)
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
        self.seq_len = 4
        self.status_encoding = nn.Linear(3, config.tf_d_model)
        self.gru_decoder = nn.GRU(input_size=config.tf_d_model, hidden_size=config.tf_d_model*2, num_layers=5, batch_first=True)
        self.fc2 = nn.Linear(config.tf_d_model*self.seq_len, self.seq_len*3)

    def forward(self, images, lidar, ego_states):
        batch_size = ego_states.shape[0]

        camera_feature = images.view(batch_size*self.seq_len, 3, 1920, 1080)

        lidar_feature = lidar.view(batch_size*self.seq_len, 2, 256, 256)

        status_feature = ego_states # [B, 4, dim]

        fused_features = self.backbone(camera_feature, lidar_feature)
        fused_features = fused_features.view(batch_size, self.seq_len, -1) # [B, 4, 512]

        status_features = self.status_encoding(status_feature) # [B, 4, 512]

        query = status_features
        key = fused_features
        value = fused_features

        memory = self.mhd(query, key, value)
        output, h_n = self.gru_decoder(memory)

        bs = output.size()[0]
        output = output.view(bs, -1)
        output = self.fc2(output)
        output = output.view(bs, -1, 2)

        #keyval = torch.cat([fused_features, status_encoding[:, None]], dim=1)
        #keyval += self.keyval_embedding.weight[None, ...]

        #query = self.query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        #query_out = self.transformer_decoder(query, keyval)

        return output
    
    















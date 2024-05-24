import torch
import train.Yujie.yj_navsim_train as config
import torch.nn as nn
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np


class TransfuserFusionModel:
    """
    Model class for gh_image_lidar_agent.

    This class provides the models needed for training, with inputs including ego states, images, and lidar point clouds.
    """

    class GRU(torch.nn.Module):
        def __init__(self, n_feature_dim=config.n_feature_dim, hidden_size=config.hidden_size,
                     n_layers=config.n_layers, n_output_dim=config.n_output_dim,
                     n_output_frames=config.n_output_frames, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_type = "gru"
            self.n_feature_dim = n_feature_dim
            self.hidden_size = hidden_size
            self.n_layers = n_layers
            self.n_output_dim = n_output_dim
            self.n_output_frames = n_output_frames

            # Define GRU layer
            self.gru = torch.nn.GRU(input_size=self.n_feature_dim, hidden_size=self.hidden_size,
                                    num_layers=self.n_layers, batch_first=True)

            # Define fully connected layer
            self.fc_gru = torch.nn.Linear(in_features=self.hidden_size,
                                          out_features=self.n_output_dim * self.n_output_frames)

        def forward(self, x, inplace=None):
            """
            Forward pass of the GRU model.

            Args:
                x (torch.Tensor): Input tensor of shape [batch_size, sequence_length, input_size].
                inplace (bool, optional): Whether to perform the operation inplace.

            Returns:
                torch.Tensor: Output tensor of shape [batch_size, sequence_length, output_dim].
            """
            batch_size, _, _ = x.size()

            # Initialize hidden state
            h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(config.device)

            # Forward pass through GRU layer
            out, _ = self.gru(x, h0)  # out -> [batch_size, sequence length, hidden_size]

            # Extract only the last timestamp to retain sequence information
            out = out[:, -1, :]  # out -> [batch_size, hidden_size]

            # Apply fully connected layer
            out = self.fc_gru(out)  # Output shape: [batch_size, n_output_dim * n_output_frames]

            return out

    class UtilitiesFunc:
        """
        Utility functions for image preprocessing and ResNet backbone.
        """

        class PositionalEncoding(nn.Module):
            """
            Positional Encoding module for Transformer-based models.
            """

            def __init__(self, d_model=config.tf_embedding_dim, dropout=0.1):
                super().__init__()
                self.emb_size = d_model
                self.dropout = nn.Dropout(dropout)

            def forward(self, embedding):
                div_term = torch.exp(-torch.arange(0, self.emb_size, 2) * math.log(10000) / self.emb_size)
                pos = torch.arange(0, embedding.shape[1]).unsqueeze(1)  # seq_len * 1
                pos_embedding = torch.zeros((embedding.shape[1], self.emb_size))  # seq_len * 512
                pos_embedding[:, 0::2] = torch.sin(pos * div_term)
                pos_embedding[:, 1::2] = torch.cos(pos * div_term)
                pos_embedding = pos_embedding.unsqueeze(0).to(embedding.device)  # 1 * seq_len * 512
                return self.dropout(embedding + pos_embedding).to(embedding.device)  # B * seq_len * 512

        class ImageResNetBackbone(torch.nn.Module):
            """
            ResNet backbone for feature extraction from images.
            """

            def __init__(self):
                super().__init__()

                self.resnet34_model = models.resnet34(pretrained=True)
                # Freeze pretrained weights
                # for param in self.resnet34_model.parameters():
                #     param.requires_grad = False

                num_features = self.resnet34_model.fc.in_features
                self.resnet34_model.fc = nn.Identity()  # Remove original fully connected layer
                # self.resnet34_model_final_fc = torch.nn.Sequential(
                #     nn.ReLU(),
                #     nn.Linear(in_features=num_features, out_features=config.tf_embedding_dim)  # dim_model = 512
                # )

            def forward(self, x):
                x = self.resnet34_model(x.to(torch.float32))
                #x = self.resnet34_model_final_fc(x)
                return x

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

        class LidarResNetBackbone(torch.nn.Module):
            """
            ResNet backbone for feature extraction from images.
            """

            def __init__(self):
                super().__init__()

                self.resnet18_model = models.resnet18()

                num_features = self.resnet18_model.fc.in_features
                self.resnet18_model.fc = nn.Identity()  # Remove original fully connected layer
                # self.resnet18_model_final_fc = torch.nn.Sequential(
                #     nn.ReLU(),
                #     nn.Linear(in_features=num_features, out_features=config.tf_embedding_dim)  # dim_model = 512
                # )
                _tmp = self.resnet18_model.conv1
                self.resnet18_model.conv1 = nn.Conv2d(config.channel_lidar_feature, out_channels=_tmp.out_channels, kernel_size=_tmp.kernel_size, stride=_tmp.stride, padding=_tmp.padding, bias=_tmp.bias) # type: ignore

            def forward(self, x):
                x = self.resnet18_model(x.to(torch.float32))
                #x = self.resnet18_model_final_fc(x)
                return x
            
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

    class AllImagesLidarsEgoFusion(nn.Module):
        def __init__(self, input_feature_dim=config.n_feature_dim, output_traj_dim=3, num_encoder_layers=1,
                     hidden_dim=config.tf_embedding_dim, num_heads=8, dropout=0.1, target_seq_len=8):
            """
            Initializes the AllImagesEgoFusion model.

            Args:
                input_feature_dim (int): Dimensionality of input features. Default is config.n_feature_dim.
                output_traj_dim (int): Dimensionality of output trajectory. Default is 3.
                num_encoder_layers (int): Number of encoder layers in the Transformer. Default is 1.
                hidden_dim (int): Dimensionality of hidden layers. Default is config.tf_embedding_dim.
                num_heads (int): Number of attention heads. Default is 8.
                dropout (float): Dropout probability. Default is 0.1.
                target_seq_len (int): Length of the target sequence. Default is 8.
            """
            super().__init__()
            self.model_type = "AllImagesEgoFusion"
            self.dim_model = hidden_dim
            self.target_feature_dim = output_traj_dim
            self.target_seq_len = target_seq_len
            self.resnet_image = TransfuserFusionModel.UtilitiesFunc.ImageResNetBackbone()
            self.resnet_lidar = TransfuserFusionModel.UtilitiesFunc.LidarResNetBackbone()
            
            self.ego_state_embedding = nn.Linear(input_feature_dim, hidden_dim)  # TODO: can also change to GRU
            # input -> B * 8 * 3, output -> B * 8 * 512
            self.output_embedding = nn.Linear(output_traj_dim, hidden_dim)
            # self.positional_encoder = TransfuserFusionModel.UtilitiesFunc.PositionalEncoding(d_model=hidden_dim,
            #                                                                          dropout=dropout)
            # positional embedding parameter (learnable), image + lidar
            self.pos_emb = nn.Parameter(torch.zeros(1, 2 * config.n_input_frames, config.tf_embedding_dim))
            self.drop = nn.Dropout(config.embd_pdrop)
            encoder_layers = TransformerEncoderLayer(self.dim_model, num_heads, dropout=dropout, batch_first=True)
            self.encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
            self.out = nn.Linear(hidden_dim, output_traj_dim)
            # 1D Convolution layer to reduce sequence length
            self.conv1d = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=2, stride=2)

            self.mhd = nn.MultiheadAttention(
                embed_dim=config.tf_d_model,
                num_heads=config.tf_num_head,
                dropout=config.tf_dropout,
                batch_first=True,
            )
            self.final_gru = TransfuserFusionModel.GRU(self.dim_model).to(config.device)

        def forward(self, ego_state, images, lidar_bev):
            """
            Forward pass of the AllImagesEgoFusion model.

            Args:
                ego_state (torch.Tensor): Ego-state tensor of shape B * seq_len * 11.
                images (torch.Tensor): Image tensor of shape B * seq_len * 1920 * 1080 * 3.

            Returns:
                torch.Tensor: Output trajectory tensor of shape B * (S * 3).
            """
            """Get image features with positional encoding: shape is B * seq_len * 1920 * 1080 * 3 --> B * seq_len * 512"""
            
            # preprocessed_images: (B * seq) * 3 * 1080 * 1080 (360 * 720)
            batch_size, preprocessed_images = TransfuserFusionModel.UtilitiesFunc.ImageResNetBackbone.preprocess_image_seq(images)
            #visual_images(preprocessed_images, 0, 0)
            image_feature_raw = self.resnet_image(preprocessed_images)  # (B * seq) * 512
            images_feature = TransfuserFusionModel.UtilitiesFunc.recover_output_features(batch_size, image_feature_raw) * math.sqrt(self.dim_model)  # B * seq * 512

            """Get lidar features: shape is B * seq_len * 1(2) * 256 * 256 --> B * seq_len * 512"""
            lidar_bev = torch.flip(lidar_bev, dims=[3, 4])  # "Flip the tensor upside down."
            #visual_lidar_bev(lidar_bev, 0, 0, 1)  # visualization lidar bev
            # preprocessed_lidar_bev: (B * seq) * 2 * 256 * 256
            batch_size, preprocessed_lidar_bev = TransfuserFusionModel.UtilitiesFunc.LidarResNetBackbone.preprocess_lidar_seq(lidar_bev)
            
            lidar_feature_raw = self.resnet_lidar(preprocessed_lidar_bev)  # (B * seq) * 512
            lidar_feature = TransfuserFusionModel.UtilitiesFunc.recover_output_features(batch_size, lidar_feature_raw) * math.sqrt(self.dim_model)  # B * seq * 512
            
            token_embeddings = torch.cat((images_feature, lidar_feature), dim=1) # 这里将lidar和image的特征拼接在一起

            """Get ego features with positional encoding: shape is B * seq_len * 11 --> B * seq_len * 512"""
            ego_state_features = self.ego_state_embedding(ego_state) * math.sqrt(self.dim_model)
            
            fused_features = self.drop(self.pos_emb + token_embeddings)  # B * (2 * seq_len) * 512
            
            # B * (2 * seq_len) * 512 --> B * seq_len * 512
            # Reshape to (B, 512, 2 * seq_len) for Conv1D
            fused_features = fused_features.permute(0, 2, 1)  # Shape: B * 512 * (2 * seq_len)

            # Apply 1D convolution to reduce sequence length
            fused_features = self.conv1d(fused_features)  # Shape: B * 512 * seq_len

            # Reshape back to (B, seq_len, 512)
            fused_features = fused_features.permute(0, 2, 1)  # Shape: B * seq_len * 512


            """Get fused features"""
            # fused_features: B * seq_len * X
            fused_features = self.encoder(fused_features)
            
            query = ego_state_features
            key = fused_features
            value = fused_features

            memory, _ = self.mhd(query, key, value)

            output_traj = self.final_gru(memory)  # B * (S * 3)
            return output_traj

    class TransfuserModified(nn.Module):
        pass

def visual_lidar_bev(lidar_bev_, batch_i_, num_frame, below_above):
    # 0: below, 1: above
    plt.imshow(lidar_bev_[batch_i_][num_frame][below_above].cpu(), cmap='gray')
    plt.title('Bird\'s Eye View Image')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar(label='Point Count')
    plt.show()

def visual_images(image, batch_i_, num_frame):
    plt.imshow(image[batch_i_][num_frame].cpu())
    plt.axis('off')
    plt.show()
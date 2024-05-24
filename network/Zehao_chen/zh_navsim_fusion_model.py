import torch
import train.Zehao_Chen.zh_navsim_train as config
import torch.nn as nn
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np


class FusionModel:
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

        class ResNetBackbone(torch.nn.Module):
            """
            ResNet backbone for feature extraction from images.
            """

            def __init__(self):
                super().__init__()

                self.resnet50_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                # Freeze pretrained weights
                for param in self.resnet50_model.parameters():
                    param.requires_grad = False

                num_features = self.resnet50_model.fc.in_features
                self.resnet50_model.fc = nn.Identity()  # Remove original fully connected layer
                self.resnet50_model_final_fc = torch.nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(in_features=num_features, out_features=config.tf_embedding_dim)  # dim_model = 512
                )

            def forward(self, x):
                x = self.resnet50_model(x.to(torch.float32))
                x = self.resnet50_model_final_fc(x)
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
                        cam_l0 = sequence[1]
                        cam_r0 = sequence[2]
                        cam_l1 = torch.rot90(sequence[3], k=1, dims=(1, 2))  # 逆时针90
                        cam_r1 = torch.rot90(sequence[4], k=-1, dims=(1, 2))  # 顺时针90
                        cam_l2 = torch.rot90(sequence[5], k=2, dims=(1, 2))  # 逆时针180
                        cam_r2 = torch.rot90(sequence[6], k=2, dims=(1, 2))  # 逆时针180
                        cam_b0 = torch.rot90(sequence[7], k=2, dims=(1, 2))  # 逆时针180

                        # 将 9 个数组图像连接成一个3x3的大图像
                        row1 = torch.cat([cam_l0, cam_f0, cam_r0], dim=2)  # 沿着宽度方向连接
                        row2 = torch.cat([cam_l1, center, cam_r1], dim=2)  # 沿着宽度方向连接
                        row3 = torch.cat([cam_l2, cam_b0, cam_r2], dim=2)  # 沿着宽度方向连接
                        big_image = torch.cat([row1, row2, row3], dim=1)  # 沿着高度方向
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
            def recover_output_features(batch_size, image_features):
                """
                Recovers output features.

                Args:
                    batch_size (int): Batch size.
                    image_features (torch.Tensor): Image features.

                Returns:
                    torch.Tensor: Recovered output features.
                """
                return image_features.view(batch_size, -1, image_features.size(-1))

    class AllImagesEgoFusion(nn.Module):
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
            self.resnet = FusionModel.UtilitiesFunc.ResNetBackbone()
            self.ego_state_embedding = nn.Linear(input_feature_dim, hidden_dim)  # TODO: can also change to GRU
            # input -> B * 8 * 3, output -> B * 8 * 512
            self.output_embedding = nn.Linear(output_traj_dim, hidden_dim)
            self.positional_encoder = FusionModel.UtilitiesFunc.PositionalEncoding(d_model=hidden_dim,
                                                                                     dropout=dropout)

            encoder_layers = TransformerEncoderLayer(self.dim_model, num_heads, dropout=dropout, batch_first=True)
            self.encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
            self.out = nn.Linear(hidden_dim, output_traj_dim)
            self.final_gru = FusionModel.GRU(self.dim_model).to(config.device)

        def forward(self, ego_state, images, lidar):
            """
            Forward pass of the AllImagesEgoFusion model.

            Args:
                ego_state (torch.Tensor): Ego-state tensor of shape B * seq_len * 11.
                images (torch.Tensor): Image tensor of shape B * seq_len * 1920 * 1080 * 3.

            Returns:
                torch.Tensor: Output trajectory tensor of shape B * (S * 3).
            """
            """Get image features with positional encoding: shape is B * seq_len * 1920 * 1080 * 3 --> B * seq_len * 512"""
            # show the lidar information
            # for sample_index in range(lidar.shape[0]):
            #     sample = lidar[sample_index]
            #     for seq_idx in range(lidar.shape[1]):
            #         sequence = sample[seq_idx]
            #         lidar_down = sequence[0]
            #         lidar_up = sequence[1]
            #         plt.imshow(np.array(lidar_up.reshape(256, 256).cpu()), cmap='gray')
            #         plt.colorbar()  # 添加颜色条
            #         plt.show()
            #         plt.imshow(np.array(lidar_down.reshape(256, 256).cpu()), cmap='gray')
            #         plt.colorbar()  # 添加颜色条
            #         plt.show()

            # preprocessed_images: (B * seq) * 3 * 1080 * 1080
            batch_size, preprocessed_images = FusionModel.UtilitiesFunc.ResNetBackbone.preprocess_image_seq(images)
            # image_features_raw: (B * seq) * 512
            image_features_raw = self.resnet(preprocessed_images)
            # images_features: B * seq * 512
            images_features = FusionModel.UtilitiesFunc.ResNetBackbone.recover_output_features(batch_size,
                                                                                                 image_features_raw) * math.sqrt(
                self.dim_model)
            images_features_pe = self.positional_encoder(images_features)

            """Get ego features with positional encoding: shape is B * seq_len * 11 --> B * seq_len * 512"""
            ego_state_features = self.ego_state_embedding(ego_state) * math.sqrt(self.dim_model)
            ego_state_features_pe = self.positional_encoder(ego_state_features)

            fused_features = ego_state_features_pe + images_features_pe

            """Get target features"""
            # fused_features: B * seq_len * self.embedding_dim
            # memory: B * seq_len * self.embedding_dim
            memory = self.encoder(fused_features)
            output_traj = self.final_gru(memory)  # B * (S * 3)
            return output_traj

    class TransfuserModified(nn.Module):
        pass


import torch
import train.Zehao_Chen.zh_navsim_train as config
import torch.nn as nn
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class MyEgoStateModel:
    class MLP(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_type = "mlp"
            self.backbone = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(config.n_input_frames * config.n_feature_dim, config.hidden_dim_fc),
                torch.nn.ReLU(),
                torch.nn.Linear(config.hidden_dim_fc, config.hidden_dim_fc),
                torch.nn.ReLU(),
                torch.nn.Linear(config.hidden_dim_fc, config.hidden_dim_fc),
                torch.nn.ReLU(),
                torch.nn.Linear(config.hidden_dim_fc, config.n_output_dim * config.n_output_frames),
            )

        def forward(self, x, inplace=None):
            return self.backbone(x)

    class RNN(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_type = "rnn"
            """
            input dim = 2 (the input x, y coordinates), hidden size: The number of features in the hidden state h
            num_layers – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two RNNs together to
            form a stacked RNN, with the second RNN taking in outputs of the first RNN and computing the final results.
            """

            self.hidden_size = config.hidden_size
            self.n_layers = config.n_layers
            self.rnn = torch.nn.RNN(input_size=config.n_feature_dim, hidden_size=self.hidden_size,
                                    num_layers=self.n_layers, batch_first=True)
            self.fc_rnn = torch.nn.Linear(in_features=config.hidden_size,
                                          out_features=config.n_output_dim * config.n_output_frames)

        def forward(self, x, inplace=None):
            batch_size, _, _ = x.size()
            h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(config.device)  # Initialize hidden state
            # The shape of x is: batch_size * sequence length * input size
            # the out containing the output features (h_t) from the last layer (the last of the 2 layers) of the RNN, for each t.
            out, _ = self.rnn(x, h0)  # out -> [batch_size, sequence length, hidden_size]
            out = out[:, -1, :]  # only the last timestamp  # out -> [batch_size, hidden_size]
            # we want the final output be [batch_size, sequence_length, 16]
            out = self.fc_rnn(out)
            return out

    class LSTM(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_type = "lstm"
            self.hidden_size = config.hidden_size
            self.n_layers = config.n_layers
            self.lstm = torch.nn.LSTM(input_size=config.n_feature_dim, hidden_size=self.hidden_size,
                                      num_layers=self.n_layers, batch_first=True)
            self.fc_lstm = torch.nn.Linear(in_features=config.hidden_size,
                                           out_features=config.n_output_dim * config.n_output_frames)

        def forward(self, x, inplace=None):
            # input x is: batch_size * sequence length * input dim
            batch_size, _, _ = x.size()
            h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(config.device)
            c0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(config.device)
            out, _ = self.lstm(x, (h0, c0))  # out -> [batch_size, sequence length, hidden_size]
            out = out[:, -1, :]  # only the last timestamp  # out -> [batch_size, hidden_size]
            # we want the final output be [batch_size, sequence_length, 16]
            out = self.fc_lstm(out)
            return out

    class GRU(torch.nn.Module):
        def __init__(self, n_feature_dim=config.n_feature_dim, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_type = "gru"
            self.hidden_size = config.hidden_size
            self.n_layers = config.n_layers
            self.gru = torch.nn.GRU(input_size=n_feature_dim, hidden_size=self.hidden_size,
                                    num_layers=self.n_layers, batch_first=True)
            self.fc_gru = torch.nn.Linear(in_features=config.hidden_size,
                                          out_features=config.n_output_dim * config.n_output_frames)

        def forward(self, x, inplace=None):
            batch_size, _, _ = x.size()
            h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(config.device)
            out, _ = self.gru(x, h0)  # out -> [batch_size, sequence length, hidden_size]
            out = out[:, -1, :]  # only the last timestamp  # out -> [batch_size, hidden_size]
            # we want the final output be [batch_size, sequence_length, 16]
            out = self.fc_gru(out)
            return out

    class GRU_XY(torch.nn.Module):
        def __init__(self, n_feature_dim=config.n_feature_dim, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_type = "gru"
            self.hidden_size = config.hidden_size
            self.n_layers = config.n_layers
            self.gru = torch.nn.GRU(input_size=n_feature_dim, hidden_size=self.hidden_size,
                                    num_layers=self.n_layers, batch_first=True)
            self.fc_gru = torch.nn.Linear(in_features=config.hidden_size,
                                          out_features=2 * config.n_output_frames)

        def forward(self, x, inplace=None):
            batch_size, _, _ = x.size()
            h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(config.device)
            out, _ = self.gru(x, h0)  # out -> [batch_size, sequence length, hidden_size]
            out = out[:, -1, :]  # only the last timestamp  # out -> [batch_size, hidden_size]
            # we want the final output be [batch_size, sequence_length, 16]
            out = self.fc_gru(out)
            return out

    class DriveUsingCommand(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_type = "driving_command"

            self.left = MyEgoStateModel.GRU(config.n_feature_dim - config.n_driving_command).to(config.device)
            self.straight = MyEgoStateModel.GRU(config.n_feature_dim - config.n_driving_command).to(config.device)
            self.right = MyEgoStateModel.GRU(config.n_feature_dim - config.n_driving_command).to(config.device)
            self.unknown = MyEgoStateModel.GRU(config.n_feature_dim - config.n_driving_command).to(config.device)

        def forward(self, x, inplace=None):
            batch_size, _, _ = x.size()
            command = x[:, -1, 7:].view(-1, 4)  # B * 4, the command at current timestamp
            command_indices = torch.argmax(command, dim=1).unsqueeze(1)  # B * 1, 选出每一行1的元素对应的index
            ego_state = x[:, :, :7]  # ego state B * 4 * 7, 是和上面的command_indices是一一对应的
            index0 = torch.where(command_indices == 0)[0]
            x_left = ego_state[index0, :, :]
            x_left = self.left(x_left.to(config.device))  # B0 * 24

            index1 = torch.where(command_indices == 1)[0]
            x_straight = ego_state[index1]
            x_straight = self.straight(x_straight.to(config.device))  # B1 * 24

            index2 = torch.where(command_indices == 2)[0]
            x_right = ego_state[index2]
            x_right = self.right(x_right.to(config.device))  # B2 * 24

            index3 = torch.where(command_indices == 3)[0]
            x_unknown = ego_state[index3]
            out_left = self.left(x_unknown.to(config.device))  # B3 * 24
            out_straight = self.straight(x_unknown.to(config.device))
            out_right = self.right(x_unknown.to(config.device))
            x_unknown = (out_left + out_straight + out_right) / 3

            y = torch.zeros(batch_size, config.n_output_dim * config.n_output_frames).to(config.device)
            y[index0] = x_left
            y[index1] = x_straight
            y[index2] = x_right
            y[index3] = x_unknown

            return y

    class TransformerModel:
        class PositionalEncoding(nn.Module):
            def __init__(self, d_model: int, dropout: float):
                super().__init__()
                self.emb_size = d_model
                self.dropout = nn.Dropout(dropout)

            def forward(self, embedding):
                div_term = torch.exp(- torch.arange(0, self.emb_size, 2) * math.log(10000) / self.emb_size)
                pos = torch.arange(0, embedding.shape[1]).unsqueeze(1)  # seq_len * 1
                pos_embedding = torch.zeros((embedding.shape[1], self.emb_size))  # seq_len * 512
                pos_embedding[:, 0::2] = torch.sin(pos * div_term)
                pos_embedding[:, 1::2] = torch.cos(pos * div_term)
                pos_embedding = pos_embedding.unsqueeze(0).to(embedding.device)  # 1 * seq_len * 512
                return self.dropout(embedding + pos_embedding).to(embedding.device)  # B * seq_len * 512

        class TransformerEgoStateModel(torch.nn.Module):
            def __init__(self, input_feature_dim, output_traj_dim=3, num_encoder_layers=1, num_decoder_layers=1,
                         hidden_dim=512, num_heads=8, dropout=0.1, target_seq_len=8):
                super().__init__()

                # Some model information
                self.model_type = "Transformer"
                self.dim_model = hidden_dim
                self.target_feature_dim = output_traj_dim
                self.target_seq_len = target_seq_len

                # input -> B * 4 * 11, output -> B * 4 * 512
                self.input_embedding = nn.Linear(input_feature_dim, hidden_dim)
                # input -> B * 8 * 3, output -> B * 8 * 512
                self.output_embedding = nn.Linear(output_traj_dim, hidden_dim)

                self.positional_encoder = MyEgoStateModel.TransformerModel.PositionalEncoding(d_model=512,
                                                                                              dropout=dropout)
                self.transformer = nn.Transformer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    num_encoder_layers=num_encoder_layers,
                    num_decoder_layers=num_decoder_layers,
                    dropout=dropout,
                    batch_first=True,
                )

                self.out = nn.Linear(hidden_dim, output_traj_dim)

            def forward(self, src, tgt):
                # expect src and tgt both be B * S * E
                """Shift left for train/ val"""
                batch_size = src.size()[0]

                tgt = torch.cat([torch.zeros(batch_size, 1, self.target_feature_dim).to(config.device), tgt[:, :-1, :]],
                                dim=1)

                src = self.input_embedding(src) * math.sqrt(
                    self.dim_model)  # input -> B * 4 * 11, output -> B * 4 * 512
                tgt = self.output_embedding(tgt.float()) * math.sqrt(
                    self.dim_model)  # input -> B * 8 * 3, output -> B * 8 * 512

                src = self.positional_encoder(src)
                tgt = self.positional_encoder(tgt)

                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[1]).to(
                    config.device)  # 生成和tgt_seq_len相同的方阵mask
                out = self.transformer(src, tgt, tgt_mask=tgt_mask)
                out = self.out(out)
                return out

        class TransformerModifiedEgoStateModel(torch.nn.Module):
            def __init__(self, input_feature_dim, output_traj_dim=3, num_encoder_layers=1, num_decoder_layers=1,
                         hidden_dim=512, num_heads=8, dropout=0.1, target_seq_len=8):
                super().__init__()
                # TODO: Encoder, Greedy decoder, first understand, then do the modification
                self.model_type = "TransformerModified"
                self.dim_model = hidden_dim
                self.target_feature_dim = output_traj_dim
                self.target_seq_len = target_seq_len

                # input -> B * 4 * 11, output -> B * 4 * 512
                self.input_embedding = nn.Linear(input_feature_dim, hidden_dim)
                # input -> B * 8 * 3, output -> B * 8 * 512
                self.output_embedding = nn.Linear(output_traj_dim, hidden_dim)
                # Positional Encoding
                self.positional_encoder = MyEgoStateModel.TransformerModel.PositionalEncoding(d_model=512,
                                                                                              dropout=dropout)
                # Encoder Layers
                encoder_layers = TransformerEncoderLayer(d_model=self.dim_model, nhead=num_heads, dropout=dropout,
                                                         batch_first=True)
                self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
                # Decoder Layers
                decoder_layers = TransformerDecoderLayer(d_model=self.dim_model, nhead=num_heads, dropout=dropout,
                                                         batch_first=True)
                self.transformer_decoder = TransformerDecoder(decoder_layers, num_decoder_layers)
                # Output Layer
                self.out = nn.Linear(hidden_dim, output_traj_dim)

            def forward(self, src, inplace=None):
                batch_size = src.size()[0]
                memory = self.transformer_encoder(
                    self.positional_encoder(self.input_embedding(src) * math.sqrt(self.dim_model)))
                y = torch.zeros(batch_size, 1, config.n_output_dim, dtype=torch.float, device=config.device)
                for _ in range(config.n_output_frames):
                    tgt_mask = nn.Transformer.generate_square_subsequent_mask(y.size()[1]).to(
                        config.device)  # 生成和tgt_seq_len相同的方阵mask
                    out = self.transformer_decoder(
                        self.positional_encoder(self.output_embedding(y) * math.sqrt(self.dim_model)), memory, tgt_mask)
                    out = self.out(out)

                    y = torch.cat((y, out[:, -1:, :]), dim=1)

                return y[:, 1:, :]


class MyFusionModel:
    class ResNetBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.resnet50_model = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT)  # weights=models.ResNet50_Weights.DEFAULT
            # 冻结预训练的权重
            for param in self.resnet50_model.parameters():
                param.requires_grad = False

            num_features = self.resnet50_model.fc.in_features
            self.resnet50_model.fc = nn.Identity()  # 去除原本的全连接层
            self.resnet50_model_final_fc = torch.nn.Sequential(
                nn.ReLU(),
                nn.Linear(in_features=num_features, out_features=config.tf_embedding_dim)  # dim_model = 512
            )

        def forward(self, x):
            x = self.resnet50_model(x.to(torch.float32))
            x = self.resnet50_model_final_fc(x)
            return x

        def multi_frames_all_images(self, x):
            pass

    class ImageOnlyGRU(torch.nn.Module):
        def __init__(self, output_traj_dim=3, hidden_dim=config.tf_embedding_dim, dropout=0.1):
            super().__init__()
            self.model_type = "ImageOnlyUsingGRU"
            self.resnet = MyFusionModel.ResNetBackbone()
            self.images_gru = MyEgoStateModel.GRU(config.tf_embedding_dim).to(config.device)

        @staticmethod
        def preprocess_image_seq(images):
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
            return image_features.view(batch_size, -1, image_features.size(-1))

        def forward(self, ego_state, images):
            batch_size, preprocessed_images = self.preprocess_image_seq(images)
            image_features_raw = self.resnet(preprocessed_images)
            images_features = self.recover_output_features(batch_size, image_features_raw)  # B * seq_len * 512
            output_traj = self.images_gru(images_features)  # B * (S * 3)
            return output_traj

    class ImageEgoFusion(torch.nn.Module):
        def __init__(self, input_feature_dim=config.n_feature_dim, output_traj_dim=3, num_encoder_layers=1,
                     num_decoder_layers=1,
                     hidden_dim=config.tf_embedding_dim, num_heads=8, dropout=0.1, target_seq_len=8):
            super().__init__()
            self.model_type = "Transformer"
            self.dim_model = hidden_dim
            self.target_feature_dim = output_traj_dim
            self.target_seq_len = target_seq_len

            self.resnet = MyFusionModel.ResNetBackbone()

            # input -> B * 4 * 11, output -> B * 4 * 512
            self.ego_state_embedding = nn.Linear(input_feature_dim, hidden_dim)  # TODO: can also change to GRU
            # input -> B * 8 * 3, output -> B * 8 * 512
            self.output_embedding = nn.Linear(output_traj_dim, hidden_dim)

            self.positional_encoder = MyEgoStateModel.TransformerModel.PositionalEncoding(d_model=hidden_dim,
                                                                                          dropout=dropout)

            encoder_layers = TransformerEncoderLayer(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            self.encoder = TransformerEncoder(encoder_layers, num_encoder_layers)

            decoder_layers = TransformerDecoderLayer(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            self.decoder = TransformerDecoder(decoder_layers, num_decoder_layers)

            self.out = nn.Linear(hidden_dim, output_traj_dim)

        def forward(self, ego_state, images, tgt):
            """Get image features with PE: shape is B * seq_len * 1920 * 1080 * 3 --> B * seq_len * 512"""
            batch_size, preprocessed_images = MyFusionModel.ImageOnlyGRU.preprocess_image_seq(images)
            image_features_raw = self.resnet(preprocessed_images)
            images_features = MyFusionModel.ImageOnlyGRU.recover_output_features(batch_size,
                                                                                 image_features_raw) * math.sqrt(
                self.dim_model)
            images_features_pe = self.positional_encoder(images_features)
            """Get ego features with PE: shape is B * seq_len * 11 --> B * seq_len * 512"""
            ego_state_features = self.ego_state_embedding(ego_state) * math.sqrt(self.dim_model)
            ego_state_features_pe = self.positional_encoder(ego_state_features)
            # 把feature相加
            fused_features = ego_state_features_pe + images_features_pe
            """Get target features"""
            tgt = torch.cat([torch.zeros(batch_size, 1, self.target_feature_dim).to(config.device), tgt[:, :-1, :]],
                            dim=1)
            tgt = self.output_embedding(tgt.float()) * math.sqrt(
                self.dim_model)  # input -> B * seq_len * 3, output -> B * seq_len * 512
            tgt_feature_pe = self.positional_encoder(tgt)
            # tgt_mask = torch.triu(torch.full((tgt.size()[1], tgt.size()[1]), float('-inf'), dtype=torch.get_default_dtype(), device=config.device), diagonal=1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[1]).to(
                config.device)  # 生成和tgt_seq_len相同的方阵mask

            memory = self.encoder(fused_features)
            output = self.decoder(tgt_feature_pe, memory, tgt_mask)

            output = self.out(output)

            return output

    class ImageEgoFusionTfGreedy(torch.nn.Module):
        def __init__(self, input_feature_dim=config.n_feature_dim, output_traj_dim=3, num_encoder_layers=1,
                     num_decoder_layers=1,
                     hidden_dim=config.tf_embedding_dim, num_heads=8, dropout=0.1, target_seq_len=8):
            super().__init__()
            self.model_type = "TransformerModified"
            self.dim_model = hidden_dim
            self.target_feature_dim = output_traj_dim
            self.target_seq_len = target_seq_len

            self.resnet = MyFusionModel.ResNetBackbone()

            # input -> B * 4 * 11, output -> B * 4 * 512
            self.ego_state_embedding = nn.Linear(input_feature_dim, hidden_dim)  # TODO: can also change to GRU
            # input -> B * 8 * 3, output -> B * 8 * 512
            self.output_embedding = nn.Linear(output_traj_dim, hidden_dim)

            self.positional_encoder = MyEgoStateModel.TransformerModel.PositionalEncoding(d_model=hidden_dim,
                                                                                          dropout=dropout)

            encoder_layers = TransformerEncoderLayer(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            self.encoder = TransformerEncoder(encoder_layers, num_encoder_layers)

            decoder_layers = TransformerDecoderLayer(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            self.decoder = TransformerDecoder(decoder_layers, num_decoder_layers)

            self.out = nn.Linear(hidden_dim, output_traj_dim)

        def forward(self, ego_state, images):
            """Get image features with PE: shape is B * seq_len * 1920 * 1080 * 3 --> B * seq_len * 512"""
            batch_size, preprocessed_images = MyFusionModel.ImageOnlyGRU.preprocess_image_seq(images)
            image_features_raw = self.resnet(preprocessed_images)
            images_features = MyFusionModel.ImageOnlyGRU.recover_output_features(batch_size,
                                                                                 image_features_raw) * math.sqrt(
                self.dim_model)
            images_features_pe = self.positional_encoder(images_features)
            """Get ego features with PE: shape is B * seq_len * 11 --> B * seq_len * 512"""
            ego_state_features = self.ego_state_embedding(ego_state) * math.sqrt(self.dim_model)
            ego_state_features_pe = self.positional_encoder(ego_state_features)
            # 把feature相加
            fused_features = ego_state_features_pe + images_features_pe
            """Get target features"""

            memory = self.encoder(fused_features)

            y = torch.zeros(batch_size, 1, config.n_output_dim, dtype=torch.float, device=config.device)
            for _ in range(config.n_output_frames):
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(y.size()[1]).to(
                    config.device)  # 生成和tgt_seq_len相同的方阵mask
                out = self.decoder(
                    self.positional_encoder(self.output_embedding(y) * math.sqrt(self.dim_model)), memory, tgt_mask)
                out = self.out(out)

                y = torch.cat((y, out[:, -1:, :]), dim=1)

            return y[:, 1:, :]

    class AllImagesEgoFusion(torch.nn.Module):
        def __init__(self, input_feature_dim=config.n_feature_dim, output_traj_dim=3, num_encoder_layers=1,
                     hidden_dim=config.tf_embedding_dim, num_heads=8, dropout=0.1, target_seq_len=8):
            super().__init__()
            self.model_type = "AllImagesEgoFusion"
            self.dim_model = hidden_dim
            self.target_feature_dim = output_traj_dim
            self.target_seq_len = target_seq_len
            self.resnet = MyFusionModel.ResNetBackbone()
            self.ego_state_embedding = nn.Linear(input_feature_dim, hidden_dim)  # TODO: can also change to GRU
            # input -> B * 8 * 3, output -> B * 8 * 512
            self.output_embedding = nn.Linear(output_traj_dim, hidden_dim)
            self.positional_encoder = MyEgoStateModel.TransformerModel.PositionalEncoding(d_model=hidden_dim,
                                                                                          dropout=dropout)

            encoder_layers = TransformerEncoderLayer(self.dim_model, num_heads, dropout=dropout, batch_first=True)
            self.encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
            self.out = nn.Linear(hidden_dim, output_traj_dim)
            self.final_gru = MyEgoStateModel.GRU(self.dim_model).to(config.device)

        def forward(self, ego_state, images):
            """

            Args:
                ego_state:
                images: B * seq * 8 * 3 * 360 * 360

            Returns:

            """
            """Get image features with PE: shape is B * seq_len * 1920 * 1080 * 3 --> B * seq_len * 512"""
            # preprocessed_images: (B * seq) * 3 * 1080 * 1080
            batch_size, preprocessed_images = MyFusionModel.ImageOnlyGRU.preprocess_image_seq(images)
            # image_features_raw: (B * seq) * 512
            image_features_raw = self.resnet(preprocessed_images)
            # images_features: B * seq * 512
            images_features = MyFusionModel.ImageOnlyGRU.recover_output_features(batch_size,
                                                                                 image_features_raw) * math.sqrt(
                self.dim_model)
            images_features_pe = self.positional_encoder(images_features)
            """Get ego features with PE: shape is B * seq_len * 11 --> B * seq_len * 512"""
            ego_state_features = self.ego_state_embedding(ego_state) * math.sqrt(self.dim_model)
            ego_state_features_pe = self.positional_encoder(ego_state_features)

            fused_features = ego_state_features_pe + images_features_pe

            """Get target features"""
            # fused_features: B * seq_len * self.embedding_dim
            # memory: B * seq_len * self.embedding_dim
            memory = self.encoder(fused_features)
            output_traj = self.final_gru(memory)  # B * (S * 3)
            return output_traj


# if __name__ == '__main__':
#     PositionalEncoding = MyModel.TransformerModel.PositionalEncoding(d_model=512, dropout=0.1, max_len=10)
#     TransformerModel = MyModel.TransformerModel.TransformerEgoStateModel(input_feature_dim=11)
#     source_input = torch.randn(1, 4, 11)
#     target = torch.randn(1, 8, 3)
#     output = TransformerModel(source_input, target)
#     print(output.size())
#     print(output)

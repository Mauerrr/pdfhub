import torch
import sys
sys.path.append('/media/yujie/data/E2EAD/endtoenddriving/')
import train.Yujie.yj_navsim_train as config
import torchvision.models as models
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

class MyEgoStateModel:

    class MLP(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
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

        def forward(self, x):
            return self.backbone(x)

    class RNN(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
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

        def forward(self, x):
            batch_size, _, _ = x.size()
            h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(config.device) # Initialize hidden state
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
            self.hidden_size = config.hidden_size
            self.n_layers = config.n_layers
            self.lstm = torch.nn.LSTM(input_size=config.n_feature_dim, hidden_size=self.hidden_size,
                                      num_layers=self.n_layers, batch_first=True)
            self.fc_lstm = torch.nn.Linear(in_features=config.hidden_size,
                                           out_features=config.n_output_dim * config.n_output_frames)

        def forward(self, x):
            # input x is: batch_size * sequence length * input dim
            batch_size, _, _ = x.size()
            
            #current_feature = torch.zeros(x.shape[0], 1, x.shape[2])
            #x = torch.cat((x[:, :-1, :] - x[:, 1:, :], current_feature), dim=1)
            
            h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(config.device)
            c0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(config.device)
            out, _ = self.lstm(x, (h0, c0))  # out -> [batch_size, sequence length, hidden_size]
            out = out[:, -1, :]  # only the last timestamp  # out -> [batch_size, hidden_size]
            # we want the final output be [batch_size, sequence_length, 3]
            out = self.fc_lstm(out)
            
            # #return torch.cumsum(out, dim=1)
            # if self.training:
            #     #print("train mode")
            #     return out
            # else:
            #     #print("val mode")
            #     # print('output size: ', out.size())
            #     # print(out)
            #     # print(torch.cumsum(out, dim=1))
            #     return torch.cumsum(out, dim=1)
            return out

    class GRU(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.hidden_size = config.hidden_size
            self.n_layers = config.n_layers
            self.gru = torch.nn.GRU(input_size=config.n_feature_dim, hidden_size=self.hidden_size,
                                    num_layers=self.n_layers, batch_first=True)
            self.fc_gru = torch.nn.Linear(in_features=config.hidden_size,
                                          out_features=config.n_output_dim * config.n_output_frames)

        def forward(self, x):
            batch_size, _, _ = x.size()
            h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(config.device)
            out, _ = self.gru(x, h0)  # out -> [batch_size, sequence length, hidden_size]
            out = out[:, -1, :]  # only the last timestamp  # out -> [batch_size, hidden_size]
            # we want the final output be [batch_size, sequence_length, 16]
            out = self.fc_gru(out)
            return out
        
    class TransformerModel:
        class PositionalEncoding(torch.nn.Module):
            def __init__(self, d_model: int, dropout: float):
                super().__init__()
                self.emb_size = d_model
                self.dropout = torch.nn.Dropout(dropout)

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
                self.input_embedding = torch.nn.Linear(input_feature_dim, hidden_dim)
                # input -> B * 8 * 3, output -> B * 8 * 512
                self.output_embedding = torch.nn.Linear(output_traj_dim, hidden_dim)

                self.positional_encoder = MyEgoStateModel.TransformerModel.PositionalEncoding(d_model=512, dropout=dropout)
                self.transformer = torch.nn.Transformer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    num_encoder_layers=num_encoder_layers,
                    num_decoder_layers=num_decoder_layers,
                    dropout=dropout,
                    batch_first=True,
                )

                self.out = torch.nn.Linear(hidden_dim, output_traj_dim)

                # val
                encoder_layers = TransformerEncoderLayer(d_model=self.dim_model, nhead=num_heads, dropout=dropout, batch_first=True)
                self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
                decoder_layers = TransformerDecoderLayer(d_model=self.dim_model, nhead=num_heads, dropout=dropout, batch_first=True)
                self.transformer_decoder = TransformerDecoder(decoder_layers, num_decoder_layers)

            def forward(self, src, tgt):
                if self.training:
                    # expect src and tgt both be B * S * E
                    """Shift left for train/ val"""
                    batch_size = src.size()[0]

                    # calculate the delta or not
                    if config.preprocess_choice == 'subtract_adjacent_row':
                        current_feature = torch.zeros(batch_size, 1, 3).to(config.device)
                        src[:, :, :3] = torch.cat((src[:, :-1, :3] - src[:, 1:, :3], current_feature), dim=1)

                    tgt = torch.cat([torch.zeros(batch_size, 1, self.target_feature_dim).to(config.device), tgt[:, :-1, :]],
                                    dim=1)

                    src = self.input_embedding(src) * math.sqrt(self.dim_model)  # input -> B * 4 * 11, output -> B * 4 * 512
                    tgt = self.output_embedding(tgt.float()) * math.sqrt(self.dim_model)  # input -> B * 8 * 3, output -> B * 8 * 512

                    src = self.positional_encoder(src)
                    tgt = self.positional_encoder(tgt)

                    tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt.size()[1]).to(
                        config.device)  # 生成和tgt_seq_len相同的方阵mask
                    out = self.transformer(src, tgt, tgt_mask=tgt_mask)
                    out = self.out(out)
                    #print('train')
                    return out
                else:
                    batch_size = src.size()[0]

                    # calculate the delta or not
                    if config.preprocess_choice == 'subtract_adjacent_row':
                        current_feature = torch.zeros(batch_size, 1, 3).to(config.device)
                        src[:, :, :3] = torch.cat((src[:, :-1, :3] - src[:, 1:, :3], current_feature), dim=1)

                    memory = self.transformer_encoder(self.positional_encoder(self.input_embedding(src) * math.sqrt(self.dim_model)))
                    y = torch.zeros(batch_size, 1, config.n_output_dim, dtype=torch.float, device=config.device)
                    for _ in range(config.n_output_frames):
                        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(y.size()[1]).to(
                            config.device)  # 生成和tgt_seq_len相同的方阵mask
                        out = self.transformer_decoder(self.positional_encoder(self.output_embedding(y) * math.sqrt(self.dim_model)), memory, tgt_mask)
                        out = self.out(out)

                        y = torch.cat((y, out[:, -1:, :]), dim=1)
                    #print('val')
                    return y[:, 1:, :]

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
                self.input_embedding = torch.nn.Linear(input_feature_dim, hidden_dim)
                # input -> B * 8 * 3, output -> B * 8 * 512
                self.output_embedding = torch.nn.Linear(output_traj_dim, hidden_dim)
                # Positional Encoding
                self.positional_encoder = MyEgoStateModel.TransformerModel.PositionalEncoding(d_model=512, dropout=dropout)
                # Encoder Layers
                encoder_layers = TransformerEncoderLayer(d_model=self.dim_model, nhead=num_heads, dropout=dropout, batch_first=True)
                self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
                # Decoder Layers
                decoder_layers = TransformerDecoderLayer(d_model=self.dim_model, nhead=num_heads, dropout=dropout, batch_first=True)
                self.transformer_decoder = TransformerDecoder(decoder_layers, num_decoder_layers)
                # Output Layer
                self.out = torch.nn.Linear(hidden_dim, output_traj_dim)

            def forward(self, src, inplace=None):

                batch_size = src.size()[0]

                # calculate the delta or not
                if config.preprocess_choice == 'subtract_adjacent_row':
                    current_feature = torch.zeros(batch_size, 1, 3).to(config.device)
                    src[:, :, :3] = torch.cat((src[:, :-1, :3] - src[:, 1:, :3], current_feature), dim=1)

                memory = self.transformer_encoder(self.positional_encoder(self.input_embedding(src) * math.sqrt(self.dim_model)))
                y = torch.zeros(batch_size, 1, config.n_output_dim, dtype=torch.float, device=config.device)
                for _ in range(config.n_output_frames):
                    tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(y.size()[1]).to(
                        config.device)  # 生成和tgt_seq_len相同的方阵mask
                    out = self.transformer_decoder(self.positional_encoder(self.output_embedding(y) * math.sqrt(self.dim_model)), memory, tgt_mask)
                    out = self.out(out)

                    y = torch.cat((y, out[:, -1:, :]), dim=1)

                return y[:, 1:, :]

class MyFusionModel:

    class ResNetBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # TODO: 可能还需要一些冻结weight的操作
            self.resnet50_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  # weights=models.ResNet50_Weights.DEFAULT
            # 冻结预训练的权重
            for param in self.resnet50_model.parameters():
                param.requires_grad = False

            num_features = self.resnet50_model.fc.in_features
            self.resnet50_model.fc = torch.nn.Identity()  # 去除原本的全连接层
            self.resnet50_model_final_fc = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=num_features, out_features=config.tf_embedding_dim)  # dim_model = 512
            )

        def forward(self, x):
            x = torch.nn.functional.interpolate(x / 255, scale_factor=0.25)
            x = self.resnet50_model(x.to(torch.float32))
            x = self.resnet50_model_final_fc(x)
            return x

    class ImageOnlyGRU(torch.nn.Module):
        def __init__(self, output_traj_dim=3, hidden_dim=config.tf_embedding_dim, dropout=0.1):
            super().__init__()
            self.model_type = "ImageOnlyUsingGRU"
            self.resnet = MyFusionModel.ResNetBackbone()
            self.images_gru = MyEgoStateModel.GRU(config.tf_embedding_dim).to(config.device)

        @staticmethod
        def preprocess_image_seq(images):
            # The input image shape is: B * seq_len * 1920 * 1080 * 3
            # The images_permuted shape become B * seq_len * 3 * 1920 * 1080
            batch_size = images.size(0)
            images_permuted = images.permute(0, 1, 4, 2, 3)
            # combine sequence with channels, (B * seq_len) * 3 * 1920 * 1080
            images_combined = images_permuted.view(-1, images_permuted.size(2), images_permuted.size(3),
                                                   images_permuted.size(4))
            return batch_size, images_combined

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
        def __init__(self, input_feature_dim=config.n_feature_dim, output_traj_dim=3, num_encoder_layers=1, num_decoder_layers=1,
                     hidden_dim=config.tf_embedding_dim, num_heads=8, dropout=0.1, target_seq_len=8):
            super().__init__()
            self.model_type = "ImageEgo"
            self.dim_model = hidden_dim
            self.target_feature_dim = output_traj_dim
            self.target_seq_len = target_seq_len

            self.resnet = MyFusionModel.ResNetBackbone()

            # input -> B * 4 * 11, output -> B * 4 * 512
            self.ego_state_embedding = torch.nn.Linear(input_feature_dim, hidden_dim)
            # input -> B * 8 * 3, output -> B * 8 * 512
            self.output_embedding = torch.nn.Linear(output_traj_dim, hidden_dim)

            self.positional_encoder = MyEgoStateModel.TransformerModel.PositionalEncoding(d_model=hidden_dim, dropout=dropout)

            encoder_layers = TransformerEncoderLayer(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            self.encoder = TransformerEncoder(encoder_layers, num_encoder_layers)

            decoder_layers = TransformerDecoderLayer(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            self.decoder = TransformerDecoder(decoder_layers, num_decoder_layers)

            self.out = torch.nn.Linear(hidden_dim, output_traj_dim)

        def forward(self, ego_state, images, tgt):
            """Get image features with PE: shape is B * seq_len * 1920 * 1080 * 3 --> B * seq_len * 512"""
            batch_size, preprocessed_images = MyFusionModel.ImageOnlyGRU.preprocess_image_seq(images)
            image_features_raw = self.resnet(preprocessed_images)
            images_features = MyFusionModel.ImageOnlyGRU.recover_output_features(batch_size, image_features_raw) * math.sqrt(self.dim_model)
            images_features_pe = self.positional_encoder(images_features)
            """Get ego features with PE: shape is B * seq_len * 11 --> B * seq_len * 512"""
            ego_state_features = self.ego_state_embedding(ego_state) * math.sqrt(self.dim_model)
            ego_state_features_pe = self.positional_encoder(ego_state_features)
            # 把feature相加
            fused_features = ego_state_features_pe + images_features_pe
            """Get target features"""
            tgt = torch.cat([torch.zeros(batch_size, 1, self.target_feature_dim).to(config.device), tgt[:, :-1, :]], dim=1)
            tgt = self.output_embedding(tgt.float()) * math.sqrt(self.dim_model)  # input -> B * 8 * 3, output -> B * 8 * 512
            tgt_feature_pe = self.positional_encoder(tgt)
            tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt.size()[1]).to(config.device)  # 生成和tgt_seq_len相同的方阵mask

            memory = self.encoder(fused_features)
            output = self.decoder(tgt_feature_pe, memory, tgt_mask)

            output = self.out(output)


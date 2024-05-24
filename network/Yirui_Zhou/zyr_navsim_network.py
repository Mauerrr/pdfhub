import torch
import train.Yujie.yj_navsim_train as config
import torchvision.models as models
import math


class MyModel:
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
            self.hidden_size = config.hidden_size
            self.n_layers = config.n_layers
            self.lstm = torch.nn.LSTM(input_size=config.n_feature_dim, hidden_size=self.hidden_size,
                                      num_layers=self.n_layers, batch_first=True)
            self.fc_lstm = torch.nn.Linear(in_features=config.hidden_size,
                                           out_features=config.n_output_dim * config.n_output_frames)

        def forward(self, x):
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

    class Resnet_backbone(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._init_resnet_model()

        def forward(self, images):
            images = images.reshape(-1, 3, config.image_height, config.image_width)
            return self.resnet_model(images)

        def resnet_model(self, x):
            x = self.resnet50_model(x)
            x = self.resnet50_model_final_fc(x)
            batch_size = x.size(0) // config.n_input_image_frames
            return x.view(batch_size, -1, config.embedding_dim)

        def _init_resnet_model(self):
            self.resnet50_model = models.resnet50(pretrained=True)
            # 修改第一个卷积层以适应输入图像大小（如果需要的话）
            self.resnet50_model.conv1 = torch.nn.Conv2d(3, 64, 7, 2, 3, bias=False)

            # 冻结 ResNet50 的预训练层
            for param in self.resnet50_model.parameters():
                param.requires_grad = False

            # 添加并训练新的全连接层
            self.resnet50_model_final_fc = torch.nn.Sequential(
                torch.nn.Linear(1000, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, config.embedding_dim)
            )

            # 确保新添加的层是可训练的
            for param in self.resnet50_model_final_fc.parameters():
                param.requires_grad = True

    class PositionalEncoding(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            """
            Args:
              d_model:      dimension of embeddings
              dropout:      randomly zeroes-out some of the input
              max_length:   max sequence length
            """
            self.d_model = config.embedding_dim
            self.dropout = config.dropout
            self.max_length = config.max_length
            # initialize dropout
            self.dropout_layer = torch.nn.Dropout(p=self.dropout)
            # create tensor of 0s
            self.pe = torch.zeros(self.max_length, self.d_model)
            # create position column
            k = torch.arange(0, self.max_length).unsqueeze(1)
            # calc divisor for positional encoding
            div_term = torch.exp(
                torch.arange(0, self.d_model, 2) * -(math.log(10000.0) / self.d_model)
            )
            # calc sine on even indices
            self.pe[:, 0::2] = torch.sin(k * div_term)
            # calc cosine on odd indices
            self.pe[:, 1::2] = torch.cos(k * div_term)
            # add dimension
            self.pe = self.pe.unsqueeze(0)
            # buffers are saved in state_dict but not trained by the optimizer
            self.register_buffer("pe", self.pe)

        def forward(self, x):
            """
            Args:
              x:        embeddings (batch_size, seq_length, d_model)

            Returns:
                        embeddings + positional encodings (batch_size, seq_length, d_model)
            """
            # add positional encoding to the embeddings
            x = x + self.pe[:, : x.size(1)].requires_grad_(False)
            # perform dropout
            output = self.dropout_layer(x)
            return output

    class TransformerModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.d_model = config.embedding_dim
            self.nhead = config.nhead
            self.nlayers = config.nlayers
            self.dropout = config.dropout
            self.model_type = 'Transformer'
            self.src_mask = None
            self.pos_encoder = MyModel.PositionalEncoding(self.d_model, self.dropout, 5000)
            self.ego_state_embedding = torch.nn.Linear(config.n_feature_dim, config.embedding_dim)
            encoder_layers = torch.nn.TransformerEncoderLayer(self.d_model, self.nhead, dropout=self.dropout,
                                                              batch_first=True)
            self.encoder = torch.nn.TransformerEncoder(encoder_layers, self.nlayers)
            decoder_layers = torch.nn.TransformerDecoderLayer(self.d_model, self.nhead, dropout=self.dropout,
                                                              batch_first=True)
            self.decoder = torch.nn.TransformerDecoder(decoder_layers, self.nlayers)
            self.backbone = MyModel.Resnet_backbone()
            self.tgt_embedding = torch.nn.Linear(config.n_feature_dim, config.embedding_dim)
            self.fc1 = torch.nn.Linear(config.embedding_dim, config.n_output_dim)
            self.lstm_decoder = torch.nn.LSTM(input_size=config.embedding_dim, hidden_size=config.embedding_dim * 2,
                                              num_layers=5, batch_first=True)
            self.fc2 = torch.nn.Linear(config.n_input_frames * config.embedding_dim,
                                       config.n_feature_dim * config.n_output_frames)
            self.gru_decoder = torch.nn.GRU(input_size=config.embedding_dim, hidden_size=config.embedding_dim * 2, num_layers=5, batch_first=True)

        def forward(self, images, ego_state, tgt):
            images = self.backbone(images)
            images = self.pos_encoder(images)
            coors = self.ego_state_embedding(ego_state)
            coors = self.pos_encoder(coors)
            # element-wise multiplication
            src = torch.mul(images, coors)
            tgt_len = tgt.size()[1]

            tgt_embedded = self.tgt_embedding(tgt)
            tgt = self.pos_encoder(tgt_embedded)

            device = tgt.device
            tgt_mask = torch.triu(
                torch.full((tgt_len, tgt_len), float('-inf'), dtype=torch.get_default_dtype(), device=device),
                diagonal=1)

            memory = self.encoder(src)
            output = self.transformer_decode(memory, tgt, tgt_mask)
            # output = self.lstm_decode(memory)
            # output = self.gru_decode(memory)
            return output

        def transformer_decode(self, memory, tgt, tgt_mask):
            output = self.decoder(tgt, memory, tgt_mask)
            output = self.fc1(output)
            return output

        def lstm_decode(self, memory):
            output, (h_0, c_0) = self.lstm_decoder(memory)
            bs = output.size()[0]
            output = output.view(bs, -1)
            output = self.fc2(output)
            output = output.view(bs, -1, config.n_output_dim)
            return output

        def gru_decode(self, memory):
            output, = self.gru_decoder(memory)
            bs = output.size()[0]
            output = output.view(bs, -1)
            output = self.fc2(output)
            output = output.view(bs, -1, config.n_output_dim)
            return output

    class FusionModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_type = 'FusionModel'
            self.d_model = config.embedding_dim
            self.nhead = config.nhead
            self.nlayers = config.nlayers
            self.dropout = config.dropout
            self.pos_encoder = MyModel.PositionalEncoding(self.d_model, self.dropout, 5000)
            self.ego_state_embedding = torch.nn.Linear(config.n_feature_dim, config.embedding_dim)
            self.mhd = torch.nn.MultiheadAttention(embed_dim=self.d_model, num_heads=self.nhead, dropout=self.dropout, batch_first=True)
            self.backbone = MyModel.Resnet_backbone()
            decoder_layers = torch.nn.TransformerDecoderLayer(self.d_model, self.nhead, dropout=self.dropout,
                                                              batch_first=True)
            self.decoder = torch.nn.TransformerDecoder(decoder_layers, self.nlayers)
            encoder_layers = torch.nn.TransformerEncoderLayer(self.d_model, self.nhead, dropout=self.dropout,
                                                              batch_first=True)
            self.encoder = torch.nn.TransformerEncoder(encoder_layers, self.nlayers)
            self.layer_norm = torch.nn.LayerNorm(self.d_model)
            self.fc1 = torch.nn.Linear(config.embedding_dim, config.n_output_dim)
            self.lstm_decoder = torch.nn.LSTM(input_size=config.embedding_dim, hidden_size=config.embedding_dim * 2,
                                              num_layers=5, batch_first=True)
            self.fc2 = torch.nn.Linear(config.n_input_frames * config.embedding_dim,
                                       config.n_feature_dim * config.n_output_frames)
            self.gru_decoder = torch.nn.GRU(input_size=config.embedding_dim, hidden_size=config.embedding_dim * 2,
                                            num_layers=5, batch_first=True)

        def forward(self, images, ego_states, tgt):
            # images: [batch_size, n_input_image_frames, 3, 224, 224]
            images = self.backbone(images)
            images = self.pos_encoder(images)
            # use ego_state_embedding to embed the ego_states, and use it as the query
            q = self.ego_state_embedding(ego_states)
            # use the images as the key and value
            k = images
            v = images

            tgt_len = tgt.size()[1]
            tgt_embedded = self.tgt_embedding(tgt)
            tgt = self.pos_encoder(tgt_embedded)
            device = tgt.device
            tgt_mask = torch.triu(
                torch.full((tgt_len, tgt_len), float('-inf'), dtype=torch.get_default_dtype(), device=device),
                diagonal=1)

            # use MultiheadAttention to get the memory
            memory = self.mhd(q, k, v)
            memory = self.dropout(memory)
            memory = self.layer_norm(memory)

            # decoder
            output = self.transformer_decode(memory, tgt, tgt_mask)
            # output = self.lstm_decode(memory)
            # output = self.gru_decode(memory)
            return output

        def lstm_decode(self, memory):
            output, (h_0, c_0) = self.lstm_decoder(memory)
            bs = output.size()[0]
            output = output.view(bs, -1)
            output = self.fc2(output)
            output = output.view(bs, -1, config.n_output_dim)
            return output

        def gru_decode(self, memory):
            output, h_n = self.gru_decoder(memory)
            bs = output.size()[0]
            output = output.view(bs, -1)
            output = self.fc2(output)
            output = output.view(bs, -1, config.n_output_dim)
            return output

        def transformer_decode(self, memory, tgt, tgt_mask):
            output = self.decoder(tgt, memory, tgt_mask)
            output = self.fc1(output)
            return output




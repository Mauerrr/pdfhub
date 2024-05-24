import torch
from zmq import device
import data_prep.Le_Cui.config as config
import math
import torch.nn.init as init
import torch.nn as nn

class UrbanDriverModel(torch.nn.Module):
    """
    Neural network model for urban driving.

    This model consists of fully connected layers with ReLU activations.
    The architecture is designed based on the input and output feature dimensions defined in the config file.
    """

    def __init__(self, model_type):
        """
        Initialize the UrbanDriverModel.

        Defines fully connected layers with appropriate input and output dimensions.
        """
        super().__init__()
        self.model_type = model_type
        print("The model used for ego state is: ", self.model_type)
        if self.model_type == 'fc':
            self._init_fc_model()
            self.model = self.fc_model
        elif self.model_type == 'lstm':
            self._init_lstm_model()
            self.model = self.lstm_model
        elif self.model_type == 'gru':
            self._init_gru_model()
            self.model = self.gru_model
        elif self.model_type == 'transformer':
            self._init_transformer_model()
            self.model = self.transformer_model
        else:
            raise ValueError("Invalid model_type. Choose between 'fc', 'lstm', 'gru', or 'transformer'.")


    def forward(self, x,y,tgtmask):
        """
        Forward pass of the neural network.

        Applies fully connected layers with ReLU activations.
        Reshapes the output tensor.

        :param x: Input tensor.
        :return: Output tensor.
        """
        return self.model(x,y,tgtmask)

    def _init_fc_model(self):
        """
        Initialize the fully connected layers model.

        Defines fully connected layers with appropriate input and output dimensions.
        """
        # Define fully connected layers
        self.fc1 = torch.nn.Linear(
            in_features=config.ego_state_feature_dim * config.ego_state_input_length,
            out_features=2 * config.ego_state_feature_dim * config.ego_state_target_length)
        self.fc2 = torch.nn.Linear(
            in_features=2 * config.ego_state_feature_dim * config.ego_state_target_length,
            out_features=config.trajectory_xy_dim * config.ego_state_target_length)

    def fc_model(self, x, y):
        """
        Define the fully connected layers model for ego states.

        :param x: Input tensor.
        :return: Output tensor.
        """
        batch_size, _, _ = x.size()
        # Flatten the input tensor
        x = x.view(batch_size, -1)
        # Apply fully connected layers with ReLU activations
        x = torch.relu(self.fc1(x))
        # TODO: Add the dropout layer
        # x = self.dropout(x)
        # Apply the final fully connected layer without activation
        x = self.fc2(x)
        # Reshape output tensor
        x = x.view(batch_size, -1, 2)
        return x

    def _init_lstm_model(self):
        self.hidden_size = config.hidden_size
        self.n_layers = config.n_layers
        self.lstm = torch.nn.LSTM(input_size=config.ego_state_feature_dim, hidden_size=self.hidden_size,
                                  num_layers=self.n_layers, batch_first=True)  
        self.fc_lstm = torch.nn.Linear(in_features=config.hidden_size,
                                       out_features=config.trajectory_xy_dim * config.ego_state_target_length)

    def lstm_model(self, x,y):
        # input x is: batch_size * sequence length * input dim
        batch_size, _, _ = x.size()
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(config.device)
        c0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(config.device)
        out, _ = self.lstm(x, (h0, c0))  # out -> [batch_size, sequence length, hidden_size]
        out = out[:, -1, :]  # only the last timestamp  # out -> [batch_size, hidden_size]
        # we want the final output be [batch_size, sequence_length, 16]
        out = self.fc_lstm(out)
        out = out.view(batch_size, -1, 2)
        return out

    def _init_rnn_model(self):
        self.hidden_size = config.hidden_size
        self.n_layers = config.n_layers
        # input dim = 2 (the input x, y coordinates)
        # hidden size: The number of features in the hidden state h
        # num_layers – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two RNNs together to
        # form a stacked RNN, with the second RNN taking in outputs of the first RNN and computing the final results.
        self.rnn = torch.nn.RNN(input_size=config.ego_state_feature_dim, hidden_size=self.hidden_size,
                                num_layers=self.n_layers, batch_first=True)
        self.fc_rnn = torch.nn.Linear(in_features=config.hidden_size,
                                       out_features=config.trajectory_xy_dim * config.ego_state_target_length)

    def rnn_model(self, x):
        batch_size, _, _ = x.size()
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(config.device)
        # The shape of x is: batch_size * sequence length * input size
        # the out containing the output features (h_t) from the last layer (the last of the 2 layers) of the RNN, for each t.
        out, _ = self.rnn(x, h0)  # out -> [batch_size, sequence length, hidden_size]
        out = out[:, -1, :]  # only the last timestamp  # out -> [batch_size, hidden_size]
        # we want the final output be [batch_size, sequence_length, 16]
        out = self.fc_rnn(out)
        out = out.view(batch_size, -1, 2)
        return out

    def _init_gru_model(self):
        self.hidden_size = config.hidden_size
        self.n_layers = config.n_layers
        self.gru = torch.nn.GRU(input_size=config.ego_state_feature_dim, hidden_size=self.hidden_size,
                                num_layers=self.n_layers, batch_first=True)
        self.fc_gru = torch.nn.Linear(in_features=config.hidden_size,
                                      out_features=config.trajectory_xy_dim * config.ego_state_target_length)

    def gru_model(self, x):
        batch_size, _, _ = x.size()
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(config.device)
        out, _ = self.gru(x, h0)  # out -> [batch_size, sequence length, hidden_size]
        out = out[:, -1, :]  # only the last timestamp  # out -> [batch_size, hidden_size]
        # we want the final output be [batch_size, sequence_length, 16]
        out = self.fc_gru(out)
        out = out.view(batch_size, -1, 2)
        return out
    
    def _init_transformer_model(self):
        """
        Initialize the Transformer model.
        """
        self.embedding = torch.nn.Linear(
            in_features=config.ego_state_feature_dim,
            out_features=config.d_model
        )
        self.transformer = torch.nn.Transformer(
            d_model=config.d_model,
            nhead=config.n_heads,
            num_encoder_layers=config.n_layers,
            num_decoder_layers=config.n_layers,
            dim_feedforward=config.hidden_size,
            dropout=config.dropout_prob,
            batch_first=True,
        )
        
        self.positional_encoder = PositionalEncoding(config.d_model,0.1)
        self.fc_transformer = torch.nn.Linear(
            in_features=config.d_model,
            out_features=config.ego_state_feature_dim
        )
        

    def transformer_model(self, x,y,tgtmask):
        """
        Define the Transformer model for ego states.

        :param x: Input tensor.
        :return: Output tensor.
        """
        # tgt_mask=torch.nn.Transformer.generate_square_subsequent_mask(config.ego_state_target_length).to(config.device)
        x_emb=self.embedding(x)
        y_emb=self.embedding(y)
        x_emb_pe = self.positional_encoder(x_emb)
        y_emb_pe = self.positional_encoder(y_emb)
        # x_emb_pe = x_emb
        # y_emb_pe = y_emb
        # print('y_after_pe',y[0])
     
        # y_shifted = torch.cat([torch.zeros(batch_size, 1, feature_dim).to(y.device), y[:, :-1, :]], dim=1)
        # y_shifted = torch.zeros_like(y_shifted)
        out = self.transformer(x_emb_pe, y_emb_pe,tgt_mask=tgtmask)
        # print('out1',out.size()) #128,8,2
        out = self.fc_transformer(out)
        # print('out1',out.size())
        #out = out.view(batch_size, -1, config.ego_state_feature_dim)
        # print('out1')
        return out
    def encode(self, src):
        # return self.transformer.encoder(self.embedding(src))
        return self.transformer.encoder(self.positional_encoder(self.embedding(src)))

    def decode(self, tgt, memory, tgt_mask):
        # return self.transformer.decoder(self.embedding(tgt), memory,tgt_mask)
        return self.transformer.decoder(self.positional_encoder(self.embedding(tgt)), memory,tgt_mask)


    

class PositionalEncoding(nn.Module):
    def __init__(self,
                emb_size: int,
                dropout: float):
        super(PositionalEncoding, self).__init__()
        self.emb_size = emb_size
        self.dropout = nn.Dropout(dropout)


    def forward(self, token_embedding):
        den = torch.exp(- torch.arange(0, self.emb_size, 2)* math.log(10000) / self.emb_size)
        pos = torch.arange(0, token_embedding.shape[1]).reshape(token_embedding.shape[1], 1)
        pos_embedding = torch.zeros((token_embedding.shape[1], self.emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0).to(token_embedding.device)
        # pos_embedding = self.pos_embedding.to(token_embedding.device)
        return self.dropout(token_embedding + pos_embedding[:token_embedding.size(0), :]).to(token_embedding.device)
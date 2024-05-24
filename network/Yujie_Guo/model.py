import torch
from torch import Tensor, lstm_cell
import math
import data_prep.Yujie_Guo.config as config


class UrbanDriverModel(torch.nn.Module):
    """
    Neural network model for urban driving.

    This model consists of fully connected layers with ReLU activations.
    The architecture is designed based on the input and output feature dimensions defined in the config file.
    """

    def __init__(self, model_type='fc'):
        """
        Initialize the UrbanDriverModel.

        Defines fully connected layers with appropriate input and output dimensions.
        """
        super().__init__()
        self.model_type = model_type
        self.preprocess_choice = config.preprocess_choice

        if self.preprocess_choice == 'subtract_adjacent_row_using_outputs':
            self.ego_state_input_length = config.ego_state_input_length - 1
        else:
            self.ego_state_input_length = config.ego_state_input_length

        print("The model used for ego state is: ", self.model_type)
        print("The method used for data is: ", self.preprocess_choice)
        
        if self.model_type == 'fc':
            self._init_fc_model()
            self.model = self.fc_model
        elif self.model_type == 'lstm':
            self._init_lstm_model()
            self.model = self.lstm_model
        elif self.model_type == 'lstmcell':
            self._init_lstmcell_model()
            self.model = self.lstmcell_model
        elif self.model_type == 'gru':
            self._init_gru_model()
            self.model = self.gru_model
        elif self.model_type == 'rnn':
            self._init_rnn_model()
            self.model = self.rnn_model
        #elif self.model_type == 'transformer':
            #self._init_transformer_model()
            self.model = self.transformer_model
        else:
            raise ValueError("Invalid model_type. Choose between 'fc','lstm','gru','rnn' or 'transformer'.")

    def forward(self, x):
        """
        Forward pass of the neural network.

        Applies fully connected layers with ReLU activations.
        Reshapes the output tensor.

        :param x: Input tensor.
        :return: Output tensor.
        """
        return self.model(x)

    def _init_fc_model(self):
        """
        Initialize the fully connected layers model.

        Defines fully connected layers with appropriate input and output dimensions.
        """
        # Define fully connected layers
        self.fc1 = torch.nn.Linear(
            in_features=config.ego_state_feature_dim * self.ego_state_input_length,
            out_features=2 * config.ego_state_feature_dim * config.ego_state_target_length)
        self.fc2 = torch.nn.Linear(
            in_features=2 * config.ego_state_feature_dim * config.ego_state_target_length,
            out_features=config.trajectory_xy_dim * config.ego_state_target_length)

    def fc_model(self, x):
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

    def lstm_model(self, x):
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

    def _init_lstmcell_model(self):
        self.hidden_size = config.hidden_size
        self.n_layers = config.n_layers
        self.lstmcell = torch.nn.LSTMCell(input_size=config.ego_state_feature_dim, hidden_size=self.hidden_size)  
        self.fc_lstmcell = torch.nn.Linear(in_features=config.hidden_size,
                                       out_features=config.trajectory_xy_dim * config.ego_state_target_length)

    def lstmcell_model(self, x):
        # input x is: batch_size * sequence length * input dim
        batch_size, sequence_length, _ = x.size()
        hx = torch.zeros(batch_size, self.hidden_size).to(config.device)
        cx = torch.zeros(batch_size, self.hidden_size).to(config.device)
        output = [] # used for recording hidden layer h

        for i in range(sequence_length):
            hx, cx = self.lstmcell(x[:, i, :], (hx, cx))  # x[:, i, :] is: batch_size * input dim
            # record output h
            output.append(hx)

        output = torch.stack(output, dim=1)  # output is: batch_size * sequence length * input dim
        
        out = output[:, -1, :]  # only the last timestamp  # out -> [batch_size, hidden_size]
        # we want the final output be [batch_size, sequence_length, 16]
        out = self.fc_lstmcell(out)
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

    # def _init_transformer_model(self):
    #     """
    #     Initialize the Transformer model.
    #     """
    #     self.input_embedding = torch.nn.Linear(
    #         in_features=config.ego_state_feature_dim,
    #         out_features=config.d_model
    #     )
    #     self.transformer = torch.nn.Transformer(
    #         d_model=config.d_model,
    #         nhead=config.n_heads,
    #         num_encoder_layers=config.n_layers,
    #         num_decoder_layers=config.n_layers,
    #         dim_feedforward=config.hidden_size,
    #         dropout=config.dropout_prob,
    #         batch_first=True,
    #     )
        
    #     self.positional_encoder = PositionalEncoder(config.d_model)
    #     self.fc_transformer = torch.nn.Linear(
    #         in_features=config.d_model*config.ego_state_target_length,
    #         out_features=config.ego_state_feature_dim*config.ego_state_target_length
    #     )
        
        #     # Initialize transformer parameters
        # for p in self.transformer.parameters():
        #     if p.dim() > 1:  # For weights
        #         init.xavier_uniform_(p)
        #     else:  # For biases
        #         init.zeros_(p)

        # # Initialize positional encoder parameters
        # for p in self.positional_encoder.parameters():
        #     if p.dim() > 1:  # For weights
        #         init.xavier_uniform_(p)
        #     else:  # For biases
        #         init.zeros_(p)

        # # Initialize linear layer parameters
        # init.xavier_uniform_(self.fc_transformer.weight)
        # init.zeros_(self.fc_transformer.bias)

#     def transformer_model(self, x,y):
#         """
#         Define the Transformer model for ego states.

#         :param x: Input tensor.
#         :return: Output tensor.
#         """
#         tgt_mask=torch.nn.Transformer.generate_square_subsequent_mask(config.ego_state_target_length).to(config.device)
#         # print('x',x[0])
#         x=self.input_embedding(x)
#         y=self.input_embedding(y)
#         #x = self.positional_encoder(x)
#         # print('x_after_pe',x[0])
#         # print('x.size()',x.size())
#         # print('y',y[0])
#         #y = self.positional_encoder(y)
#         # print('y_after_pe',y[0])
#         batch_size, seq_len, feature_dim = y.size()

#         y_shifted = torch.cat([torch.zeros(batch_size, 1, feature_dim).to(y.device), y[:, :-1, :]], dim=1)

#         # print('x.size',x.size())#128,16,2(batch_size, seq_len, feature_dim)
#         # Transformer requires input in the shape (batch_size, seq_len, feature_dim)
#         # print('y.size()',y.size())
#         out = self.transformer(x, y_shifted,tgt_mask=tgt_mask)
#         # print('out1',out.size()) #128,8,2
#         out = self.fc_transformer(out.reshape(batch_size, -1))
#         # print('out1',out.size())
#         out = out.view(batch_size, -1, config.ego_state_feature_dim)
#         # print('out1')
#         return out

# class PositionalEncoder(torch.nn.Module):
#     def __init__(self, d_model, max_len=config.ego_state_input_length):
#         super(PositionalEncoder, self).__init__()
#         self.d_model = d_model
#         self.max_len = max_len
#         self.positional_encoding = self.get_positional_encoding()

#     def get_positional_encoding(self):
#         positional_encoding = torch.zeros(self.max_len, self.d_model)
#         position = torch.arange(0, self.max_len).unsqueeze(1).float()
#         div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
#         positional_encoding[:, 0::2] = torch.sin(position * div_term)
#         positional_encoding[:, 1::2] = torch.cos(position * div_term)
#         positional_encoding = positional_encoding.unsqueeze(0)
#         return positional_encoding

#     def forward(self, x):
#         batch_size, seq_len, _ = x.size()
#         positional_encoding = self.positional_encoding[:, :seq_len, :]
#         return x + positional_encoding.to(x.device)
import torch
import data_prep.Zehao_Chen.config as config
import torchvision.models as models
import torch.nn as nn


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
        self.ego_state_model_type = model_type
        self._init_ego_model()
        # self._init_image_model()

    def forward(self, ego_state, images):
        """
        Forward pass of the neural network.

        Applies fully connected layers with ReLU activations.
        Reshapes the output tensor.

        :param ego_state: Input tensor.
        :param images: Input tensor.
        :return: Output tensor.

        """
        # self.image_model(images)
        # self.ego_model(ego_state)
        return self.ego_model(ego_state)

    def fc_model(self, x):
        """
        Define the fully connected layers model for ego states.

        :param x: Input tensor.
        :return: Output tensor.
        """
        batch_size, _, _ = x.size()
        # Flatten the input tensor
        x = x.view(batch_size, -1)
        x = self._mlp(x)
        # Reshape output tensor
        x = x.view(batch_size, -1, config.n_feature_dim)
        return x

    def image_model(self, x):
        """
        Define the fully connected layers model for ego states.

        :param x: Input tensor.
        :return: Output tensor.
        """
        batch_size = x.size()[0]
        x = self.resnet50_model(x)
        x = self.resnet50_model_final_fc(x)
        x = x.view(batch_size, -1, config.n_feature_dim)
        return x

    def _init_image_model(self):
        self.resnet50_model = models.resnet50(pretrained=True)
        self.resnet50_model.conv1 = nn.Conv2d(in_channels=config.n_input_image_frames * 3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = self.resnet50_model.fc.out_features
        self.resnet50_model_final_fc = torch.nn.Sequential(
            nn.Linear(in_features=num_features, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=16)
        )

    def _init_ego_model(self):
        print("The model used for ego state is: ", self.ego_state_model_type)
        print("The ego state feature dim is: ", config.n_feature_dim)
        if self.ego_state_model_type == 'fc':
            self._init_fc_model()
            self.ego_model = self.fc_model
        elif self.ego_state_model_type == 'lstm':
            self._init_lstm_model()
            self.ego_model = self.lstm_model
        elif self.ego_state_model_type == 'gru':
            self._init_gru_model()
            self.ego_model = self.gru_model
        elif self.ego_state_model_type == 'rnn':
            self._init_rnn_model()
            self.ego_model = self.rnn_model
        else:
            return
            # raise ValueError("Invalid model_type. Choose between 'fc' or 'lstm'.")

    def _init_fc_model(self):
        """
        Initialize the fully connected layers model.

        Defines fully connected layers with appropriate input and output dimensions.
        """
        # Define fully connected layers
        self._mlp = torch.nn.Sequential(
            torch.nn.Linear(config.n_input_frames * config.n_feature_dim, 2 * config.n_feature_dim * config.n_output_frames),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * config.n_feature_dim * config.n_output_frames, config.n_output_frames * config.n_output_dim),
        )

    def _init_lstm_model(self):
        self.hidden_size = config.hidden_size
        self.n_layers = config.n_layers
        self.lstm = torch.nn.LSTM(input_size=config.n_feature_dim, hidden_size=self.hidden_size,
                                  num_layers=self.n_layers, batch_first=True)
        self.fc_lstm = torch.nn.Linear(in_features=config.hidden_size,
                                       out_features=config.n_output_dim * config.n_output_frames)

    def lstm_model(self, x):
        # input x is: batch_size * sequence length * input dim
        batch_size, _, _ = x.size()
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(config.device)
        c0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(config.device)
        out, _ = self.lstm(x, (h0, c0))  # out -> [batch_size, sequence length, hidden_size]
        out = out[:, -1, :]  # only the last timestamp  # out -> [batch_size, hidden_size]
        # we want the final output be [batch_size, sequence_length, 16]
        out = self.fc_lstm(out)
        out = out.view(batch_size, -1, config.n_feature_dim)
        return out

    def _init_rnn_model(self):
        self.hidden_size = config.hidden_size
        self.n_layers = config.n_layers
        # input dim = 2 (the input x, y coordinates)
        # hidden size: The number of features in the hidden state h
        # num_layers – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two RNNs together to
        # form a stacked RNN, with the second RNN taking in outputs of the first RNN and computing the final results.
        self.rnn = torch.nn.RNN(input_size=config.n_feature_dim, hidden_size=self.hidden_size,
                                num_layers=self.n_layers, batch_first=True)
        self.fc_rnn = torch.nn.Linear(in_features=config.hidden_size,
                                      out_features=config.n_output_dim * config.n_output_frames)

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
        out = out.view(batch_size, -1, config.n_feature_dim)
        return out

    def _init_gru_model(self):
        self.hidden_size = config.hidden_size
        self.n_layers = config.n_layers
        self.gru = torch.nn.GRU(input_size=config.n_feature_dim, hidden_size=self.hidden_size,
                                num_layers=self.n_layers, batch_first=True)
        self.fc_gru = torch.nn.Linear(in_features=config.hidden_size,
                                      out_features=config.n_output_dim * config.n_output_frames)

    def gru_model(self, x):
        batch_size, _, _ = x.size()
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(config.device)
        out, _ = self.gru(x, h0)  # out -> [batch_size, sequence length, hidden_size]
        out = out[:, -1, :]  # only the last timestamp  # out -> [batch_size, hidden_size]
        # we want the final output be [batch_size, sequence_length, 16]
        out = self.fc_gru(out)
        out = out.view(batch_size, -1, config.n_feature_dim)
        return out

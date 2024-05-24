import torch
import torch.nn as nn
from torch import Tensor
import data_prep.Yujie_Guo.config_old as config


# modify based on Hao_Guo version
class SimpleMLP(nn.Module):
    """
    Neural network model for urban driving.

    This model consists of fully connected layers with ReLU activations.
    The architecture is designed based on the input and output feature dimensions defined in the config file.
    """

    def __init__(self):
        """
        Initialize the Model.

        Defines fully connected layers with appropriate input and output dimensions.
        """
        super().__init__()
        # Define fully connected layers
        self.fc1 = torch.nn.Linear(
            in_features=config.ego_state_feature_dim * config.ego_state_input_length,
            out_features=2 * config.ego_state_feature_dim * config.ego_state_target_length)
        self.fc2 = torch.nn.Linear(
            in_features=2 * config.ego_state_feature_dim * config.ego_state_target_length,
            out_features=config.trajectory_xy_dim * config.ego_state_target_length)
        # Define dropout layer
        # self.dropout = torch.nn.Dropout(p=0.5)  # Adjust dropout probability as needed


    def forward(self, x):
        """
        Forward pass of the neural network.

        Applies fully connected layers with ReLU activations.
        Reshapes the output tensor.

        :param x: Input tensor.
        :return: Output tensor.
        """
        batch_size, _, _ = x.size()
        # Flatten the input tensor
        x = x.view(batch_size, -1)
        # Apply fully connected layers with ReLU activations
        x = torch.relu(self.fc1(x))
        # Apply dropout
        # x = self.dropout(x) 
        # Apply the final fully connected layer without activation
        x = self.fc2(x)
        # Reshape output tensor
        x = x.view(batch_size, -1, 2)
        return x
    

class LSTMModel(nn.Module):
    def __init__(self, pretrained=True, 
                 input_n=config.ego_state_input_length, 
                 in_feature=config.ego_state_feature_dim, 
                 out_features=config.trajectory_xy_dim * config.ego_state_target_length) -> None:
        super().__init__()
        self.input_n = input_n
        self.in_feature = in_feature

        self.lstm = nn.LSTM(in_feature, 512, 2)
        self.linear = nn.Linear(512, out_features)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.in_feature)
        x = x.permute(1,0,2)
        x, (h, c) = self.lstm(x)
        x = x.permute(1,0,2)
        x = self.linear(x[:,-1])
        # Reshape output tensor
        x = x.view(batch_size, -1, 2)
        
        return x

# class LSTM(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.hidden_size = 512
#         self.n_layers = 2
#         self.lstm = torch.nn.LSTM(input_size=config.ego_state_feature_dim, hidden_size=self.hidden_size,
#                                   num_layers=self.n_layers, batch_first=True)
#         self.fc_lstm = torch.nn.Linear(in_features=self.hidden_size,
#                                        out_features=config.trajectory_xy_dim * config.ego_state_target_length)

#     def forward(self, x):
#         # input x is: batch_size * sequence length * input dim
#         batch_size, _, _ = x.size()
#         h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(config.device)
#         c0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(config.device)
#         out, _ = self.lstm(x, (h0, c0))  # out -> [batch_size, sequence length, hidden_size]
#         out = out[:, -1, :]  # only the last timestamp  # out -> [batch_size, hidden_size]
#         # we want the final output be [batch_size, sequence_length, 16]
#         out = self.fc_lstm(out)
#         out = out.view(batch_size, -1, 2)
#         return out


"""1. General"""
# run evaluation/submission的时候需要改回cpu，训练的时候使用GPU
device = 'cpu'
"""2. Input"""
# General
n_input_frames = 4
# Ego state
n_feature_dim = 11
n_driving_command = 4
# Images

# Lidar
max_height_lidar: float = 100.0
pixels_per_meter: float = 4.0
hist_max_per_pixel: int = 5

lidar_min_x: float = -32
lidar_max_x: float = 32
lidar_min_y: float = -32
lidar_max_y: float = 32

lidar_split_height: float = 0.2
use_ground_plane: bool = True

if not use_ground_plane:
    channel_lidar_feature = 1
else:
    channel_lidar_feature = 2

"""2. Output"""
n_output_frames = 8
n_output_dim = 3

"""3. Model"""
# FC
hidden_dim_fc = 512
# LSTM, GRU, RNN
hidden_size = 512
n_layers = 2
# Resnet
tf_embedding_dim = 512

"""4. Training & Evaluation"""
batch_size = 512
lr = 1e-4
betas = (0.9, 0.999)
eps = 1e-8
weight_decay = 0.
log_interval = 64
eval_freq = 64

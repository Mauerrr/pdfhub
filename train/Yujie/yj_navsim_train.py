"""1. General"""
# run evaluation/submission的时候需要改回cpu，训练的时候使用GPU
device = 'cuda'
"""2. Input"""
# General
n_input_frames = 4
# Ego state
n_feature_dim = 11 # 11
# Images
n_input_image_frames = 4
image_height = 224
image_width = 224
# Lidar
n_input_lidar_frames = n_input_image_frames
max_height_lidar: float = 100.0
pixels_per_meter: float = 4.0
hist_max_per_pixel: int = 5

lidar_min_x: float = -32
lidar_max_x: float = 32
lidar_min_y: float = -32
lidar_max_y: float = 32

lidar_split_height: float = 0.2
use_ground_plane: bool = True

if use_ground_plane == False:
    channel_lidar_feature = 1
else: 
    channel_lidar_feature = 2

# detection
num_bounding_boxes: int = 30
# Processing method
preprocess_choice = 'none'  # ['none', 'subtract_adjacent_row']

"""2. Output"""
n_output_frames = 8
n_output_dim = 3

"""3. Model"""
# FC
hidden_dim_fc = 512
# LSTM, GRU, RNN
hidden_size = 512
n_layers = 2
# Transformer for image
embedding_dim = 128
dropout = 0.1
max_length = 5000
nhead = 8
nlayers = 1
# Transformer for ego state
d_model=64
n_heads = 1
dropout_ego=0.1
# ResNet
tf_embedding_dim = 51
# Transformer
tf_d_model: int = 1024
tf_d_ffn: int = 1024
tf_num_layers: int = 3
tf_num_head: int = 8
tf_dropout: float = 0.0

# Transfuser
image_architecture: str = "resnet34"
lidar_architecture: str = "resnet18"

max_height_lidar: float = 100.0
pixels_per_meter: float = 4.0
hist_max_per_pixel: int = 5

lidar_min_x: float = -32
lidar_max_x: float = 32
lidar_min_y: float = -32
lidar_max_y: float = 32

camera_width: int = 720
camera_height: int = 360
lidar_resolution_width = 256
lidar_resolution_height = 256

img_vert_anchors: int = 8
img_horz_anchors: int = 8
lidar_vert_anchors: int = 8
lidar_horz_anchors: int = 8

block_exp = 4
n_layer = 2  # Number of transformer layers used in the vision backbone
n_head = 4
n_scale = 4
embd_pdrop = 0.1
resid_pdrop = 0.1
attn_pdrop = 0.1
# Mean of the normal distribution initialization for linear layers in the GPT
gpt_linear_layer_init_mean = 0.0
# Std of the normal distribution initialization for linear layers in the GPT
gpt_linear_layer_init_std = 0.02
# Initial weight of the layer norms in the gpt.
gpt_layer_norm_init_weight = 1.0

perspective_downsample_factor = 1
transformer_decoder_join = False
detect_boxes = True
use_bev_semantic = True
use_velocity = True
use_semantic = False
use_depth = False
add_features = True

# Transformer
tf_d_model: int = 256
tf_d_ffn: int = 1024
tf_num_layers: int = 3
tf_num_head: int = 8
tf_dropout: float = 0.0

# detection
num_bounding_boxes: int = 30

# loss weights
trajectory_weight: float = 10.0
agent_class_weight: float = 10.0
agent_box_weight: float = 1.0
bev_semantic_weight: float = 10.0

bev_pixel_width: int = lidar_resolution_width
bev_pixel_height: int = lidar_resolution_height // 2
bev_pixel_size: float = 0.25

num_bev_classes = 7
bev_features_channels: int = 64
bev_down_sample_factor: int = 4
bev_upsample_factor: int = 2

"""4. Training & Evaluation"""
batch_size = 8
lr = 1e-4
betas = (0.9, 0.999)
eps = 1e-8
weight_decay = 0.
log_interval = 310
eval_freq = 310


import os
# data preparation related parameters

# Get the Openscene path
current_file_path = os.path.abspath(__file__)
endtoenddriving_Path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))))
OpenScene_Path = os.path.join(os.path.dirname(endtoenddriving_Path), "OpenScene")

# mini set Images path
images_mini_root_path = os.path.join(endtoenddriving_Path, "navsim_workspace/dataset/sensor_blobs/mini")
images_test_root_path = os.path.join(endtoenddriving_Path, "navsim_workspace/dataset/sensor_blobs/test")
image_size = (224, 224)
image_height = 224
image_width = 224

train_csv_folder = os.path.join(endtoenddriving_Path, 'data/openscene/train')
val_csv_folder = os.path.join(endtoenddriving_Path, 'data/openscene/val')
overfit_csv_folder = os.path.join(endtoenddriving_Path, 'data/openscene/overfit')
test_csv_folder = os.path.join(endtoenddriving_Path, 'data/openscene/test')

# ego state hyperparameters TODO: Probably can be optimized later with Optuna
ego_xy = 2
ego_xy_theta = 3
n_input_frames = 4
n_output_frames = 8
n_feature_dim = ego_xy_theta
n_output_dim = ego_xy_theta
num_workers = 8
preprocess_choice = "transform_to_local_coor"  # ['none', 'transform_to_local_coor']

# images hyperparameters
n_input_image_frames = 4
use_8_camera = True  # False only use front camera
device = "cuda"

# training hyperparameters TODO: Probably can be optimized later with Optuna
lr = 0.0001
num_epochs = 1000
batch_size = 512
log_interval = 320
eval_freq = log_interval
save_freq = log_interval
betas = (0.9, 0.999)
eps = 1e-8
weight_decay = 0.

"""
    model parameters
"""

"""For ego state model:"""
model_type = "fc"  # ['fc', 'lstm', 'rnn', 'gru']， ['resnet']
# RNN LSTM hyperparameters
n_layers = 2
hidden_size = 512  # 512
# trained model path
trained_model = os.path.join(endtoenddriving_Path, 'train/Hao_Guo/best_model/model_best.ckpt')
# embedding dimension
embedding_dim = 128

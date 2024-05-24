import os

# data preparation related parameters
current_file_path = os.path.abspath(__file__)
endtoenddriving_Path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))))
OpenScene_Path = os.path.join(os.path.dirname(endtoenddriving_Path), "OpenScene")
# Navsim_Workspace_Path = os.path.join(os.path.dirname(endtoenddriving_Path), "navsim_workspace")
# trained model path
trained_model = os.path.join(endtoenddriving_Path, 'train/Yujie_Guo/best_model/model_best.ckpt')

# mini set Images path
images_train_path = os.path.join(endtoenddriving_Path, "dataset/sensor_blobs/mini")
images_val_path = os.path.join(endtoenddriving_Path, "dataset/sensor_blobs/test")
images_test_path = os.path.join(endtoenddriving_Path, "dataset/sensor_blobs/test")
image_size = (224, 224)
image_height = image_size[0]
image_width = image_size[1]

train_csv_folder = os.path.join(endtoenddriving_Path, 'data/openscene/train')
val_csv_folder = os.path.join(endtoenddriving_Path, 'data/openscene/val')
overfit_csv_folder = os.path.join(endtoenddriving_Path, 'data/openscene/overfit')
test_csv_folder = os.path.join(endtoenddriving_Path, 'data/openscene/val')

# ego state hyperparameters TODO: Probably can be optimized later with Optuna
ego_translation_xyz = 3
ego_translation_xy = 2
n_input_frames = 4
n_output_frames = 8
ego_state_feature_dim = ego_translation_xy
trajectory_xy_dim = 2
num_workers = 16
preprocess_choice = "subtract_adjacent_row"  # ['none', 'subtract_first_row', 'subtract_adjacent_row']

# images hyperparameters
n_input_image_frames = n_input_frames
use_8_camera = True  # False only use front camera
device = "cuda"

# training hyperparameters TODO: Probably can be optimized later with Optuna
lr = 0.0001
num_epochs = 1000
batch_size = 16
log_interval = 1280
eval_freq = log_interval
save_freq = log_interval
betas = (0.9, 0.999)
eps = 1e-8
weight_decay = 0.

"""
    model parameters
"""

"""For ego state model:"""
ego_state_model_part = "resnet"  # ['fc', 'lstm', 'rnn', 'gru']， ['resnet']
# RNN LSTM hyperparameters
n_layers = 2
hidden_size = 512  # 512
load_trained_model = False # True
# trained model path
trained_model = os.path.join(endtoenddriving_Path, 'train/Yujie_Guo/best_model/model_best_resnet.ckpt')
# embedding demension
embedding_dim = 128


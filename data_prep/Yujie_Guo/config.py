import os

# data preparation related parameters

# Get the Openscene path
current_file_path = os.path.abspath(__file__)
endtoenddriving_Path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))))
OpenScene_Path = os.path.join(os.path.dirname(endtoenddriving_Path), "OpenScene")

# mini set Images path
images_mini_root_path = os.path.join(OpenScene_Path, "dataset/openscene-v1.1/sensor_blobs/mini")
images_test_root_path = os.path.join(OpenScene_Path, "dataset/openscene-v1.1/sensor_blobs/test")
image_size = (224, 224)
image_height = 224
image_width = 224

# Get train/ val pkl file paths
train_pkl_path = os.path.join(OpenScene_Path, 'dataset/data', 'openscene_mini_train.pkl')
val_pkl_path = os.path.join(OpenScene_Path, 'dataset/data', 'openscene_mini_val.pkl')
overfit_pkl_path = os.path.join(OpenScene_Path, 'dataset/data', 'openscene_mini_overfit.pkl')
test_pkl_path = os.path.join(OpenScene_Path, 'dataset/data', 'openscene_test.pkl')

train_csv_folder = os.path.join(endtoenddriving_Path, 'data/openscene/train')
val_csv_folder = os.path.join(endtoenddriving_Path, 'data/openscene/val')
overfit_csv_folder = os.path.join(endtoenddriving_Path, 'data_prep/Yujie_Guo/data/openscene/overfit')
test_csv_folder = os.path.join(endtoenddriving_Path, 'data_prep/Yujie_Guo/data/openscene/test')

# train_csv_folder = os.path.join(endtoenddriving_Path, 'data_prep/Yujie_Guo/data/openscene/train')
# val_csv_folder = os.path.join(endtoenddriving_Path, 'data_prep/Yujie_Guo/data/openscene/val')
# overfit_csv_folder = os.path.join(endtoenddriving_Path, 'data_prep/Yujie_Guo/data/openscene/overfit')
# test_csv_folder = os.path.join(endtoenddriving_Path, 'data_prep/Yujie_Guo/data/openscene/test')
# ego state hyperparameters TODO: Probably can be optimized later with Optuna
ego_translation_xyz = 3
ego_translation_xy = 2
ego_state_input_length = 4
ego_state_target_length = 8
ego_state_feature_dim = ego_translation_xy
trajectory_xy_dim = 2
num_workers = 4
preprocess_choice = "subtract_first_row"  # ['subtract_adjacent_row_using_outputs', 'subtract_first_row']

device = "cuda"

# images hyperparameters
n_input_image_frames = 4
use_8_camera = True  # False only use front camera
device = "cuda"

# training hyperparameters TODO: Probably can be optimized later with Optuna
lr = 0.0001
num_epochs = 1000
batch_size = 128
eval_freq = 320
save_freq = eval_freq
log_interval = eval_freq
betas = (0.9, 0.999)
eps = 1e-8
weight_decay = 0.

"""
    model parameters
"""

"""For ego state model:"""
ego_state_model_part = "lstm"  # ['fc', 'lstm', 'lstmcell', 'rnn', 'gru', 'transformer']
# RNN LSTM hyperparameters
n_layers = 2
hidden_size = 512  # 512
# Transformer hyperparameters
n_heads = 1
d_model= 128
dropout_prob=0.1
# trained model path
trained_model = os.path.join(endtoenddriving_Path, 'train/Yujie_Guo/best_model/model_best.ckpt')

# tensorboard log path
tb_log = os.path.join(endtoenddriving_Path, 'train/Yujie_Guo')



import os
# modify based on Yujie_Guo version

# data preparation related parameters

# Get the Openscene path
current_file_path = os.path.abspath(__file__)
endtoenddriving_Path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))))
OpenScene_Path = os.path.join(os.path.dirname(endtoenddriving_Path), "OpenScene")
# Get train/ val pkl file paths
train_pkl_path = os.path.join(OpenScene_Path, 'dataset/data', 'openscene_mini_train.pkl')
val_pkl_path = os.path.join(OpenScene_Path, 'dataset/data', 'openscene_mini_val.pkl')
overfit_pkl_path = os.path.join(OpenScene_Path, 'dataset/data', 'openscene_mini_overfit.pkl')
test_pkl_path = os.path.join(OpenScene_Path, 'dataset/data', 'openscene_test.pkl')

train_csv_folder = os.path.join(endtoenddriving_Path, 'data/openscene/train')
val_csv_folder = os.path.join(endtoenddriving_Path, 'data/openscene/val')
overfit_csv_folder = os.path.join(endtoenddriving_Path, 'data/openscene/overfit')
test_csv_folder = os.path.join(endtoenddriving_Path, 'data/openscene/test')

# ego state parameters
ego_translation_xyz = 3
ego_translation_xy = 2
ego_state_input_length = 16
ego_state_target_length = 8
ego_state_feature_dim = ego_translation_xy
trajectory_xy_dim = 2

preprocess_choice = "subtract_first_row"  # ['none', 'subtract_first_row']

# training parameters
device = 'cuda'
lr = 0.0001
num_epochs = 1000
batch_size = 128
num_workers = 4
eval_freq = 320
save_freq = eval_freq
log_interval = eval_freq
betas = (0.9, 0.999)
eps = 1e-8
weight_decay = 0.
patience_num = 10

# trained model path
trained_model = os.path.join(endtoenddriving_Path, 'train/Yujie_Guo/output/model_best.ckpt')

# tensorboard log path
tb_log = os.path.join(endtoenddriving_Path, 'train/Yujie_Guo/runs')


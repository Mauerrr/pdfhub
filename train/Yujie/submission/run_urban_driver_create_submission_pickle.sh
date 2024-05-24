TEAM_NAME="Yujie2024"
AUTHORS="Yujie"
EMAIL="yujie.guo@tum.de"
INSTITUTION="TUM"
COUNTRY="Germany"

export PYTHONPATH=/media/yujie/data/E2EAD/endtoenddriving:$PYTHONPATH # 导出系统环境变量
python $NAVSIM_DEVKIT_ROOT/planning/script/run_create_submission_pickle.py \
agent=yj_image_lidar_agent \
split=private_test_e2e \
scene_filter=private_test_e2e \
experiment_name=submission_image_lidar_ego_state_formal \
team_name=$TEAM_NAME \
authors=$AUTHORS \
email=$EMAIL \
institution=$INSTITUTION \
country=$COUNTRY \
agent.checkpoint_path=/media/yujie/data/E2EAD/endtoenddriving/exp/training_urban_driver_pytorch/2024.05.21.11.04.39/pt_models/model_best_05_21_14_30.ckpt \

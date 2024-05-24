TEAM_NAME="SmallFish"
AUTHORS="Hao"
EMAIL="hao.guo@tum.de"
INSTITUTION="TUM"
COUNTRY="Germany"
export PYTHONPATH=/home/hguo/e2eAD/endtoenddriving:$PYTHONPATH # 导出系统环境变量
python $NAVSIM_DEVKIT_ROOT/planning/script/run_create_submission_pickle.py \
agent=gh_image_lidar_agent \
agent.checkpoint_path=/home/hguo/e2eAD/endtoenddriving/exp/model_best.ckpt \
split=mini \
scene_filter=warmup_test_e2e \
experiment_name=submission_warmup \
team_name=$TEAM_NAME \
authors=$AUTHORS \
email=$EMAIL \
institution=$INSTITUTION \
country=$COUNTRY \

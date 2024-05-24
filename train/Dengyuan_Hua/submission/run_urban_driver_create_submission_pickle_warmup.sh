TEAM_NAME="Dengyuan_Hua"
AUTHORS="Dengyuan_Hua"
EMAIL="dengyuan.hua@tum.de"
INSTITUTION="TUM"
COUNTRY="Germany"
export PYTHONPATH=/home/leohua/e2e/endtoenddriving/:$PYTHONPATH # 导出系统环境变量
python $NAVSIM_DEVKIT_ROOT/planning/script/run_create_submission_pickle.py \

agent=hua_navsim_agent \
split=mini \
scene_filter=warmup_test_e2e \
experiment_name=submission_cv_agent_warmup \
team_name=$TEAM_NAME \
authors=$AUTHORS \
email=$EMAIL \
institution=$INSTITUTION \
country=$COUNTRY \
agent.checkpoint_path=/home/leohua/e2e/endtoenddriving/exp/hua_training_urban_driver_pytorch/2024.05.13.16.40.05/pt_models/model_best_05_13_16_41.ckpt \




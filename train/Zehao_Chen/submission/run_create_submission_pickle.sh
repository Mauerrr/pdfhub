TEAM_NAME="Testzehao"
AUTHORS="Zehao"
EMAIL="chenzehao618@gmail.com"
INSTITUTION="TUM"
COUNTRY="Germany"
export PYTHONPATH=/home/hguo/e2eAD/endtoenddriving:$PYTHONPATH # 导出系统环境变量
python $NAVSIM_DEVKIT_ROOT/planning/script/run_create_submission_pickle.py \
agent=gh_navsim_agent \
agent.checkpoint_path=/home/hguo/e2eAD/endtoenddriving/exp/trainings/2024.05.23.20.53.52/pt_models/best_model.ckpt \
split=private_test_e2e \
scene_filter=private_test_e2e \
experiment_name=submission_formal \
team_name=$TEAM_NAME \
authors=$AUTHORS \
email=$EMAIL \
institution=$INSTITUTION \
country=$COUNTRY \

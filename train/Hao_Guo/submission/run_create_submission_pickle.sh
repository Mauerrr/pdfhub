TEAM_NAME="czhttt"
AUTHORS="Zehao"
EMAIL="chenzehao618@gmail.com"
INSTITUTION="TUM"
COUNTRY="Germany"

python $NAVSIM_DEVKIT_ROOT/planning/script/run_create_submission_pickle.py \
agent=gh_navsim_agent \
agent.checkpoint_path=/home/eze2szh/endtoenddriving/exp/trainings_gh_trans/2024.05.24.17.14.53/pt_models/model_best_05_24_17_23_nomodule.ckpt \
split=private_test_e2e \
scene_filter=private_test_e2e \
experiment_name=submission_formal524ghtrans \
team_name=$TEAM_NAME \
authors=$AUTHORS \
email=$EMAIL \
institution=$INSTITUTION \
country=$COUNTRY \

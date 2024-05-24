TEAM_NAME="Dengyuan_Hua"
AUTHORS="Dengyuan_Hua"
EMAIL="dengyuan.hua@tum.de"
INSTITUTION="TUM"
COUNTRY="Germany"

python $NAVSIM_DEVKIT_ROOT/planning/script/run_create_submission_pickle.py \
agent=urban_driver_agent \
split=private_test_e2e \
scene_filter=private_test_e2e \
experiment_name=submission_urban_driver_agent \
team_name=$TEAM_NAME \
authors=$AUTHORS \
email=$EMAIL \
institution=$INSTITUTION \
country=$COUNTRY \


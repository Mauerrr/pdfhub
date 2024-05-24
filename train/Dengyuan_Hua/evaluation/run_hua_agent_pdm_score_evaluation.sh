SPLIT=mini
CHECKPOINT="/home/leohua/e2e/endtoenddriving/exp/hua_training_urban_driver_pytorch/2024.05.09.15.48.57/pt_models/model_best_05_09_15_49.ckpt"

export PYTHONPATH=/home/leohua/e2e/endtoenddriving:$PYTHONPATH # 导出系统环境变量
python $NAVSIM_DEVKIT_ROOT/planning/script/run_pdm_score.py \
agent=hua_navsim_agent \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=evaluations \
split=$SPLIT \
metric_cache_path=/home/leohua/e2e/endtoenddriving/exp/metric_cache_mini \
scene_filter=warmup_test_e2e \

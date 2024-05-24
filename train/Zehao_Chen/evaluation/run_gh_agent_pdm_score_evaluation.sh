SPLIT=mini
CHECKPOINT="/home/hguo/e2eAD/endtoenddriving/exp/trainings/2024.05.07.13.57.17/pt_models/model_best.ckpt"

export PYTHONPATH=/home/hguo/e2eAD/endtoenddriving:$PYTHONPATH # 导出系统环境变量
python $NAVSIM_DEVKIT_ROOT/planning/script/run_pdm_score.py \
agent=gh_navsim_agent \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=evaluations \
split=$SPLIT \
metric_cache_path=/home/hguo/e2eAD/endtoenddriving/exp/all_metric_caches/metric_cache_mini \
scene_filter=warmup_test_e2e \

SPLIT=mini

export PYTHONPATH=/home/hguo/e2eAD/endtoenddriving:$PYTHONPATH # 导出系统环境变量
python $NAVSIM_DEVKIT_ROOT/planning/script/run_metric_caching.py \
split=$SPLIT \
cache.cache_path=$NAVSIM_EXP_ROOT/all_metric_caches/metric_cache_mini_camera \

SPLIT=mini

export PYTHONPATH=/home/hguo/e2eAD/endtoenddriving:$PYTHONPATH # 导出系统环境变量
python $NAVSIM_DEVKIT_ROOT/planning/script/run_dataset_caching.py \
split=$SPLIT \
agent=gh_image_lidar_agent \
dataloader.params.batch_size=1 \
dataloader.params.num_workers=4 \
cache_path=$NAVSIM_EXP_ROOT/all_training_caches/mini_cache_all_datas \
experiment_name=cache_data

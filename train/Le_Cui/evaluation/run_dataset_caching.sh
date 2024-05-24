SPLIT=mini

export PYTHONPATH=/media/yujie/data/E2EAD/endtoenddriving:$PYTHONPATH # 导出系统环境变量
python $NAVSIM_DEVKIT_ROOT/planning/script/run_dataset_caching.py \
split=$SPLIT \
agent=yj_image_lidar_agent \
dataloader.params.batch_size=1 \
dataloader.params.num_workers=2 \
cache_path=$NAVSIM_EXP_ROOT/all_training_caches/transfuser_agent_cache_mini_all \
experiment_name=cache_data

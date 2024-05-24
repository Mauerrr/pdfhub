python yj_run_pytorch_training.py \
agent=yj_image_lidar_agent \
experiment_name=training_urban_driver_pytorch \
trainer.params.max_epochs=1000 \
split=mini \
cache_path=/media/yujie/data/E2EAD/endtoenddriving/exp/all_training_caches/transfuser_agent_cache_mini_all/ \
use_cache_without_dataset=true \
# cache_path=/data/tumdriving/E2EAD/endtoenddriving/exp/all_training_caches/trainval_cache_all/ \
# use_cache_without_dataset=true \
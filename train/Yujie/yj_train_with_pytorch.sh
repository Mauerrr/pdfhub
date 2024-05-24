python yj_run_pytorch_training.py \
agent=yj_navsim_agent \
experiment_name=training_yj_navsim_agent \
trainer.params.max_epochs=200 \
split=trainval \
scene_filter=all_scenes \
cache_path=/home/eze2szh/endtoenddriving/exp/training_cache \
use_cache_without_dataset=true \
# cache_path=/data/tumdriving/E2EAD/endtoenddriving/exp/all_training_caches/trainval_cache_all/ \
# use_cache_without_dataset=true \
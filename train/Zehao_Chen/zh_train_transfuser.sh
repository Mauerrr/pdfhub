python zh_run_transfuser_training.py \
agent=transfuser_agent \
experiment_name=trainings \
trainer.params.max_epochs=100 \
split=mini \
dataloader.params.batch_size=8 \
dataloader.params.num_workers=8 \
scene_filter=all_scenes \
cache_path=/media/yujie/data/E2EAD/endtoenddriving/exp/training_cache/ \
use_cache_without_dataset=true \

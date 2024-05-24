python train1.py \
agent=xzt_agent \
experiment_name=xzt_train_x \
trainer.params.max_epochs=200 \
split=trainval \
dataloader.params.batch_size=256 \
dataloader.params.num_workers=4 \
scene_filter=all_scenes \
cache_path=/mnt/data/exp/training_cache \
use_cache_without_dataset=ture \


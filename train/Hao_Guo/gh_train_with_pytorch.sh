python gh_run_pytorch_training.py \
agent=gh_navsim_agent \
experiment_name=trainings_gh_trans \
trainer.params.max_epochs=200 \
split=trainval \
dataloader.params.batch_size=256 \
dataloader.params.num_workers=8 \
scene_filter=all_scenes \
cache_path=/home/eze2szh/endtoenddriving/exp/training_cache \
use_cache_without_dataset=true \


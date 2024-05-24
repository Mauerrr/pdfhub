python zh_run_pytorch_training.py \
agent=gh_navsim_agent \
experiment_name=trainings \
trainer.params.max_epochs=200 \
split=trainval \
dataloader.params.batch_size=512 \
dataloader.params.num_workers=8 \
scene_filter=all_scenes \
cache_path=/home/hguo/e2eAD/endtoenddriving/exp/all_training_caches/training_cache_intervalnull \
use_cache_without_dataset=true \


python /home/eze2szh/endtoenddriving/train/Boyang_Zhong/run_pytorch_training.py \
agent=egoagent_xy \
experiment_name=training_ego_xy \
trainer.params.max_epochs=200 \
split=trainval \
scene_filter=all_scenes \
cache_path=/home/eze2szh/endtoenddriving/exp/training_cache \
use_cache_without_dataset=true \
import torch

# 假设 state_dict 是从保存的模型中加载的状态字典
state_dict = torch.load('/home/eze2szh/endtoenddriving/exp/trainings_gh_trans/2024.05.24.17.14.53/pt_models/model_best_05_24_17_23.ckpt')

# 移除每个键名中的 'module.' 前缀
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

# 保存修改后的状态字典到新的文件
torch.save(new_state_dict, '/home/eze2szh/endtoenddriving/exp/trainings_gh_trans/2024.05.24.17.14.53/pt_models/model_best_05_24_17_23_nomodule.ckpt')


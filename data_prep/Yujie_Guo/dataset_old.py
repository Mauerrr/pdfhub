
from cProfile import label
import pickle
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# import config # used for test
import data_prep.Yujie_Guo.config as config

# modify based on Hao_Guo version
class OpensceneDataset(Dataset):
    """
    Run collect_data.py firstly to create raw pkl file for training and validation data
    """
    def __init__(self, split):
        """
        Initialize the NuplanDataset.

        Args:
            split (str): Specifies whether to load the training or validation split.
        """
        assert split in ['train', 'val', 'test', 'overfit']
        if split == 'train':
            data_path = config.train_pkl_path
        elif split == 'val':
            data_path = config.val_pkl_path
        elif split == 'test':
            data_path = config.test_pkl_path
        elif split == 'overfit':
            data_path = config.overfit_pkl_path
        else:
            raise ValueError("The split must be 'train', 'val', 'test', or 'overfit'")

        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        pkl_length = len(self.data)
        print(f"The {split} pkl file length is:", pkl_length)

        self.input = []
        self.target = []
        for i in range(len(self.data)):#len(self.data)
            if i + config.ego_state_input_length + config.ego_state_target_length < pkl_length:
                start_log_token = self.data[i]["scene_token"]
                end_log_token = self.data[i + config.ego_state_input_length + config.ego_state_target_length - 1][
                    "scene_token"]
                # TODO: extend to accept Images/ PointClouds
                if end_log_token == start_log_token:  
                    # TODO: Use x, y, z or only x, y, or also include quaternion?
                    input_data = [torch.tensor(item['ego2global_translation'][:config.ego_translation_xy], dtype=torch.float64) for item in
                                  self.data[i:i + config.ego_state_input_length]]
                    #print(input_data)
                    ego_state_at_current_timestamps = input_data[-1]
                    target_data = [torch.tensor(item['ego2global_translation'][:config.ego_translation_xy], dtype=torch.float64) for item in
                                   self.data[
                                   i + config.ego_state_input_length: i + config.ego_state_input_length + config.ego_state_target_length]]
                    self.input.append(self.ego_state_preprocess(torch.stack(input_data), ego_state_at_current_timestamps, config.preprocess_choice, split))
                    self.target.append(self.ego_state_preprocess(torch.stack(target_data), ego_state_at_current_timestamps, config.preprocess_choice, split))

        if len(self.input) != len(self.target):
            raise ValueError("The input and target must have the same length.")
        else:
            self.n_samples = len(self.input)
            print(f"The {split} set has {self.n_samples} samples")


    def __len__(self):
        """
        Get the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return self.n_samples


    def __getitem__(self, idx):
        """
        Get a sample from the dataset at the specified index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the input and target tensors.
        """
        return self.input[idx], self.target[idx]
        
    @staticmethod
    def ego_state_preprocess(tensor, current_state, preprocess_choice, split):
        """
        Preprocess ego state tensor based on the preprocessing choice.

        Args:
            tensor (torch.Tensor): Ego state tensor.
            current_state (torch.Tensor): Current state tensor.
            preprocess_choice (str): Preprocessing choice ('none' or 'subtract_first_row').
            split (str): Dataset split ('train', 'val', 'test', or 'overfit').

        Returns:
            torch.Tensor: Preprocessed ego state tensor.
        """
        assert preprocess_choice in ['none', 'subtract_first_row']
        # TODO: need to be improved later to also accept preprocessed data
        # if the split equals test, then returns the raw data for easier inference
        if split == 'test':
            return tensor
        if preprocess_choice == 'subtract_first_row':
            return (tensor - current_state).to(torch.float32) # change to float32 to reduce memory consumption during training
        elif preprocess_choice == 'none':
            return tensor
        else:
            raise ValueError("The preprocess_choice must be 'none' or 'subtract_first_row'")

    @staticmethod
    def trajectory_recovery(input_tensor, prediction, preprocess_choice):
        """
        Recover the trajectory based on the preprocessing choice.

        Args:
            input_tensor (torch.Tensor): Input tensor.
            prediction (torch.Tensor): Prediction tensor.
            preprocess_choice (str): Preprocessing choice ('none' or 'subtract_first_row').

        Returns:
            torch.Tensor: Recovered trajectory tensor.
        """
        assert preprocess_choice in ['none', 'subtract_first_row']
        if preprocess_choice == 'subtract_first_row':
            pred_global_way_points = prediction + input_tensor[-1]
        elif preprocess_choice == 'none':
            pred_global_way_points = prediction
        else:
            raise ValueError("The preprocess_choice must be 'none' or 'subtract_first_row'")
        return pred_global_way_points


# # 测试
# if __name__=='__main__':

#     train_data = OpensceneDataset('train')

#     train_dataloader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    
#     for i, batch in enumerate(train_dataloader):
#         input_data, target_labels = batch

#     print(input_data)
#     print(target_labels)
#     print("Size of the input: ", input_data.size())
#     print("Size of the label: ", target_labels.size())
    




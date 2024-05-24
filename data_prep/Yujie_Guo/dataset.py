import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
from PIL import Image
#import config # used for test
import data_prep.Yujie_Guo.config as config
from torchvision.transforms import transforms
import os
import pandas as pd


class NuplanDataset(Dataset):
    """
    Dataset class for Nuplan data.

    This class loads and prepares the Nuplan dataset for training or validation. Remember to run collect_data.py to
    create raw data.
    """

    def __init__(self, split):
        """
        Initialize the NuplanDataset.

        Args:
            split (str): Specifies whether to load the training or validation split.
        """
        assert split in ['train', 'val', 'test', 'overfit']
        if split == 'train':
            data_path = config.train_csv_folder
            image_root_path = config.images_mini_root_path
        elif split == 'val':
            data_path = config.val_csv_folder
            image_root_path = config.images_test_root_path
        elif split == 'test':
            data_path = config.test_csv_folder
            image_root_path = config.images_mini_root_path
        elif split == 'overfit':
            data_path = config.overfit_csv_folder
            image_root_path = config.images_mini_root_path
        else:
            raise ValueError("The split must be 'train', 'val', 'test', or 'overfit'")
        
        print(f"The {split} set has {len(os.listdir(data_path))} different scenes")
        # 0 - 7: cameras [CAM_F0, CAM_L0, CAM_R0, CAM_L1, CAM_R1, CAM_L2, CAM_R2, CAM_B0]
        # 8 - 10: ego2global_translation x, y, z
        # 11 - 14： ego2global ego2global_rotation q0 q1 q2 q3
        # 15 - 17： lidar2ego_translation x, y, z
        # 18 - 21: lidar2ego_rotation
        # location in whole image
        self.paste_locations = np.array([[1, 0], [0, 0], [2, 0], [0, 1], [2, 1], [0, 2], [1, 2], [2, 2]]) * \
                               config.image_size[0]
        self.rotate_degree_images = np.array([0, 0, 0, 90, -90, 180, 180, 180])

        self.images_past = []
        self.coors_past = []
        self.coors_future = []

        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

        self.transform = transforms.Compose([
            transforms.Resize(config.image_size),  # Image size
            transforms.ToTensor(),
            # normalize
        ])

        self.white_image_tensor = self.transform(Image.new("RGB", config.image_size, (255, 255, 255)))


        csv_files = os.listdir(data_path)
        # each csv file is a different scene
        for csv_file in csv_files:
            csv_file_path = os.path.join(data_path, csv_file)
            #csv_file_path = os.path.join(data_path, "log-0039-scene-0001.csv")
            # read csv file from the csv path
            df = pd.read_csv(csv_file_path, header=None)
            len_csv = len(df)
            # print(f"The length of the scene {csv_file} is {len_csv}")
            if len_csv < config.ego_state_input_length + config.ego_state_target_length:
                continue
            else:#
                for start_index in range(len(df) - config.ego_state_input_length - config.ego_state_target_length + 1):
                    raw_input = df.iloc[start_index:start_index + config.ego_state_input_length]
                    raw_target = df.iloc[start_index + config.ego_state_input_length: start_index + config.ego_state_input_length + config.ego_state_target_length]
                    ego_state_input_raw = torch.tensor(raw_input.iloc[:, 8:10].values, dtype=torch.float64)
                    ego_state_target_raw = torch.tensor(raw_target.iloc[:, 8:10].values, dtype=torch.float64)
                    #print(ego_state_input_raw)
                    ego_state_input, ego_state_target = self.ego_state_preprocess(ego_state_input_raw, ego_state_target_raw, config.preprocess_choice, split)
                    # change to float32 to reduce memory consumption during training
                    ego_state_input = ego_state_input.to(torch.float32)
                    ego_state_target = ego_state_target.to(torch.float32)
                    self.coors_past.append(ego_state_input)
                    self.coors_future.append(ego_state_target)
                    
        if len(self.coors_past) != len(self.coors_future):
            raise ValueError("The input and target must have the same length.")
        else:
            self.n_samples = len(self.coors_past)
            print(f"The {split} set has {self.n_samples} samples")

    def __getitem__(self, index):
        """
        Get a sample from the dataset at the specified index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the input and target tensors.
        """

        return self.coors_past[index], self.coors_future[index]

    def __len__(self):
        """
        Get the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return self.n_samples

    @staticmethod
    def ego_state_preprocess(raw_input, raw_target, preprocess_choice, split):
        """
        Preprocess ego state tensor based on the preprocessing choice.

        Args:
            raw_input: The raw ego state data for the input
            raw_target: The raw ego state data for the target
            preprocess_choice (str): Preprocessing choice ('subtract_adjacent_row_using_outputs' or 'subtract_first_row').
            split (str): Dataset split ('train', 'val', 'test', or 'overfit').

        Returns:
            processed_input: The processed data for input
            processed_target: The processed data for target
        """
        assert preprocess_choice in ['subtract_adjacent_row_using_outputs', 'subtract_first_row']
        # TODO: need to be improved later to also accept preprocessed data
        # if the split equals test, then returns the raw data for easier inference
        if split == 'test':
            return raw_input, raw_target
        if preprocess_choice == 'subtract_first_row':
            # process the ego state x, y
            return raw_input - raw_input[-1], raw_target - raw_input[-1]
        elif preprocess_choice == 'subtract_adjacent_row_using_outputs':
            # The input is to subtract adjacent numbers
            return raw_input[:-1] - raw_input[1:], raw_target - raw_input[-1]
        else:
            raise ValueError("The preprocess_choice must be 'subtract_adjacent_row_using_outputs' or 'subtract_first_row'")

    @staticmethod
    def trajectory_recovery(input_tensor, prediction, preprocess_choice):
        """
        Recover the trajectory based on the preprocessing choice.

        Args:
            input_tensor (torch.Tensor): Input tensor.
            prediction (torch.Tensor): Prediction tensor.
            preprocess_choice (str): Preprocessing choice ('subtract_adjacent_row_using_outputs' or 'subtract_first_row').

        Returns:
            torch.Tensor: Recovered trajectory tensor.
        """
        assert preprocess_choice in ['subtract_adjacent_row_using_outputs', 'subtract_first_row']
        if preprocess_choice == 'subtract_first_row':
            pred_global_way_points = prediction + input_tensor[-1]
        elif preprocess_choice == 'subtract_adjacent_row_using_outputs':
            pred_global_way_points = prediction
        else:
            raise ValueError("The preprocess_choice must be 'subtract_adjacent_row_using_outputs' or 'subtract_first_row'")
        return pred_global_way_points
    
    def load_image_as_tensor(self, image_paths):
        """Load image from image_path as numpy array"""
        # 0 - 7: cameras [CAM_F0, CAM_L0, CAM_R0, CAM_L1, CAM_R1, CAM_L2, CAM_R2, CAM_B0]
        multi_frame_images = []
        for frame_paths in image_paths:
            big_image = Image.new('RGB', (config.image_size[0] * 3, config.image_size[0] * 3), color=(0, 0, 0))
            for i, path in enumerate(frame_paths):
                if config.use_8_camera:
                    # with rotation:
                    big_image.paste(Image.open(path).resize(config.image_size).rotate(self.rotate_degree_images[i]),
                                    (self.paste_locations[i, 0], self.paste_locations[i, 1]))
                    # without rotation
                    # big_image.paste(Image.open(path).resize(config.image_size),
                    #                 (self.paste_locations[i, 0], self.paste_locations[i, 1]))
                else:
                    image = self.transform(Image.open(path))
                    multi_frame_images.append(image)
            if config.use_8_camera:
                # Commenting out the matplotlib display line to prevent image from showing
                # plt.imshow(big_image)
                # plt.show()  # This line is responsible for showing the image
                multi_frame_images.append(self.transform(big_image))
        return torch.cat(multi_frame_images, dim=0) 

# if __name__=='__main__':

#     train_data = NuplanDataset('train')

#     train_dataloader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    
#     for i, batch in enumerate(train_dataloader):
#         input_data, target_labels = batch

#     print(input_data)
#     print(target_labels)
#     print("Size of the input: ", input_data.size())
#     print("Size of the label: ", target_labels.size())
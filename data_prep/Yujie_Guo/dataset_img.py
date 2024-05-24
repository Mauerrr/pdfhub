import torch
from torch.utils.data import Dataset
import pickle
import data_prep.Yujie_Guo.config_img as config
import os
import pandas as pd
from PIL import Image
import numpy as np
from torchvision.transforms import transforms
import matplotlib.pyplot as plt


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
            images_root_path = config.images_train_path
        elif split == 'val':
            data_path = config.val_csv_folder
            images_root_path = config.images_val_path
        elif split == 'test':
            data_path = config.test_csv_folder
            images_root_path = config.images_test_path
        elif split == 'overfit':
            data_path = config.overfit_csv_folder
        else:
            raise ValueError("The split must be 'train', 'val', 'test', or 'overfit'")

        print(f"The {split} set has {len(os.listdir(data_path))} different scenes")
        # 0 - 7: cameras [CAM_F0, CAM_L0, CAM_R0, CAM_L1, CAM_R1, CAM_L2, CAM_R2, CAM_B0]
        # location in whole image
        self.paste_locations = np.array([[1, 0], [0, 0], [2, 0], [0, 1], [2, 1], [0, 2], [1, 2], [2, 2]]) * config.image_size[0]
        self.rotate_degree_images = np.array([0, 0, 0, 90, -90, 180, 180, 180])
        # 8 - 10: ego2global_translation x, y, z
        # 11 - 14： ego2global ego2global_rotation q0 q1 q2 q3
        # 15 - 17： lidar2ego_translation x, y, z
        # 18 - 21: lidar2ego_rotation
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
            # csv_file_path = os.path.join(data_path, "log-0001-scene-0001.csv")
            # read csv file from the csv path
            df = pd.read_csv(csv_file_path, header=None)
            len_csv = len(df)
            # print(f"The length of the scene {csv_file} is {len_csv}")
            if len_csv < config.n_input_frames + config.n_output_frames:
                continue
            else:
                for start_index in range(len(df) - config.n_input_frames - config.n_output_frames + 1):
                    # Get the raw input and raw target (both including images, global coordinates information)
                    raw_input = df.iloc[start_index:start_index + config.n_input_frames]
                    raw_target = df.iloc[
                                 start_index + config.n_input_frames: start_index + config.n_input_frames + config.n_output_frames]

                    images_past_paths = raw_input.iloc[-config.n_input_image_frames:,
                                        0:8 if config.use_8_camera else 1].values.tolist()
                    # print(images_past_paths)
                    # Change to absolute path, helpful later when using getitem function
                    for frame_index, frame_paths in enumerate(images_past_paths):
                        for i, path in enumerate(frame_paths):
                            images_past_paths[frame_index][i] = os.path.join(images_root_path, path)

                    self.images_past.append(images_past_paths)
                    # Prepare processed past coordinates and future coordinates (target) for training
                    coors_past_raw = torch.tensor(raw_input.iloc[:, 8:10].values, dtype=torch.float64)
                    coors_future_raw = torch.tensor(raw_target.iloc[:, 8:10].values, dtype=torch.float64)
                    coors_past_processed, coors_future_processed = self.ego_state_preprocess(coors_past_raw,
                                                                                             coors_future_raw,
                                                                                             config.preprocess_choice,
                                                                                             split)
                    # change to float32 to reduce memory consumption during training
                    coors_past_processed = coors_past_processed.to(torch.float32)
                    coors_future_processed = coors_future_processed.to(torch.float32)
                    self.coors_past.append(coors_past_processed)
                    self.coors_future.append(coors_future_processed)

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
        images_past_paths = self.images_past[index]
        return {"images": self.load_image_as_tensor(images_past_paths), "ego_states": self.coors_past[index],
                "future_waypoints": self.coors_future[index]}

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
            preprocess_choice (str): Preprocessing choice ('none' or 'subtract_first_row').
            split (str): Dataset split ('train', 'val', 'test', or 'overfit').

        Returns:
            processed_input: The processed data for input
            processed_target: The processed data for target
        """
        assert preprocess_choice in ['none', 'subtract_first_row', 'subtract_adjacent_row']
        # TODO: need to be improved later to also accept preprocessed data
        # if the split equals test, then returns the raw data for easier inference
        if split == 'test':
            #return raw_input, raw_target
            return raw_input - raw_input[-1], raw_target - raw_input[-1]
        if preprocess_choice == 'subtract_first_row':
            # process the ego state x, y
            return raw_input - raw_input[-1], raw_target - raw_input[-1]
        elif preprocess_choice == 'subtract_adjacent_row':
            # create [0, 0] as input and target in current frame
            zero_row = torch.zeros_like(raw_input[-1]).unsqueeze(0)
            return torch.cat((raw_input[:-1] - raw_input[1:], zero_row), dim=0), raw_target - raw_input[-1] # torch.cat((raw_target[1:] - raw_target[:-1], zero_row), dim=0)
        elif preprocess_choice == 'none':
            return raw_input, raw_target
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
        assert preprocess_choice in ['none', 'subtract_first_row', 'subtract_adjacent_row']
        if preprocess_choice == 'subtract_first_row':
            pred_global_way_points = prediction + input_tensor[-1]
        elif preprocess_choice == 'none':
            pred_global_way_points = prediction
        else:
            raise ValueError("The preprocess_choice must be 'none' or 'subtract_first_row'")
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
                multi_frame_images.append(self.transform(big_image))
        return torch.cat(multi_frame_images, dim=0)


# if __name__ == '__main__':
#     val_dataset = NuplanDataset('val')
#     print(val_dataset[500])


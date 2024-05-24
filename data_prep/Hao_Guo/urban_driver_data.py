import torch
from torch.utils.data import Dataset
import data_prep.Hao_Guo.config as config
import os
import pandas as pd
from PIL import Image
import numpy as np
from torchvision.transforms import transforms
from data_prep.Hao_Guo.utilities import quaternion_to_heading, convert_absolute_to_relative_se2_array


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

        assert split in ['train', 'val', 'test']
        if split == 'train':
            data_path = config.train_csv_folder
            image_root_path = config.images_mini_root_path
        elif split == 'val':
            data_path = config.val_csv_folder
            image_root_path = config.images_test_root_path
        elif split == 'test':
            data_path = config.test_csv_folder
            image_root_path = config.images_mini_root_path
        else:
            raise ValueError("The split must be 'train', 'val', 'test' ")

        print(f"The {split} set has {len(os.listdir(data_path))} different scenes")

        self.white_image_tensor, self.transform, self.rotate_degree_images, self.paste_locations = None, None, None, None
        self.initialize_image_parameters()

        # 0 - 7: cameras [CAM_F0, CAM_L0, CAM_R0, CAM_L1, CAM_R1, CAM_L2, CAM_R2, CAM_B0]
        # 8 - 10: ego2global_translation x, y, z
        # 11 - 14： ego2global ego2global_rotation q0 q1 q2 q3
        # 15 - 17： lidar2ego_translation x, y, z
        # 18 - 21: lidar2ego_rotation
        self.images_past = []
        self.coors_past = []
        self.coors_future = []

        csv_files = os.listdir(data_path)
        # each csv file is a different scene
        for csv_file in csv_files:
            csv_file_path = os.path.join(data_path, csv_file)
            # read csv file from the csv path
            df = pd.read_csv(csv_file_path, header=None)

            if len(df) < config.n_input_frames + config.n_output_frames:
                continue
            else:
                for start_index in range(len(df) - config.n_input_frames - config.n_output_frames + 1):
                    # Get the raw input and raw target (both including images, global coordinates information)
                    raw_input = df.iloc[start_index:start_index + config.n_input_frames]
                    raw_target = df.iloc[start_index + config.n_input_frames: start_index + config.n_input_frames + config.n_output_frames]

                    """For images"""
                    images_past_paths = raw_input.iloc[-config.n_input_image_frames:, 0:8 if config.use_8_camera else 1].values.tolist()
                    # Relative image path --> Absolute image path
                    for frame_index, frame_paths in enumerate(images_past_paths):
                        for i, path in enumerate(frame_paths):
                            images_past_paths[frame_index][i] = os.path.join(image_root_path, path)
                    self.images_past.append(images_past_paths)

                    """For ego status"""
                    current_ego_status, coors_past_processed = self.ego_status_feature_builder(raw_input, config.preprocess_choice)
                    coors_future_processed = self.trajectory_target_builder(current_ego_status, raw_target, config.preprocess_choice)

                    self.coors_past.append(coors_past_processed.to(torch.float32))
                    self.coors_future.append(coors_future_processed.to(torch.float32))

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
        # self.load_image_as_tensor(self.images_past[index])
        # This will accelerate training because we do not load images if we do not need them
        # TODO： Need to be improved
        if config.model_type != 'resnet':
            return {"images": 0, "ego_states": self.coors_past[index],
                    "future_waypoints": self.coors_future[index]}
        else:
            return {"images": self.load_image_as_tensor(self.images_past[index]), "ego_states": self.coors_past[index],
                    "future_waypoints": self.coors_future[index]}

    def __len__(self):
        """
        Get the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return self.n_samples

    def initialize_image_parameters(self):
        self.paste_locations = np.array([[1, 0], [0, 0], [2, 0], [0, 1], [2, 1], [0, 2], [1, 2], [2, 2]]) * config.image_size[0]
        self.rotate_degree_images = np.array([0, 0, 0, 90, -90, 180, 180, 180])
        self.transform = transforms.Compose([
            transforms.Resize(config.image_size),  # Image size
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
        # TODO: clear every time? check again
        self.white_image_tensor = self.transform(Image.new("RGB", config.image_size, (255, 255, 255)))

    def ego_status_feature_builder(self, _raw_input, preprocess_choice):
        coors_past_xy_raw = torch.tensor(_raw_input.iloc[:, 8:10].values, dtype=torch.float64)
        quaternion_past_raw = torch.tensor(_raw_input.iloc[:, 11:15].values, dtype=torch.float64)
        heading_past_raw = quaternion_to_heading(quaternion_past_raw)
        xy_yaw_past_raw = torch.cat((coors_past_xy_raw, heading_past_raw), dim=1)
        current_status = xy_yaw_past_raw[-1]
        return current_status, self.get_preprocessed_history_trajectory(current_status, xy_yaw_past_raw, preprocess_choice)

    def trajectory_target_builder(self, current_status, _raw_target, preprocess_choice):
        coors_future_xy_raw = torch.tensor(_raw_target.iloc[:, 8:10].values, dtype=torch.float64)
        quaternion_future_raw = torch.tensor(_raw_target.iloc[:, 11:15].values, dtype=torch.float32)
        heading_future_raw = quaternion_to_heading(quaternion_future_raw)
        xy_yaw_future_raw = torch.cat((coors_future_xy_raw, heading_future_raw), dim=1)
        return self.get_preprocessed_future_trajectory(current_status, xy_yaw_future_raw, preprocess_choice)

    @staticmethod
    def get_preprocessed_history_trajectory(current_status, raw_input, preprocess_choice):
        assert preprocess_choice in ['none', 'transform_to_local_coor']
        if preprocess_choice == 'transform_to_local_coor':
            # process the ego state x, y, theta
            return convert_absolute_to_relative_se2_array(current_status, raw_input)
        elif preprocess_choice == 'none':
            return raw_input
        else:
            raise ValueError("The preprocess_choice must be 'none' or 'transform_to_local_coor'")

    @staticmethod
    def get_preprocessed_future_trajectory(current_status, raw_target, preprocess_choice):
        assert preprocess_choice in ['none', 'transform_to_local_coor']
        if preprocess_choice == 'transform_to_local_coor':
            # process the ego state x, y, theta
            return convert_absolute_to_relative_se2_array(current_status, raw_target)
        elif preprocess_choice == 'none':
            return raw_target
        else:
            raise ValueError("The preprocess_choice must be 'none' or 'transform_to_local_coor'")

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
                else:
                    image = self.transform(Image.open(path))
                    multi_frame_images.append(image)
            if config.use_8_camera:
                multi_frame_images.append(self.transform(big_image))
        return torch.cat(multi_frame_images, dim=0)


# if __name__ == '__main__':
#     val_dataset = NuplanDataset('val')
#     print(val_dataset[0])

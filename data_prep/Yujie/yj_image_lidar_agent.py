import sys

from sklearn.datasets import fetch_openml

sys.path.append('.')

from enum import IntEnum
from typing import Any, List, Dict, Union, Tuple
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig
from navsim.common.enums import BoundingBoxIndex, LidarIndex
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)
from navsim.common.dataclasses import Scene, Trajectory, Annotations
import torch
from network.Yujie.yj_navsim_network import MyEgoStateModel, MyFusionModel
from network.Yujie.yj_navsim_fusion_model import FusionModel
from network.Yujie.yj_navsim_transfuser_model import TransfuserFusionModel
from network.Yujie.yj_transfuser_model import TransfuserModel
import train.Yujie.yj_navsim_train as config
import torch.nn as nn
from torchvision import transforms
import numpy as np
import numpy.typing as npt

class ImageLidarStatusFeatureBuilder(AbstractFeatureBuilder):
    def __init__(self):
        super().__init__()

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "ImageLidarStatusFeatureBuilder"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        features = {"camera_feature": self.get_camera_features(agent_input),
                    "status_feature": self.get_status_features(agent_input),
                    "lidar_feature": self.get_lidar_features(agent_input)}

        return features

    @staticmethod
    def get_camera_features(agent_input: AgentInput) -> torch.Tensor:
        """
        Extract camera information from AgentInput, raw image is: 1920 * 1080
        """
        transform = transforms.Compose([
            transforms.ToPILImage(),  # 将张量转换为 PIL 图像
            transforms.Resize((1080//3, 1080//3)),  # 缩小为原来的三分之一
            transforms.ToTensor()  # 将 PIL 图像转换为张量
        ])
        # 0 - 7: cameras [CAM_F0, CAM_L0, CAM_R0, CAM_L1, CAM_R1, CAM_L2, CAM_R2, CAM_B0]
        # The front camera sequence information
        image_multi_frames = []
        for cameras in agent_input.cameras:
            cam_f0 = transform(torch.tensor(cameras.cam_f0.image[:, :]).permute(2, 0, 1))  # 3 * 360 * 360
            cam_l0 = transform(torch.tensor(cameras.cam_l0.image[:, :]).permute(2, 0, 1))
            cam_ro = transform(torch.tensor(cameras.cam_r0.image[:, :]).permute(2, 0, 1))
            cam_l1 = transform(torch.tensor(cameras.cam_l1.image[:, :]).permute(2, 0, 1))
            cam_r1 = transform(torch.tensor(cameras.cam_r1.image[:, :]).permute(2, 0, 1))
            cam_l2 = transform(torch.tensor(cameras.cam_l2.image[:, :]).permute(2, 0, 1))
            cam_r2 = transform(torch.tensor(cameras.cam_r2.image[:, :]).permute(2, 0, 1))
            cam_b0 = transform(torch.tensor(cameras.cam_b0.image[:, :]).permute(2, 0, 1))

            stacked_images = torch.stack([cam_f0, cam_l0, cam_ro, cam_l1, cam_r1, cam_l2, cam_r2, cam_b0])  # 8 * 3 * 360 * 360
            image_multi_frames.append(stacked_images)
        camera_tensor = torch.stack(image_multi_frames)  # 4 * 8 * 3 * 360 * 360

        return camera_tensor

    @staticmethod
    def get_lidar_features(agent_input: AgentInput) -> torch.Tensor:
        """
        Compute LiDAR feature as 2D histogram, according to Transfuser
        :param agent_input: input dataclass
        :return: LiDAR histogram as torch tensors
        """
        # NOTE: Code from
        # https://github.com/autonomousvision/carla_garage/blob/main/team_code/data.py#L873
        def splat_points(point_cloud):
            # 256 x 256 grid
            xbins = np.linspace(
                config.lidar_min_x,
                config.lidar_max_x,
                (config.lidar_max_x - config.lidar_min_x)
                * int(config.pixels_per_meter)
                + 1, # type: ignore
            ) # type: ignore
            ybins = np.linspace(
                config.lidar_min_y,
                config.lidar_max_y,
                (config.lidar_max_y - config.lidar_min_y)
                * int(config.pixels_per_meter)
                + 1, # type: ignore
            ) # type: ignore
            hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
            hist[hist > config.hist_max_per_pixel] = config.hist_max_per_pixel
            overhead_splat = hist / config.hist_max_per_pixel
            return overhead_splat

        lidar_multi_frames = []
        for lidars in agent_input.lidars:
            # only consider (x,y,z) & swap axes for (N,3) numpy array
            lidar_pc = lidars.lidar_pc[LidarIndex.POSITION].T # type: ignore shape: (n, 3)
            
            # Remove points above the vehicle
            lidar_pc = lidar_pc[lidar_pc[..., 2] < config.max_height_lidar]
            below = lidar_pc[lidar_pc[..., 2] <= config.lidar_split_height]
            above = lidar_pc[lidar_pc[..., 2] > config.lidar_split_height]
            above_features = splat_points(above)
            if config.use_ground_plane:
                below_features = splat_points(below)
                features = np.stack([below_features, above_features], axis=-1)
            else:
                features = np.stack([above_features], axis=-1)
            features = np.transpose(features, (2, 0, 1)).astype(np.float32)  # 1(2)*256*256
            
            lidar_multi_frames.append(torch.tensor(features))
        features_multi_frames = torch.stack(lidar_multi_frames)
        return torch.tensor(features_multi_frames)

    @staticmethod
    def get_status_features(agent_input: AgentInput) -> torch.Tensor:
        # the velocity information
        ego_multi_frames = []
        for ego_status in agent_input.ego_statuses:
            velocity = torch.tensor(ego_status.ego_velocity)
            acceleration = torch.tensor(ego_status.ego_acceleration)
            driving_command = torch.tensor(ego_status.driving_command)
            ego_pose = torch.tensor(ego_status.ego_pose, dtype=torch.float)
            ego_status_feature = torch.cat([ego_pose, velocity, acceleration, driving_command], dim=-1)
            ego_multi_frames.append(ego_status_feature)
        ego_tensor = torch.stack(ego_multi_frames)
        return ego_tensor


class ImageLidarStatusTargetBuilder(AbstractTargetBuilder):
    def __init__(self, trajectory_sampling: TrajectorySampling):
        super().__init__()
        self._trajectory_sampling = trajectory_sampling

    def get_unique_name(self) -> str:
        return "ImageLidarStatusTargetBuilder"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        # TODO: Currently we only need trajectory as targets
        future_trajectory = scene.get_future_trajectory(
            num_trajectory_frames=self._trajectory_sampling.num_poses
        )
        #frame_idx = scene.scene_metadata.num_history_frames - 1
        #annotations = scene.frames[frame_idx].annotations

        #agent_states, agent_labels = self._compute_agent_targets(annotations)
        # return {"trajectory": torch.tensor(future_trajectory.poses),
        #         "agent_states": agent_states,
        #         "agent_labels": agent_labels,}
        return {"trajectory": torch.tensor(future_trajectory.poses)}
    
    # def _compute_agent_targets(self, annotations: Annotations) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Extracts 2D agent bounding boxes in ego coordinates
    #     :param annotations: annotation dataclass
    #     :return: tuple of bounding box values and labels (binary)
    #     """

    #     max_agents = config.num_bounding_boxes
    #     agent_states_list: List[npt.NDArray[np.float32]] = []

    #     def _xy_in_lidar(x: float, y: float) -> bool:
    #         return (config.lidar_min_x <= x <= config.lidar_max_x) and (
    #             config.lidar_min_y <= y <= config.lidar_max_y
    #         )

    #     for box, name in zip(annotations.boxes, annotations.names):
    #         box_x, box_y, box_heading, box_length, box_width = (
    #             box[BoundingBoxIndex.X],
    #             box[BoundingBoxIndex.Y],
    #             box[BoundingBoxIndex.HEADING],
    #             box[BoundingBoxIndex.LENGTH],
    #             box[BoundingBoxIndex.WIDTH],
    #         )

    #         if name == "vehicle" and _xy_in_lidar(box_x, box_y):
    #             agent_states_list.append(
    #                 np.array([box_x, box_y, box_heading, box_length, box_width], dtype=np.float32)
    #             )

    #     agents_states_arr = np.array(agent_states_list)

    #     # filter num_instances nearest
    #     agent_states = np.zeros((max_agents, BoundingBox2DIndex.size()), dtype=np.float32)
    #     agent_labels = np.zeros(max_agents, dtype=bool)

    #     if len(agents_states_arr) > 0:
    #         distances = np.linalg.norm(agents_states_arr[..., BoundingBox2DIndex.POINT], axis=-1)
    #         argsort = np.argsort(distances)[:max_agents]

    #         # filter detections
    #         agents_states_arr = agents_states_arr[argsort]
    #         agent_states[: len(agents_states_arr)] = agents_states_arr
    #         agent_labels[: len(agents_states_arr)] = True

    #     return torch.tensor(agent_states), torch.tensor(agent_labels)


class UrbanDriverAgentFull(AbstractAgent):
    def __init__(
            self,
            trajectory_sampling: TrajectorySampling,
            checkpoint_path: str = None, # type: ignore
    ):
        super().__init__()
        self._trajectory_sampling = trajectory_sampling
        self._checkpoint_path = checkpoint_path
        self._lr = config.lr
        self.my_model = TransfuserModel()

    def name(self) -> str:
        """Inherited, see superclass."""

        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if torch.cuda.is_available() and config.device == "cuda":
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)
        else:
            state_dict: Dict[str, Any] = torch.load(
                self._checkpoint_path, map_location=torch.device("cpu")
            )
        # print(state_dict)
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_all_sensors(include=True)  # TODO: Here only use the front camera [0]

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [ImageLidarStatusTargetBuilder(trajectory_sampling=self._trajectory_sampling)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [ImageLidarStatusFeatureBuilder()]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:  # TODO: add later
        poses: torch.Tensor = self.my_model(features["status_feature"].to(config.device), features["camera_feature"].to(config.device), features["lidar_feature"].to(config.device))
        return {"trajectory": poses.reshape(-1, self._trajectory_sampling.num_poses, 3)} # type: ignore

    def compute_loss(
            self,
            features: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return torch.nn.functional.mse_loss(predictions["trajectory"].float().to(config.device),
                                           targets["trajectory"].float().to(config.device))  # TODO: change later

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        return torch.optim.Adam(self.my_model.parameters(), lr=self._lr)

    def get_training_callbacks(self):  # TODO: probably add later
        pass

    # def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
    #     """
    #     Computes the ego vehicle trajectory.
    #     :param agent_input: Dataclass with agent inputs.
    #     :return: Trajectory representing the predicted ego's position in future
    #     """
    #     self.eval()
    #     features: Dict[str, torch.Tensor] = {}
    #     # build features
    #     for builder in self.get_feature_builders():
    #         features.update(builder.compute_features(agent_input))

    #     # add batch dimension
    #     features = {k: v.unsqueeze(0) for k, v in features.items()}

    #     if self.my_model.model_type == "Transformer" or "TransformerModified":
    #         print("my_model_type is:", self.my_model.model_type)
    #         ego_state = features["status_feature"]
    #         images = features["camera_feature"]
    #         lidar_bev = features["lidar_feature"] 
    #         batch_size = ego_state.size()[0]
    #         # forward pass
    #         with torch.no_grad():
    #             target_infer = torch.zeros(batch_size, 1, config.n_output_dim, dtype=torch.long, device=config.device)
    #             for _ in range(config.n_output_frames):
    #                 pred = self.my_model(ego_state, images, lidar_bev, target_infer)
    #                 # Concatenate previous input with predicted best word
    #                 target_infer = torch.cat((target_infer, pred[:, -1:, :]), dim=1)
                
    #             poses = target_infer[:, 1:, :].squeeze(0).numpy()
    #     else:
    #         with torch.no_grad():
    #             predictions = self.forward(features)
                
    #             poses = predictions["trajectory"].squeeze(0).numpy()

    #     # extract trajectory
    #     return Trajectory(poses)
    
# class BoundingBox2DIndex(IntEnum):

#     _X = 0
#     _Y = 1
#     _HEADING = 2
#     _LENGTH = 3
#     _WIDTH = 4

#     @classmethod
#     def size(cls):
#         valid_attributes = [
#             attribute
#             for attribute in dir(cls)
#             if attribute.startswith("_")
#             and not attribute.startswith("__")
#             and not callable(getattr(cls, attribute))
#         ]
#         return len(valid_attributes)

#     @classmethod
#     @property
#     def X(cls):
#         return cls._X

#     @classmethod
#     @property
#     def Y(cls):
#         return cls._Y

#     @classmethod
#     @property
#     def HEADING(cls):
#         return cls._HEADING

#     @classmethod
#     @property
#     def LENGTH(cls):
#         return cls._LENGTH

#     @classmethod
#     @property
#     def WIDTH(cls):
#         return cls._WIDTH

#     @classmethod
#     @property
#     def POINT(cls):
#         # assumes X, Y have subsequent indices
#         return slice(cls._X, cls._Y + 1)

#     @classmethod
#     @property
#     def STATE_SE2(cls):
#         # assumes X, Y, HEADING have subsequent indices
#         return slice(cls._X, cls._HEADING + 1)

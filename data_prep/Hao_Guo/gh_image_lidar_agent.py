import sys

sys.path.append('.')

from typing import Any, List, Dict, Union
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
from navsim.common.dataclasses import Scene, Trajectory
import torch
from network.Hao_Guo.gh_navsim_network import MyEgoStateModel, MyFusionModel
import train.Hao_Guo.gh_navsim_train as config
import torch.nn as nn
from torchvision import transforms
import numpy as np
import numpy.typing as npt
from network.Hao_Guo.gh_navsim_fusion_model import FusionModel


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

        # code source: https://github.com/autonomousvision/carla_garage/blob/main/team_code/data.py#L873
        def splat_points(point_cloud):
            # 256 x 256 grid
            xbins = np.linspace(
                config.lidar_min_x,
                config.lidar_max_x,
                (config.lidar_max_x - config.lidar_min_x)
                * int(config.pixels_per_meter)
                + 1,  # type: ignore
            )  # type: ignore
            ybins = np.linspace(
                config.lidar_min_y,
                config.lidar_max_y,
                (config.lidar_max_y - config.lidar_min_y)
                * int(config.pixels_per_meter)
                + 1,  # type: ignore
            )  # type: ignore
            # 将数据分成多个箱子，并统计每个箱子中的样本数量，生成一个多维的直方图
            hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
            hist[hist > config.hist_max_per_pixel] = config.hist_max_per_pixel
            # 将数据进行归一化，最大值为1, 最小值为0
            overhead_splat = hist / config.hist_max_per_pixel
            return overhead_splat

        # only consider (x,y,z) & swap axes for (N,3) numpy array
        lidar_features = []
        for lidar in agent_input.lidars:
            lidar_pc = lidar.lidar_pc[LidarIndex.POSITION].T  # type: ignore

            # Remove points above the vehicle， 只选取低于车辆高度的点
            lidar_pc = lidar_pc[lidar_pc[..., 2] < config.max_height_lidar]
            # 把点分成两部分，一部分高于0.2， 一部分低于0.2
            below = lidar_pc[lidar_pc[..., 2] <= config.lidar_split_height]
            above = lidar_pc[lidar_pc[..., 2] > config.lidar_split_height]
            above_features = splat_points(above)
            if config.use_ground_plane:
                below_features = splat_points(below)
                features = np.stack([below_features, above_features], axis=-1)
            else:
                features = np.stack([above_features], axis=-1)
            features = torch.tensor(np.transpose(features, (2, 0, 1)).astype(np.float32))
            lidar_features.append(features)
        return torch.stack(lidar_features)

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
        return {"trajectory": torch.tensor(future_trajectory.poses)}


class UrbanDriverAgentFull(AbstractAgent):
    def __init__(
            self,
            trajectory_sampling: TrajectorySampling,
            checkpoint_path: str = None,
    ):
        super().__init__()
        self._trajectory_sampling = trajectory_sampling
        self._checkpoint_path = checkpoint_path
        self._lr = config.lr
        # self.my_model = MyFusionModel.AllImagesEgoFusion().to(config.device)
        # self.my_model = MyFusionModel.ImageEgoFusionTfGreedy()
        self.my_model = FusionModel.AllImagesEgoFusion().to(config.device)

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
        self.load_state_dict({k.replace("module.agent.", ""): v for k, v in state_dict.items()})

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_all_sensors(include=True)

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [ImageLidarStatusTargetBuilder(trajectory_sampling=self._trajectory_sampling)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [ImageLidarStatusFeatureBuilder()]

    def forward(self, features: Dict[str, torch.Tensor], tgt=None) -> Dict[str, torch.Tensor]:  # TODO: add later
        if self.my_model.model_type == "Transformer":
            poses: torch.Tensor = self.my_model(features["status_feature"].to(config.device), features["camera_feature"].to(config.device), tgt["trajectory"].to(config.device))
        else:
            poses: torch.Tensor = self.my_model(features["status_feature"].to(config.device), features["camera_feature"].to(config.device), features["lidar_feature"].to(config.device))
        return {"trajectory": poses.reshape(-1, self._trajectory_sampling.num_poses, 3)}

    def compute_loss(
            self,
            features: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return torch.nn.functional.l1_loss(predictions["trajectory"].to(config.device),
                                           targets["trajectory"].to(config.device))  # TODO: change later

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
    #
    #     # add batch dimension
    #     features = {k: v.unsqueeze(0) for k, v in features.items()}
    #
    #     if self.my_model.model_type == "Transformer" or "TransformerModified":
    #         ego_state = features["status_feature"]
    #         images = features["camera_feature"]
    #         batch_size = ego_state.size()[0]
    #         # forward pass
    #         with torch.no_grad():
    #             target_infer = torch.zeros(batch_size, 1, config.n_output_dim, dtype=torch.long, device=config.device)
    #             for _ in range(config.n_output_frames):
    #                 pred = self.my_model(ego_state, images, target_infer)
    #                 # Concatenate previous input with predicted best word
    #                 target_infer = torch.cat((target_infer, pred[:, -1:, :]), dim=1)
    #
    #             poses = target_infer[:, 1:, :].squeeze(0).numpy()
    #     else:
    #         with torch.no_grad():
    #             predictions = self.forward(features)
    #             poses = predictions["trajectory"].squeeze(0).numpy()
    #
    #     # extract trajectory
    #     return Trajectory(poses)


# if __name__ == '__main__':
#     point_cloud = np.array([
#         [1.0, 2.0, 3.0],
#         [2.5, 3.5, 4.0],
#         [3.0, 4.0, 2.0],
#         [1.5, 2.5, 1.0],
#         [2.0, 3.0, 1.0]
#     ])
#     xbins = np.linspace(1.0, 3.0, 3 + 1)
#     ybins = np.linspace(2.0, 4.0, 3 + 1)
#     hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
#     print("finished")


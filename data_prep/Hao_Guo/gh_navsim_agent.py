import sys

sys.path.append('.')

from typing import Any, List, Dict, Union
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)
from navsim.common.dataclasses import Scene, Trajectory
import torch
from network.Hao_Guo.gh_navsim_network import MyEgoStateModel
import train.Hao_Guo.gh_navsim_train as config
import torch.nn as nn


class EgoStatusFeatureBuilder(AbstractFeatureBuilder):
    def __init__(self):
        super().__init__()

    def get_unique_name(self) -> str:
        return "ego_status_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        ego_multi_frames = []
        for ego_status in agent_input.ego_statuses:
            velocity = torch.tensor(ego_status.ego_velocity)
            acceleration = torch.tensor(ego_status.ego_acceleration)
            driving_command = torch.tensor(ego_status.driving_command)
            ego_pose = torch.tensor(ego_status.ego_pose, dtype=torch.float)
            ego_status_feature = torch.cat([ego_pose, velocity, acceleration, driving_command], dim=-1)
            ego_multi_frames.append(ego_status_feature)
        ego_tensor = torch.stack(ego_multi_frames)
        return {"ego_status": ego_tensor}


class TrajectoryTargetBuilder(AbstractTargetBuilder):
    def __init__(self, trajectory_sampling: TrajectorySampling):
        super().__init__()
        self._trajectory_sampling = trajectory_sampling

    def get_unique_name(self) -> str:
        return "trajectory_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        future_trajectory = scene.get_future_trajectory(
            num_trajectory_frames=self._trajectory_sampling.num_poses
        )
        return {"trajectory": torch.tensor(future_trajectory.poses)}


class UrbanDriverAgent(AbstractAgent):
    def __init__(
            self,
            trajectory_sampling: TrajectorySampling,
            checkpoint_path: str = "",
    ):
        super().__init__()
        self._trajectory_sampling = trajectory_sampling
        self._checkpoint_path = checkpoint_path
        self._lr = config.lr
        #self.my_model = MyEgoStateModel.DriveUsingCommand().to(device=config.device)
        # self.my_model = MyEgoStateModel.MLP().to(config.device)
        # self.my_model = MyEgoStateModel.TransformerModel.TransformerModifiedEgoStateModel(config.n_feature_dim).to(device=config.device)
        self.my_model = MyEgoStateModel.TransformerModel.TransformerEgoStateModel(config.n_feature_dim).to(config.device)
        #self.my_model = MyEgoStateModel.GRU().to(config.device)
        #self.my_model = MyEgoStateModel.MLP().to(device=config.device)

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
        return SensorConfig.build_no_sensors()

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [
            TrajectoryTargetBuilder(trajectory_sampling=self._trajectory_sampling),
        ]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [EgoStatusFeatureBuilder()]

    # def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    #     poses: torch.Tensor = self.my_model(features["ego_status"].to(config.device))
    #     return {"trajectory": poses.reshape(-1, self._trajectory_sampling.num_poses, 3)}

    def forward(self, features: Dict[str, torch.Tensor], tgt=None) -> Dict[str, torch.Tensor]:
        if self.my_model.model_type == "Transformer":
            poses: torch.Tensor = self.my_model(features["ego_status"].to(config.device), tgt["trajectory"].to(config.device))
            poses: torch.Tensor = self.my_model(features["ego_status"].to(config.device),
                                                tgt["trajectory"].to(config.device))
        else:
            poses: torch.Tensor = self.my_model(features["ego_status"].to(config.device))
        return {"trajectory": poses.reshape(-1, self._trajectory_sampling.num_poses, 3)}

    def compute_loss(
            self,
            features: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return torch.nn.functional.l1_loss(predictions["trajectory"].to(config.device),
                                           targets["trajectory"].to(config.device))
        loss_func = torch.nn.MSELoss()
        return loss_func(predictions["trajectory"].to(config.device),
                         targets["trajectory"].float().to(config.device))

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        return torch.optim.Adam(self.my_model.parameters(), lr=self._lr)

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        """
        Computes the ego vehicle trajectory.
        :param agent_input: Dataclass with agent inputs.
        :return: Trajectory representing the predicted ego's position in future
        """
        self.eval()
        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        if self.my_model.model_type == "Transformer":
            src = features["ego_status"]
            batch_size = src.size()[0]
            # forward pass
            with torch.no_grad():
                target_infer = torch.zeros(batch_size, 1, config.n_output_dim, dtype=torch.long, device=config.device)
                for _ in range(config.n_output_frames):
                    pred = self.my_model(src, target_infer)
                    # Concatenate previous input with predicted best word
                    target_infer = torch.cat((target_infer, pred[:, -1:, :]), dim=1)

                poses = target_infer[:, 1:, :].squeeze(0).numpy()
        else:
            with torch.no_grad():
                predictions = self.forward(features)
                poses = predictions["trajectory"].squeeze(0).numpy()

        # extract trajectory
        return Trajectory(poses)

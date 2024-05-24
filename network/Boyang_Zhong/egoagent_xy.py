from typing import Any, List, Dict, Union
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.abstract_agent import Trajectory
from navsim.common.dataclasses import AgentInput, SensorConfig
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)
from navsim.common.dataclasses import Scene
import navsim.agents.MyAgents.xy_train as config
import torch
from navsim.agents.MyAgents.my_models import MyModel

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
            num_trajectory_frames = self._trajectory_sampling.num_poses
        )
        return {"trajectory": torch.tensor(future_trajectory.poses)}


class EgoAgentXY(AbstractAgent):
    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        lr: float,
        checkpoint_path: str = None,
    ):
        super().__init__()
        self._trajectory_sampling = trajectory_sampling
        # self._trajectory_sampling.num_poses = 9
        # self._trajectory_sampling.interval_length = 0.5
        # self._trajectory_sampling.time_horizon = 4.5
        self._checkpoint_path = checkpoint_path
        self._lr = config.lr
        self.my_model = MyModel.LSTM().to(device=config.device)

    def name(self) -> str:
        """Inherited, see superclass."""

        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if torch.cuda.is_available():
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

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        poses: torch.Tensor = self.my_model(features["ego_status"].to(config.device))
        # batch_size = config.batch_size
        # output_frame = config.n_output_frames
        # output_dim = config.n_output_dim
        # poses = poses.view(list(poses.size())[0], output_frame, output_dim)
 
        # # Extract coordinates
        # coordinates = poses[:, :, :2]  # Shape: (batch_size, output_frame, 2)

        # # Compute vectors and angles
        # vectors = coordinates[:, 2:, :] - coordinates[:, :-2, :]  # Compute vectors connecting the first and third points, etc.
        # angles = torch.atan2(vectors[:, :, 1], vectors[:, :, 0])  # Compute angles from the vectors
        # angles = torch.cat([torch.zeros(list(poses.size())[0], 1, device=config.device), angles], dim=1)  # Add a zero angle for the first point

        # # Concatenate angles with original coordinates
        # angles = angles.unsqueeze(-1)  # Change shape to (batch_size, output_frame, 1)
        # poses_with_angles = torch.cat([coordinates[:,0:-1,:], angles], dim=-1) 
        
        return {"trajectory": poses.reshape(-1, self._trajectory_sampling.num_poses, 2)}

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        #print(f"targets_shape:{targets['trajectory'].size()}")
        return torch.nn.functional.mse_loss(predictions["trajectory"].to(config.device, dtype=float), targets["trajectory"][:,:,:2].to(config.device, dtype=float))

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
    
        # if self.my_model.model_type == "Transformer" or "TransformerModified":
        #     ego_state = features["status_feature"]
        #     images = features["camera_feature"]
        #     batch_size = ego_state.size()[0]
        #     # forward pass
        #     with torch.no_grad():
        #         target_infer = torch.zeros(batch_size, 1, config.n_output_dim, dtype=torch.long, device=config.device)
        #         for _ in range(config.n_output_frames):
        #             pred = self.my_model(ego_state, images, target_infer)
        #             # Concatenate previous input with predicted best word
        #             target_infer = torch.cat((target_infer, pred[:, -1:, :]), dim=1)
    
        #         poses = target_infer[:, 1:, :].squeeze(0).numpy()
        # else:
        ego_state = features["ego_status"]
        batch_size = ego_state.size()[0]
        
        output_frames = config.n_output_frames
        output_dim = config.n_output_dim
        with torch.no_grad():
            # #poses = poses.view(list(poses.size())[0], output_frame, output_dim)
            # print(f"poses_shape:{poses.size()}")
            # # Extract coordinates
            # coordinates = poses[:, :, :2]  # Shape: (batch_size, output_frame, 2)

            #     # Compute vectors and angles
            # vectors = coordinates[:, 2:, :] - coordinates[:, :-2, :]  # Compute vectors connecting the first and third points, etc.
            # angles = torch.atan2(vectors[:, :, 1], vectors[:, :, 0])  # Compute angles from the vectors
            # angles = torch.cat([torch.zeros(list(poses.size())[0], 1, device=config.device), angles], dim=1)  # Add a zero angle for the first point

            #     # Concatenate angles with original coordinates
            # angles = angles.unsqueeze(-1)  # Change shape to (batch_size, output_frame, 1)
            # poses_with_angles = torch.cat([coordinates[:,0:-1,:], angles], dim=-1) 
            # shape = poses_with_angles.reshape(-1, self._trajectory_sampling.num_poses, 3)
            # #poses = predictions["trajectory"].squeeze(0).numpy()
            
            
            # Extract coordinates
            poses = self.my_model(ego_state)
            poses = poses.view(list(poses.size())[0], output_frames, output_dim)
            #print(f"poses_shape:{poses.size()}")
            angles = torch.zeros(batch_size, output_frames, 1, device = config.device)
            # Compute vectors and angles
            for i in range(1, output_frames):
                
                d_x = poses[:, i, 0] - poses[:, i-1, 0]
                d_y = poses[:, i, 1] - poses[:, i-1, 1]
                angles[:, i, 0] = torch.atan2(d_y, d_x)
                #print(f"angles_shape:{angles.size()}")
            #angles = torch.cat([torch.zeros(list(poses.size())[0], 1, device=config.device), angles], dim=1)  # Add a zero angle for the first point

            poses_with_angles = torch.cat((poses, angles), dim = -1)
            shape = poses_with_angles.reshape(-1, self._trajectory_sampling.num_poses, 3)
            poses_with_angles = poses_with_angles.squeeze(0).numpy()
        # extract trajectory
        return Trajectory(poses_with_angles)

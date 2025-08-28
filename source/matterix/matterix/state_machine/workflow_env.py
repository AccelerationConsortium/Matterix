from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence, Tuple, Optional
import torch
# --- basic types ---


@dataclass()
class Pose:
    """
    Pose expressed in the *local frame* of the environment/entity this proxy defines.
    position: meters, orientation: quaternion (x, y, z, w).
    """
    position: torch.Tensor # (N, 3)
    orientation: torch.Tensor # (N, 4)
    
@dataclass()
class ObjectState:
    """State for a passive object: only a pose in the local frame."""
    pose: Pose
    frames: Mapping[str, Sequence] = field(default_factory=dict)

@dataclass()
class RobotState:
    """
    State for a robot:
    - base_pose: robot base/link pose in the local frame
    - joint_positions: ordered joint values (e.g., radians/meters)
    """
    pose: Pose
    joint_positions: torch.Tensor # (N, joint_num)
    ee_position: Optional[torch.Tensor] # (N, 3)

class WorkflowEnv(ABC):
    """
    Proxy interface for exposing positions of things in the scene.
    Implementations **must** return data in the **local frame**.

    Subclass this and implement the abstract properties/methods.
    """
    def __init__(self):
        self.num_envs = 1
    @property
    @abstractmethod
    def objects(self) -> Mapping[str, ObjectState]:
        """
        Mapping: object name -> ObjectState (pose in local frame).
        Must be cheap to read (cache or compute-once-per-step as needed).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def robots(self) -> Mapping[str, RobotState]:
        """
        Mapping: robot name -> RobotState (base pose + joint positions in local frame).
        """
        raise NotImplementedError
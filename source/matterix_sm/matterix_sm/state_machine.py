# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import time
import torch
from typing import TYPE_CHECKING, Any, Optional

from .action_base import ActionBase, ActionBaseCfg
from .compositional_actions import CompositionalAction
from .primitive_actions import PrimitiveAction
from .scene_data import ArticulationData, Pose, RigidObjectData, SceneData

if TYPE_CHECKING:
    from .robot_action_spaces import ActionSpaceInfo


class StateMachine:
    """Simple sequential state machine orchestrating a list of actions over parallel environments.

    The StateMachine automatically injects execution parameters (num_envs, device)
    into primitive action configs, so you don't need to specify them for every action.
    Timeout uses the global TIMEOUT_DEFAULT constant defined in action_constants.py.

    Example:
        >>> from matterix_sm import StateMachine, MoveToFrameCfg, CloseGripperCfg
        >>> from matterix_sm.robot_action_spaces import FRANKA_IK_ACTION_SPACE
        >>>
        >>> # Create state machine with required parameters from environment
        >>> sm = StateMachine(num_envs=env.num_envs, dt=env.step_dt, device="cuda")
        >>>
        >>> # Define actions WITHOUT specifying num_envs/device
        >>> # (timeout defaults to TIMEOUT_DEFAULT from action_constants.py)
        >>> actions = [
        ...     MoveToFrameCfg(
        ...         agent_assets="robot",
        ...         object="beaker",
        ...         frame="grasp",
        ...         action_space_info=FRANKA_IK_ACTION_SPACE,
        ...     ),
        ...     CloseGripperCfg(
        ...         agent_assets="robot",
        ...         duration=10,
        ...         action_space_info=FRANKA_IK_ACTION_SPACE,
        ...     ),
        ... ]
        >>>
        >>> # StateMachine automatically fills in num_envs/device
        >>> sm.set_action_sequence(actions)
        >>>
        >>> # Run state machine
        >>> while not (sm.action_sequence_success | sm.action_sequence_failure).all():
        ...     action = sm.step(obs)
        ...     obs, reward, done, info = env.step(action)
    """

    def __init__(
        self,
        num_envs: int,
        dt: float,
        device: torch.device | str | None = None,
        enable_timing: bool = False,
    ):
        """
        Args:
            num_envs: Number of parallel environments.
            dt: Simulation timestep in seconds.
            device: Torch device used for internal tensors and action execution.
                   If None, auto-detects CUDA availability.
            enable_timing: Enable performance timing reports (useful for development/debugging).
        """
        self.enable_timing = enable_timing
        self.num_envs = num_envs
        self.dt = dt

        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.current_action_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.actions: list[PrimitiveAction] = []
        self.action_hierarchy_paths = []
        self.all_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.obs_history = None
        self.frame_history = []

        # SceneData - auto-created on first step
        self.scene_data: SceneData | None = None

        # Action dictionary (per-agent actions)
        self.action_dict: dict[str, torch.Tensor] = {}

        # Timing tracking for performance monitoring (only if enabled)
        self.step_times: list[float] = []
        self.step_time_threshold = 2.0  # Flag steps taking more than 2x mean time

    def reset(self) -> None:
        """Reset per-episode bookkeeping and all registered actions.

        Call this after env.reset() but before the step loop. The first action
        will be computed on the first call to step(obs).
        """
        self.current_action_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.action_sequence_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.action_sequence_failure = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Clear action dict so first action gets fresh robot state from observations
        self.action_dict.clear()
        for action in self.actions:
            action.reset()

    def _initialize_action_dict_for_agent(
        self,
        agent_name: str,
        action_dim: int,
        action_space_info: Optional["ActionSpaceInfo"],
    ) -> torch.Tensor:
        """Initialize action dictionary for an agent with current robot state from scene_data.

        This ensures that partial actions (e.g., gripper-only) don't send zero position/orientation
        or zero joint commands when they run as the first action.

        Handles both joint-space and task-space (EE pose) action spaces based on action_space_info.

        Args:
            agent_name: Name of the agent/robot to initialize.
            action_dim: Dimension of the action space.
            action_space_info: Optional action space metadata defining which indices control which DOFs.

        Returns:
            Initialized action tensor with shape (num_envs, action_dim), populated with current
            robot state if available in scene_data.
        """
        init_tensor = torch.zeros((self.num_envs, action_dim), device=self.device)

        # Try to populate with current robot state from scene_data
        if (
            self.scene_data is not None
            and agent_name in self.scene_data.articulations
            and action_space_info is not None
        ):
            robot_data = self.scene_data.articulations[agent_name]

            # Option 1: Joint space control (direct joint positions)
            if action_space_info.joint_indices is not None and robot_data.joint_pos is not None:
                # Use current joint positions for joint-space actions
                for i, idx in enumerate(action_space_info.joint_indices):
                    # Check bounds to avoid index errors
                    if i < robot_data.joint_pos.shape[1]:
                        init_tensor[:, idx] = robot_data.joint_pos[:, i]

            # Option 2: Task space control (EE pose: position + orientation)
            else:
                # Transform EE pose from world frame to robot base frame
                # Observations are in world frame, but IK actions expect base frame
                if (
                    robot_data.ee_pos_w is not None
                    and robot_data.ee_quat_w is not None
                    and action_space_info.position_indices is not None
                    and action_space_info.orientation_indices is not None
                ):
                    # Import frame transformation utility
                    try:
                        from isaaclab.utils.math import subtract_frame_transforms
                    except ImportError:
                        # Fallback: use world frame directly (will cause issues in multi-env)
                        # This import error should only happen during edge deployment
                        subtract_frame_transforms = None

                    if subtract_frame_transforms is not None:
                        # Transform from world frame to base frame
                        ee_pos_b, ee_quat_b = subtract_frame_transforms(
                            robot_data.root_pos_w,  # Base position in world
                            robot_data.root_quat_w,  # Base orientation in world
                            robot_data.ee_pos_w,  # EE position in world
                            robot_data.ee_quat_w,  # EE orientation in world
                        )

                        # Set position from current EE position (in base frame)
                        for i, idx in enumerate(action_space_info.position_indices):
                            init_tensor[:, idx] = ee_pos_b[:, i]

                        # Set orientation from current EE orientation (in base frame)
                        for i, idx in enumerate(action_space_info.orientation_indices):
                            init_tensor[:, idx] = ee_quat_b[:, i]
                    else:
                        # Fallback: use world frame (will cause multi-env issues)
                        for i, idx in enumerate(action_space_info.position_indices):
                            init_tensor[:, idx] = robot_data.ee_pos_w[:, i]
                        for i, idx in enumerate(action_space_info.orientation_indices):
                            init_tensor[:, idx] = robot_data.ee_quat_w[:, i]
                elif robot_data.ee_pos_w is not None and action_space_info.position_indices is not None:
                    # Only position available - use world frame
                    for i, idx in enumerate(action_space_info.position_indices):
                        init_tensor[:, idx] = robot_data.ee_pos_w[:, i]
                elif robot_data.ee_quat_w is not None and action_space_info.orientation_indices is not None:
                    # Only orientation available - use world frame
                    for i, idx in enumerate(action_space_info.orientation_indices):
                        init_tensor[:, idx] = robot_data.ee_quat_w[:, i]

            # Gripper is common to both joint and task space
            if robot_data.gripper_pos is not None and action_space_info.gripper_indices is not None:
                # Gripper pos is typically (num_envs, 2) for two fingers
                # Average to get single value per environment
                gripper_avg = robot_data.gripper_pos.mean(dim=-1)
                for idx in action_space_info.gripper_indices:
                    init_tensor[:, idx] = gripper_avg

        return init_tensor

    def _flatten_actions_recursive(
        self, actions: list[ActionBase], hierarchy_path: list[str] | None = None
    ) -> list[tuple[PrimitiveAction, list[str]]]:
        """Recursively flatten nested CompositionalActions into a flat list.

        This enables unlimited hierarchy depth. For example:
        - PickAndPlace (CompositionalAction)
          └─ PickObject (CompositionalAction)
              └─ MoveToFrame, CloseGripper, MoveRelative (PrimitiveActions)
          └─ PlaceObject (CompositionalAction)
              └─ MoveToFrame, OpenGripper, MoveRelative (PrimitiveActions)

        All nested CompositionalActions are recursively expanded into
        a single flat list of primitive PrimitiveActions.

        Args:
            actions: Iterable of Action or CompositionalAction instances.
            hierarchy_path: Current path in the hierarchy (for tracking).

        Returns:
            List of tuples: (Action, hierarchy_path) where hierarchy_path tracks
            the compositional action ancestry.
        """
        if hierarchy_path is None:
            hierarchy_path = []

        flattened = []
        for action in actions:
            if isinstance(action, CompositionalAction):
                # Build hierarchy path: parent > child
                action_name = f"{type(action).__name__}"
                new_path = hierarchy_path + [action_name]
                # Recursively flatten nested compositional actions
                flattened.extend(self._flatten_actions_recursive(action.actions_list, new_path))
            else:
                # Leaf action: store with its hierarchy path
                action_name = f"{type(action).__name__}"
                full_path = hierarchy_path + [action_name]
                flattened.append((action, full_path))
        return flattened

    def set_action_sequence(
        self,
        actions: ActionBase | ActionBaseCfg | list[ActionBase | ActionBaseCfg],
    ) -> None:
        """Register a new action sequence.

        Compositional actions are recursively expanded into their constituent actions.
        All actions are reset. Supports unlimited nesting depth.

        Args:
            actions: Single action/config OR iterable of Action instances/ActionBaseCfg configs.
                     Configs are automatically converted to actions using the factory pattern.

        Example:
            >>> # Single action
            >>> sm.set_action_sequence(PickObject(object="beaker", ...))

            >>> # List of actions (traditional)
            >>> sm.set_action_sequence([
            ...     PickObject(object="beaker", ...),
            ...     MoveToFrame(...)
            ... ])

            >>> # Single config
            >>> sm.set_action_sequence(PickObjectCfg(object="beaker", ...))

            >>> # List of configs (new - more flexible!)
            >>> sm.set_action_sequence([
            ...     PickObjectCfg(object="beaker", ...),
            ...     MoveToFrameCfg(...)
            ... ])
        """
        from .action_base import ActionBase, ActionBaseCfg

        # Ensure actions is iterable (convert single action to list)
        if not isinstance(actions, (list, tuple)):
            actions = [actions]

        # Convert configs to actions
        action_instances = []
        for item in actions:
            if isinstance(item, ActionBaseCfg):
                # Convert to action using factory
                action_instances.append(ActionBase.from_cfg(item))
            else:
                # It's already an action instance
                action_instances.append(item)

        flattened_with_hierarchy = self._flatten_actions_recursive(action_instances)

        # Separate actions and hierarchy paths
        self.actions = [action for action, _ in flattened_with_hierarchy]
        self.action_hierarchy_paths = [path for _, path in flattened_with_hierarchy]

        # Set execution parameters on all primitive actions, then reset them
        for action in self.actions:
            action.set_execution_params(self.num_envs, self.device, self.dt)
            action.reset()  # Reset after execution params are set

        self.obs_history = {}
        self.frame_history = []

        # Print action sequence info
        print("\n" + "=" * 80)
        print("STATE MACHINE ACTION SEQUENCE INITIALIZED")
        print("=" * 80)
        print(f"Total actions: {len(self.actions)}")
        print(f"Number of environments: {self.num_envs}")
        print("\nAction hierarchy:")
        for i, (action, path) in enumerate(zip(self.actions, self.action_hierarchy_paths)):
            hierarchy_str = " > ".join(path)
            print(f"  [{i}] {action.__class__.__name__:20s} :: {hierarchy_str}")
        print("=" * 80 + "\n")

    def execute_action_sequence(
        self,
        actions: ActionBase | ActionBaseCfg | list[ActionBase | ActionBaseCfg],
    ) -> tuple[torch.Tensor, list[Any]]:
        """Run a full sequence until success or failure for all environments.

        Args:
            actions: Single action/config OR iterable of Action instances/ActionBaseCfg configs.
                     Configs are automatically converted to actions using the factory pattern.

        Returns:
            Tuple:
                - success mask (num_envs,) bool tensor
                - frame history (opaque log/list as produced during stepping)
        """
        self.set_action_sequence(actions)
        while not (self.action_sequence_success | self.action_sequence_failure).all():
            self.step()
        return self.action_sequence_success, self.frame_history

    def step(
        self, obs: dict | None = None, reward=None, terminated=None, truncated=None
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """Advance exactly one step for all environments.

        Executes the current action for each active environment, updates per-env
        success/failure masks, and advances to the next action when complete.

        Args:
            obs: Observation dictionary from environment. If None, uses existing scene_data.
            reward: Reward values (not used currently, reserved for future).
            terminated: Termination flags (not used currently, reserved for future).
            truncated: Truncation flags (not used currently, reserved for future).

        Returns:
            dict[str, torch.Tensor] or torch.Tensor: Action dictionary mapping agent names
            to action tensors. If only one agent, returns single tensor for backward compatibility.
        """
        step_start = time.perf_counter() if self.enable_timing else None
        timings = {} if self.enable_timing else None

        # Update scene data from observations
        if self.enable_timing:
            obs_update_start = time.perf_counter()
        if obs is not None:
            self.update_scene_data_from_obs(obs)
        if self.enable_timing:
            timings["obs_update"] = time.perf_counter() - obs_update_start

        # Note: We do NOT clear action_dict here. It's preserved across steps so that
        # partial actions (like gripper) can maintain position/orientation from previous actions.
        # New actions will overwrite the dimensions they control via mask-based accumulation.

        if self.enable_timing:
            mask_init_start = time.perf_counter()
        active_mask = ~(self.action_sequence_success | self.action_sequence_failure)
        unique_action_indices = torch.unique(self.current_action_idx[active_mask])

        combined_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.enable_timing:
            timings["mask_init"] = time.perf_counter() - mask_init_start
            action_compute_times = []
            action_dict_update_times = []

        for action_idx in unique_action_indices:
            # env_running_mask: Shape (num_envs,) - which environments are running this action
            env_running_mask = (self.current_action_idx == action_idx) & active_mask
            env_ids = torch.nonzero(env_running_mask).squeeze(-1)
            action = self.actions[action_idx]

            # Actions now return tensor + dimension mask for partial updates
            if self.enable_timing:
                compute_start = time.perf_counter()
            action_tensor, action_dim_mask, env_success_mask, env_timeout_mask = action.compute_action(
                self.scene_data, env_ids
            )
            if self.enable_timing:
                action_compute_times.append(time.perf_counter() - compute_start)

            if self.enable_timing:
                device_transfer_start = time.perf_counter()
            env_success_mask = env_success_mask.to(self.device)
            env_timeout_mask = env_timeout_mask.to(self.device)
            action_tensor = action_tensor.to(self.device)
            action_dim_mask = action_dim_mask.to(self.device)
            if self.enable_timing:
                timings[f"device_transfer_action_{action_idx}"] = time.perf_counter() - device_transfer_start

            # Map action tensor to controlled asset(s) using mask-based accumulation
            if self.enable_timing:
                dict_update_start = time.perf_counter()
            for agent_name in action.agent_assets:
                if agent_name not in self.action_dict:
                    # Initialize with current robot state from scene_data (already populated from obs)
                    # This prevents gripper-only actions from sending zero position/orientation
                    self.action_dict[agent_name] = self._initialize_action_dict_for_agent(
                        agent_name,
                        action_tensor.shape[-1],
                        action.action_space_info,
                    )
                # Apply partial update using mask (last-write-wins strategy)
                # action_dim_mask (action_dim,): which dimensions to update
                # env_ids (num_active_envs,): which environments to update
                # action_tensor is already (num_envs, action_dim), so we index it for active envs
                # Use torch.where for efficient masked assignment
                self.action_dict[agent_name][env_ids, :] = torch.where(
                    action_dim_mask[None, :],  # Broadcast mask to (1, action_dim) -> (num_active_envs, action_dim)
                    action_tensor[env_ids, :],  # Use action values from active envs (index into full tensor)
                    self.action_dict[agent_name][env_ids, :],  # Keep existing values where mask is False
                )
            if self.enable_timing:
                action_dict_update_times.append(time.perf_counter() - dict_update_start)

            # Track success/failure per environment
            if self.enable_timing:
                mask_update_start = time.perf_counter()
            combined_success[env_running_mask] = env_success_mask[env_running_mask]
            self.action_sequence_failure = self.action_sequence_failure | env_timeout_mask
            if self.enable_timing:
                timings[f"mask_update_action_{action_idx}"] = time.perf_counter() - mask_update_start

        if self.enable_timing:
            timings["action_compute_total"] = sum(action_compute_times)
            timings["action_dict_update_total"] = sum(action_dict_update_times)

        # Advance environments that completed current action
        if self.enable_timing:
            advance_start = time.perf_counter()
        self.current_action_idx[combined_success] += 1
        self.action_sequence_success = self.action_sequence_success | (self.current_action_idx >= len(self.actions))
        if self.enable_timing:
            timings["advance_envs"] = time.perf_counter() - advance_start

        # Note: We do NOT clear action_dict here. The last action's output is preserved
        # so that subsequent actions (like gripper actions) can maintain the current pose
        # by using the mask-based accumulation to preserve position/orientation dimensions.

        # Return dict or single tensor for backward compatibility
        if self.enable_timing:
            return_prep_start = time.perf_counter()
        if len(self.action_dict) == 1:
            result = list(self.action_dict.values())[0]
        else:
            result = self.action_dict

        self.print_status()
        if self.enable_timing:
            timings["return_prep"] = time.perf_counter() - return_prep_start

        # Calculate total step time and print timing report (only if enabled)
        if self.enable_timing:
            step_time = time.perf_counter() - step_start
            self.step_times.append(step_time)
            self._print_timing_report(step_time, timings)

        return result

    def _print_timing_report(self, step_time: float, timings: dict[str, float]) -> None:
        """Print color-coded timing report for current step.

        Args:
            step_time: Total time for this step in seconds
            timings: Dictionary of intermediate timing breakdowns
        """
        # Calculate mean time (only if we have multiple samples)
        if len(self.step_times) > 1:
            mean_time = sum(self.step_times) / len(self.step_times)
            is_slow = step_time > mean_time * self.step_time_threshold
        else:
            mean_time = step_time
            is_slow = False

        # ANSI color codes
        RED_BOLD = "\033[91m\033[1m"
        RESET = "\033[0m"
        YELLOW = "\033[93m"
        GREEN_BOLD = "\033[92m\033[1m"

        # Print header with color coding and visual separator
        step_ms = step_time * 1000
        mean_ms = mean_time * 1000

        if is_slow:
            ratio = step_time / mean_time
            print(f"\n{RED_BOLD}{'=' * 80}")
            print(f"⚠️  SLOW STEP DETECTED: {step_ms:.2f}ms ({ratio:.1f}x mean: {mean_ms:.2f}ms)")
            print(f"{'=' * 80}{RESET}")

            # Print breakdown of slow steps
            if timings:
                print(f"{YELLOW}  Breakdown (operations >5% of total time):{RESET}")
                # Sort by time descending to show slowest operations first
                sorted_timings = sorted(timings.items(), key=lambda x: x[1], reverse=True)
                for name, timing_val in sorted_timings:
                    timing_ms = timing_val * 1000
                    percent = (timing_val / step_time) * 100 if step_time > 0 else 0
                    if percent > 5:  # Only show operations taking >5% of total time
                        print(f"{YELLOW}    ▸ {name}: {timing_ms:.2f}ms ({percent:.1f}%){RESET}")
            print(f"{RED_BOLD}{'=' * 80}{RESET}\n")
        else:
            # Compact single-line output for normal steps
            print(f"{GREEN_BOLD}[TIMING]{RESET} {step_ms:.2f}ms (mean: {mean_ms:.2f}ms)")

    def get_status(self) -> list[dict]:
        """Get current status grouped by unique action states.

        Returns:
            List of dicts, each containing:
                - action_idx: Current action index in the sequence
                - action_name: Name of the action class
                - hierarchy_path: Full hierarchy path (e.g., "PickObject > MoveToFrame")
                - status: "running", "failed", or "success"
                - env_ids_preview: First 3 environment IDs (list of ints)
                - total_envs: Total number of environments in this state
                - action_output_sample: Sample 8-dim action vector from first env
        """
        # Group environments by their current action index (reuse logic from step())
        unique_indices = torch.unique(self.current_action_idx)

        status_list = []
        for action_idx in unique_indices:
            action_idx_int = action_idx.item()
            env_mask = self.current_action_idx == action_idx
            env_ids = torch.nonzero(env_mask, as_tuple=False).squeeze(-1)

            # Convert to Python list for preview (only first 3 to avoid overhead)
            total_envs = env_ids.shape[0]
            env_ids_preview = env_ids[:3].cpu().tolist()
            if not isinstance(env_ids_preview, list):
                env_ids_preview = [env_ids_preview]

            # Determine status using existing state variables
            if self.action_sequence_success[env_ids[0]]:
                status = "success"
            elif self.action_sequence_failure[env_ids[0]]:
                status = "failed"
            else:
                status = "running"

            # Get action metadata (already precomputed during set_action_sequence)
            if action_idx_int < len(self.actions):
                action = self.actions[action_idx_int]
                action_name = action.__class__.__name__
                hierarchy_path = " > ".join(self.action_hierarchy_paths[action_idx_int])
                # Sample action output from action_dict
                # Note: If print_status is called right after an env advanced to next action,
                # but before step() is called again, action_dict will be empty (cleared at step start)
                if self.action_dict:
                    # Get first agent's action (for multi-agent, could show all)
                    first_agent_action = list(self.action_dict.values())[0]
                    action_output_sample = first_agent_action[env_ids[0]].cpu().tolist()
                else:
                    action_output_sample = None
            else:
                # Beyond last action = completed
                action_name = None
                hierarchy_path = None
                action_output_sample = None

            status_list.append({
                "action_idx": action_idx_int,
                "action_name": action_name,
                "hierarchy_path": hierarchy_path,
                "status": status,
                "env_ids_preview": env_ids_preview,
                "total_envs": total_envs,
                "action_output_sample": action_output_sample,
            })

        # Sort by number of environments (descending) for better readability
        status_list.sort(key=lambda x: x["total_envs"], reverse=True)
        return status_list

    def print_status(self, step: int | None = None, episode: int | None = None) -> None:
        """Print formatted status showing current state grouped by action.

        Args:
            step: Optional step number to display
            episode: Optional episode number to display
        """
        status_list = self.get_status()

        print("\n" + "=" * 80)
        # Build header with optional step/episode info
        header_parts = []
        if episode is not None:
            header_parts.append(f"Episode {episode}")
        if step is not None:
            header_parts.append(f"Step {step}")
        header_parts.append(f"Actions: {len(self.actions)}")
        header_parts.append(f"Envs: {self.num_envs}")

        print("STATE MACHINE STATUS (" + ", ".join(header_parts) + ")")
        print("-" * 80)

        for item in status_list:
            action_idx = item["action_idx"]
            total_envs = item["total_envs"]
            percentage = (total_envs / self.num_envs) * 100

            # Format action index display
            if action_idx >= len(self.actions):
                idx_str = "✓"
            elif item["status"] == "failed":
                idx_str = "✗"
            else:
                idx_str = f"{action_idx}/{len(self.actions)}"

            # Environment IDs preview
            env_preview = item["env_ids_preview"]
            if total_envs > 3:
                env_str = f"{env_preview} ... (+{total_envs - 3} more)"
            else:
                env_str = str(env_preview)

            # Display action name and hierarchy (handle None for completed states)
            action_name_str = item["action_name"] if item["action_name"] is not None else "Completed"
            hierarchy_str = item["hierarchy_path"] if item["hierarchy_path"] is not None else "All actions done"

            print(f"\n[{idx_str:>10s}] {action_name_str:20s} [{item['status']:12s}]")
            print(f"             {hierarchy_str}")
            print(f"             Envs ({total_envs:3d}, {percentage:5.1f}%): {env_str}")

            # Show sample action output
            if item["action_output_sample"] is not None:
                output_str = "[" + ", ".join(f"{x:6.3f}" for x in item["action_output_sample"]) + "]"
                print(f"             Sample action: {output_str}")

        print("\n" + "=" * 80 + "\n")

    # ==================== SceneData Management ====================

    def update_scene_data_from_obs(self, obs: dict) -> None:
        """Update or create scene_data from observation dictionary.

        On first call, auto-creates SceneData structure based on observation keys.
        On subsequent calls, updates existing structure with new values.

        Args:
            obs: Observation dictionary from environment. Expected structure:
                 {
                     "articulations": dict[str, dict|tensor],  # Robot, articulated objects
                     "rigid_objects": dict[str, dict|tensor],   # Beakers, flasks, etc.
                     # ... other groups as needed
                 }
        """
        if self.scene_data is None:
            # First call: create structure
            self.scene_data = self._create_scene_data_from_obs(obs)
        else:
            # Subsequent calls: update existing structure
            self._update_scene_data_from_obs(obs)

    def _create_scene_data_from_obs(self, obs: dict) -> SceneData:
        """Auto-create SceneData structure from first observation.

        Args:
            obs: Observation dictionary from environment.

        Returns:
            Initialized SceneData with empty or populated fields based on obs keys.
        """
        scene_data = SceneData(
            articulations={},
            rigid_objects={},
        )

        # Parse articulations group if exists
        if "articulations" in obs:
            artic_obs = obs["articulations"]
            if isinstance(artic_obs, dict):
                # Group observations by entity name (split on __)
                grouped_obs = self._group_observations_by_entity(artic_obs)
                for name, data in grouped_obs.items():
                    scene_data.articulations[name] = self._parse_articulation_data(data)

        # Parse rigid_objects group if exists
        if "rigid_objects" in obs:
            obj_obs = obs["rigid_objects"]
            if isinstance(obj_obs, dict):
                # Group observations by entity name (split on __)
                grouped_obs = self._group_observations_by_entity(obj_obs)
                for name, data in grouped_obs.items():
                    scene_data.rigid_objects[name] = self._parse_rigid_object_data(data, name)

        return scene_data

    def _group_observations_by_entity(self, obs_group: dict) -> dict[str, dict]:
        """Group observations by entity name using double-underscore delimiter.

        Isaac Lab observation keys use format: "entity__observation_name"
        This converts {"robot__pos": tensor, "robot__vel": tensor}
        into {"robot": {"pos": tensor, "vel": tensor}}

        Args:
            obs_group: Dictionary with keys like "entity__obs_name"

        Returns:
            Nested dictionary grouped by entity name
        """
        grouped = {}
        for key, value in obs_group.items():
            if "__" in key:
                # Split on double underscore
                entity_name, obs_name = key.split("__", 1)
                if entity_name not in grouped:
                    grouped[entity_name] = {}
                grouped[entity_name][obs_name] = value
            else:
                # Legacy format: treat key as entity name with single observation dict
                if key not in grouped:
                    grouped[key] = value if isinstance(value, dict) else {key: value}

        return grouped

    def _update_scene_data_from_obs(self, obs: dict) -> None:
        """Update existing scene_data with new observation values.

        Args:
            obs: Observation dictionary from environment.
        """
        # Update articulations
        if "articulations" in obs and isinstance(obs["articulations"], dict):
            grouped_obs = self._group_observations_by_entity(obs["articulations"])
            for name, data in grouped_obs.items():
                self.scene_data.articulations[name] = self._parse_articulation_data(data)

        # Update rigid_objects
        if "rigid_objects" in obs and isinstance(obs["rigid_objects"], dict):
            grouped_obs = self._group_observations_by_entity(obs["rigid_objects"])
            for name, data in grouped_obs.items():
                self.scene_data.rigid_objects[name] = self._parse_rigid_object_data(data, name)

    def _parse_articulation_data(self, data: dict | torch.Tensor) -> ArticulationData:
        """Parse observation data into ArticulationData.

        Args:
            data: Either dict with named fields or flat tensor (structure TBD).

        Returns:
            ArticulationData with parsed fields.
        """
        if isinstance(data, dict):
            # Dict format - extract fields (support both old _w and new _world_ naming)
            # Use if-else chain instead of 'or' to avoid tensor boolean ambiguity
            return ArticulationData(
                root_pos_w=(data.get("root_world_pos") if "root_world_pos" in data else data.get("root_pos_w")),
                root_quat_w=(data.get("root_world_quat") if "root_world_quat" in data else data.get("root_quat_w")),
                joint_pos=data.get("joint_pos"),
                joint_vel=data.get("joint_vel"),
                ee_pos_w=(data.get("ee_world_pos") if "ee_world_pos" in data else data.get("ee_pos_w")),
                ee_quat_w=(data.get("ee_world_quat") if "ee_world_quat" in data else data.get("ee_quat_w")),
                gripper_pos=data.get("gripper_pos"),
            )
        else:
            # Flat tensor format - TODO: define standard layout
            # For now, raise error until we define the convention
            raise NotImplementedError(
                "Flat tensor parsing not yet implemented. ObservationManager should return dict with named fields."
            )

    def _parse_rigid_object_data(self, data: dict | torch.Tensor, asset_name: str = None) -> RigidObjectData:
        """Parse observation data into RigidObjectData.

        Args:
            data: Either dict with named fields or flat tensor (structure TBD).
            asset_name: Name of the asset (currently unused, reserved for future use).

        Returns:
            RigidObjectData with parsed fields.
        """
        if isinstance(data, dict):
            # Dict format - extract fields
            frames_dict = None

            # Parse frames from observation data
            if "frames" in data and isinstance(data["frames"], dict):
                # Parse frames into Pose objects
                frames_dict = {}
                for frame_name, frame_data in data["frames"].items():
                    if isinstance(frame_data, dict):
                        frames_dict[frame_name] = Pose(
                            position=frame_data.get("position"),
                            orientation=frame_data.get("orientation"),
                        )
                    elif isinstance(frame_data, torch.Tensor) and frame_data.shape[-1] == 7:
                        # Assume [pos(3), quat(4)]
                        frames_dict[frame_name] = Pose(
                            position=frame_data[..., :3],
                            orientation=frame_data[..., 3:7],
                        )

            # Also check for individual frame terms (e.g., "pre_grasp_frame", "grasp_frame")
            if frames_dict is None:
                frame_keys = [k for k in data.keys() if k.endswith("_frame")]
                if frame_keys:
                    frames_dict = {}
                    for key in frame_keys:
                        # Extract frame name (e.g., "pre_grasp_frame" -> "pre_grasp")
                        frame_name = key.replace("_frame", "")
                        frame_data = data[key]
                        if isinstance(frame_data, torch.Tensor) and frame_data.shape[-1] == 7:
                            # Parse 7D tensor: [x, y, z, qw, qx, qy, qz]
                            frames_dict[frame_name] = Pose(
                                position=frame_data[..., :3],
                                orientation=frame_data[..., 3:7],
                            )

            # Priority order for position: object_world_pos > pos_w > position
            pos_w = data.get("object_world_pos", data.get("pos_w", data.get("position")))

            # Priority order for orientation: object_world_quat > quat_w > orientation
            quat_w = data.get("object_world_quat", data.get("quat_w", data.get("orientation")))

            lin_vel_w = data.get("object_lin_vel") if "object_lin_vel" in data else data.get("lin_vel_w")
            ang_vel_w = data.get("object_ang_vel") if "object_ang_vel" in data else data.get("ang_vel_w")

            return RigidObjectData(
                pos_w=pos_w,
                quat_w=quat_w,
                lin_vel_w=lin_vel_w,
                ang_vel_w=ang_vel_w,
                frames=frames_dict,
            )
        else:
            # Flat tensor format - TODO: define standard layout
            raise NotImplementedError(
                "Flat tensor parsing not yet implemented. ObservationManager should return dict with named fields."
            )

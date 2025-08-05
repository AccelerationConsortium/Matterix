# Copyright (c) 2022-2025, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass


@configclass
class ActionManagerCfg:
    """Configuration of the action manager for the MDP.

    The terms of this manager will be populated with the actions added to the environment and the assets in the scene.
    """


@configclass
class EventManagerCfg:
    """Configuration of the event manager for the MDP.

    The terms of this manager will be populated with the events added to the environment and the assets in the scene.
    """


@configclass
class DefaultEventManagerCfg(EventManagerCfg):
    """Configuration of the default event manager for the MDP.

    This manager is used to reset the scene to a default state. The default state is specified
    by the scene configuration.
    """

    reset_scene_to_default = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class ObservationManagerCfg:
    """Configuration of the observattion manager for the MDP.

    The terms of this manager will be populated with the events added to the environment and the assets in the scene.
    """


class InitialStateRecorder(RecorderTerm):
    """Recorder term that records the initial state of the environment after reset."""

    def record_post_reset(self, env_ids: Sequence[int] | None):
        def extract_env_ids_values(value):
            nonlocal env_ids
            if isinstance(value, dict):
                return {k: extract_env_ids_values(v) for k, v in value.items()}
            return value[env_ids]

        return "initial_state", extract_env_ids_values(self._env.scene.get_state(is_relative=True))


class PostStepStatesRecorder(RecorderTerm):
    """Recorder term that records the state of the environment at the end of each step."""

    def record_post_step(self):
        return "states", self._env.scene.get_state(is_relative=True)


class PreStepActionsRecorder(RecorderTerm):
    """Recorder term that records the actions in the beginning of each step."""

    def record_pre_step(self):
        return "actions", self._env.action_manager.action


class PreStepObservationsRecorder(RecorderTerm):
    """Recorder term that records the observations in each step."""

    def record_pre_step(self):
        return "obs", self._env.obs_buf


@configclass
class InitialStateRecorderCfg(RecorderTermCfg):
    """Configuration for the initial state recorder term."""

    class_type: type[RecorderTerm] = InitialStateRecorder


@configclass
class PostStepStatesRecorderCfg(RecorderTermCfg):
    """Configuration for the step state recorder term."""

    class_type: type[RecorderTerm] = PostStepStatesRecorder


@configclass
class PreStepActionsRecorderCfg(RecorderTermCfg):
    """Configuration for the step action recorder term."""

    class_type: type[RecorderTerm] = PreStepActionsRecorder


@configclass
class PreStepObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the step policy observation recorder term."""

    class_type: type[RecorderTerm] = PreStepObservationsRecorder


##
# Recorder manager configurations.
##


@configclass
class MatterixBaseRecorderCfg(RecorderManagerBaseCfg):
    """Recorder configurations for recording actions and states."""

    record_initial_state = InitialStateRecorderCfg()
    record_post_step_states = PostStepStatesRecorderCfg()
    record_pre_step_actions = PreStepActionsRecorderCfg()
    record_pre_step_observations = PreStepObservationsRecorderCfg()

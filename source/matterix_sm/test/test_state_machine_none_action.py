# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for StateMachine.step() returning action=None.

matterix_sm has no isaacsim/omni imports, so these run as plain pytest - no
Isaac Sim launch required.

Covers the two StateMachine-side scenarios from the action=None bug:
  1. A sequence made up entirely of SemanticActionCfg-derived steps (no
     agent_assets anywhere) - StateMachine.step() must return action=None,
     since there is no agent to hold a pose for.
  2. A sequence that starts with semantic-only steps but has a later step
     with agent_assets set - the upfront fallback scan should already see
     that agent and return a valid (non-None) hold tensor even during the
     initial semantic-only step.
"""

import torch

from matterix_sm import StateMachine, TurnOnHeaterCfg, WaitCfg
from matterix_sm.scene_data import SceneData


def test_pure_semantic_sequence_returns_none_action():
    """A workflow of only TurnOnHeaterCfg steps: step() returns action=None,
    exactly one IsHeaterOn semantic action is emitted, and the sequence succeeds."""
    sm = StateMachine(num_envs=2, dt=1 / 60, device="cpu")
    sm.set_action_sequence(
        [
            TurnOnHeaterCfg(asset_name="ika_plate", value=True, target_temperature=373.15),
        ]
    )
    sm.reset()

    action, semantic_actions = sm.step(obs=None)

    assert action is None, "action must be None when no action in the sequence has agent_assets"

    assert semantic_actions is not None
    assert len(semantic_actions) == 1
    info = semantic_actions[0]
    assert info.type == "IsHeaterOn"
    assert info.asset_name == "ika_plate"
    assert info.value is True

    assert sm.action_sequence_success.all()
    assert not sm.action_sequence_failure.any()


def test_pure_semantic_sequence_multi_step_stays_none():
    """Two chained semantic-only steps: action stays None across both steps."""
    sm = StateMachine(num_envs=1, dt=1 / 60, device="cpu")
    sm.set_action_sequence(
        [
            TurnOnHeaterCfg(asset_name="ika_plate", value=True),
            TurnOnHeaterCfg(asset_name="ika_plate", value=False),
        ]
    )
    sm.reset()

    action1, semantics1 = sm.step(obs=None)
    assert action1 is None
    assert semantics1[0].value is True
    assert not sm.action_sequence_success.all()

    action2, semantics2 = sm.step(obs=None)
    assert action2 is None
    assert semantics2[0].value is False
    assert sm.action_sequence_success.all()


def test_mixed_sequence_returns_hold_tensor_during_initial_semantic_step():
    """A sequence that starts with a semantic-only step but has a later robot
    step: the upfront fallback scan (over the *entire* action list) already
    knows about "robot", so even the first (semantic) step must return a
    valid non-None hold tensor rather than None."""
    sm = StateMachine(num_envs=2, dt=1 / 60, device="cpu")
    sm.set_action_sequence(
        [
            TurnOnHeaterCfg(asset_name="ika_plate", value=True),
            # Not a realistic workflow ordering, but validates the fallback scan sees
            # "robot" from anywhere in the full action list, not just the current step.
            WaitCfg(duration=0.01, agent_assets="robot"),
        ]
    )
    sm.reset()
    # Fallback initialization only runs once scene_data is available.
    sm.scene_data = SceneData(articulations={}, rigid_objects={})

    action, semantic_actions = sm.step(obs=None)

    assert action is not None, "a later robot action in the sequence must give a hold tensor, not None"
    assert isinstance(action, torch.Tensor)
    assert action.shape[0] == 2
    assert semantic_actions is not None
    assert semantic_actions[0].value is True

# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import torch

from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass

from ..semantics import SemanticPredicate
from ..semantics_cfg import SemanticPredicateCfg, semantic_info

CONTACT_FORCE_THRESHOLD = 0.5  # N
CONTACT_SENSOR_KEY = "contact_sensor"  # scene key convention: "contact_sensor_{asset_name}"


# ---------------------------------------------------------------------------
# Base class (shared interface for downstream semantics)
# ---------------------------------------------------------------------------


class IsInContact(SemanticPredicate):
    """Base class for contact predicates.

    Downstream semantics (HeatConductivity, HeatConvection) use
    ``find_corresponding_semantics(IsInContact)`` to find whichever
    subclass is attached to the asset.
    """

    def __init__(self, env, cfg=None, parent_asset=None, name: str = None):
        super().__init__(env, cfg, parent_asset, name)
        self.is_in_contact: torch.Tensor = torch.full(
            (self.env.num_envs,), self.cfg.default_value, dtype=torch.bool, device=self.env.device
        )
        # Asset names currently in contact — consumed by HeatConductivity, HeatConvection etc.
        self.contacting_assets: set[str] = set()


# ---------------------------------------------------------------------------
# Manual / legacy mode
# ---------------------------------------------------------------------------


@configclass
class IsInContactCfg(SemanticPredicateCfg):
    """Contact predicate driven by workflow semantic actions (manual/legacy mode).

    Contact state is set explicitly via ``set_value()`` calls from the workflow
    state machine (``SemanticActionCfg``). No physics sensor is used.

    Use this when you do not need physics-derived contact detection or when
    the contact logic is determined by the workflow itself.

    Example::

        SemanticActionCfg(
            semantics=[
                SemanticInfo(type="IsInContact", asset_name="beaker",
                             value=True,
                             additional_info={"contacting_asset": "robot"}),
            ]
        )
    """

    type = "IsInContact"
    default_value: bool = False


class IsInContactManual(IsInContact):
    """Contact predicate that is set manually from workflow semantic actions."""

    def __init__(self, env, cfg: IsInContactCfg = None, parent_asset=None, name: str = None):
        super().__init__(env, cfg, parent_asset, name)

    def set_value(self, semantic_info: semantic_info):
        with contextlib.suppress(KeyError):
            self.contacting_assets.add(semantic_info.additional_info["contacting_asset"])
        with contextlib.suppress(KeyError):
            self.contacting_assets.discard(semantic_info.additional_info["remove_contacting_asset"])
        self.is_in_contact[:] = bool(self.contacting_assets)
        return True

    def step(self):
        self.print_status(f"is_in_contact={self.is_in_contact.tolist()}, contacting_assets={self.contacting_assets}")

    def reset(self, env_ids):
        self.is_in_contact[env_ids] = self.cfg.default_value
        if env_ids is ... or len(env_ids) == self.env.num_envs:
            self.contacting_assets.clear()


# ---------------------------------------------------------------------------
# Physics mode
# ---------------------------------------------------------------------------


@configclass
class IsInContactPhysicsCfg(SemanticPredicateCfg):
    """Contact predicate derived from physics simulation via a ContactSensor.

    Uses the same ``filter_prim_paths_expr`` naming as Isaac Lab's ``ContactSensorCfg``.
    Entries are **relative** prim paths — ``setup_scene()`` automatically prepends
    ``{ENV_REGEX_NS}/Articulations_`` or ``{ENV_REGEX_NS}/RigidObjects_`` based on
    which asset dict each name belongs to.

    The asset name for ``contacting_assets`` is derived from the part before the first
    ``/`` in each filter path entry (e.g. ``"robot/panda_leftfinger"`` → ``"robot"``,
    ``"ika_plate"`` → ``"ika_plate"``).

    ``setup_scene()`` resolves and registers the ``ContactSensorCfg`` automatically —
    no manual sensor declaration is needed in the task cfg.

    Example::

        "beaker": BEAKER_500ML_INST_CFG(
            pos=(...),
            semantics=[
                IsInContactPhysicsCfg(
                    filter_prim_paths_expr=[
                        "robot/panda_leftfinger",
                        "robot/panda_rightfinger",
                        "ika_plate",
                    ]
                ),
                HeatTransferCfg(temp=298.15, C=7920.0, A=0.1, K=200),
            ],
        )

    ``setup_scene()`` resolves the above to::

        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Articulations_robot/panda_leftfinger",
            "{ENV_REGEX_NS}/Articulations_robot/panda_rightfinger",
            "{ENV_REGEX_NS}/RigidObjects_ika_plate",
        ]
    """

    type = "IsInContactPhysics"
    default_value: bool = False
    filter_prim_paths_expr: list[str] = []


class IsInContactPhysics(IsInContact):
    """Contact predicate that reads a ContactSensor from the scene each step."""

    def __init__(self, env, cfg: IsInContactPhysicsCfg = None, parent_asset=None, name: str = None):
        super().__init__(env, cfg, parent_asset, name)
        self._contact_sensor: ContactSensor = None

    def init(self):
        asset_name = self.name.split("/")[0]
        sensor_key = f"{CONTACT_SENSOR_KEY}_{asset_name}"
        if sensor_key not in self.env.scene.keys():
            raise KeyError(
                f"[IsInContactPhysics] Sensor '{sensor_key}' not found in scene for '{self.name}'. "
                "Ensure activate_contact_sensors=True on the asset cfg."
            )
        self._contact_sensor = self.env.scene[sensor_key]
        self.print_status(f"sensor '{sensor_key}' found.")

    def step(self):
        # force_matrix_w: (num_envs, num_bodies, num_filter_prims, 3)
        force_matrix = self._contact_sensor.data.force_matrix_w

        if force_matrix is not None:
            # (num_envs, num_filter_prims): max force norm across bodies
            force_norms = force_matrix.norm(dim=-1).amax(dim=1)
            per_filter = force_norms > CONTACT_FORCE_THRESHOLD  # (num_envs, num_filter_prims)

            self.is_in_contact = per_filter.any(dim=-1)  # (num_envs,)

            # Asset name = part before the first "/" in each relative filter path.
            self.contacting_assets.clear()
            for idx, filter_path in enumerate(self.cfg.filter_prim_paths_expr):
                asset_name = filter_path.split("/")[0]
                if per_filter[:, idx].any():
                    self.contacting_assets.add(asset_name)

            self.print_status(
                f"is_in_contact={self.is_in_contact.tolist()}, "
                f"contacting_assets={self.contacting_assets}, "
                f"forces per filter={force_norms.amax(dim=0).tolist()}"
            )
        else:
            # Multi-body sensor: force_matrix_w is None, fall back to net force
            net_norms = self._contact_sensor.data.net_forces_w.norm(dim=-1).amax(dim=1)
            self.is_in_contact = net_norms > CONTACT_FORCE_THRESHOLD
            self.print_status(f"net force fallback: {net_norms.tolist()}")

    def reset(self, env_ids):
        self.is_in_contact[env_ids] = self.cfg.default_value
        # contacting_assets rebuilt from sensor each step — nothing to reset

# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Legacy Orbit hierarchy discovery and child addressing on Isaac Lab 3.0.

Nested assets are recursively decomposed into separately spawned referenced
leaf USDs. Isaac Lab 3.0's ``RigidObjectCollection`` is only the batched runtime
container for those independent leaves; it is not a materialization fallback.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Iterable, Sequence

try:
    from isaaclab.utils import configclass
except ImportError:
    # Matterix supports importing configuration modules without Isaac Lab.
    # Runtime materialization still requires the real Isaac Lab types.
    configclass = dataclasses.dataclass

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
LEGACY_LEAF_MATERIALIZATION = "legacy_leaf_assets"


class RigidObjectCollectionError(ValueError):
    """A nested USD hierarchy cannot be represented safely or unambiguously."""


class LegacyCollectionMaterializationUnsupported(RigidObjectCollectionError):
    """The hierarchy cannot be decomposed into the legacy leaf-asset model."""


@dataclasses.dataclass(frozen=True)
class CollectionRigidBodyDescriptor:
    """One addressable rigid-body descendant in source-USD local coordinates."""

    identifier: str
    relative_prim_path: str
    local_position: tuple[float, float, float]
    local_orientation: tuple[float, float, float, float]
    source_usd_path: str | None = None
    local_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self):
        if not _IDENTIFIER_PATTERN.fullmatch(self.identifier):
            raise RigidObjectCollectionError(
                f"Invalid child identifier {self.identifier!r}; expected {_IDENTIFIER_PATTERN.pattern}"
            )
        if not self.relative_prim_path or self.relative_prim_path.startswith("/"):
            raise RigidObjectCollectionError("Child relative_prim_path must be '.' or a relative USD prim path")
        if self.relative_prim_path != "." and any(
            part in {"", ".", ".."} for part in self.relative_prim_path.split("/")
        ):
            raise RigidObjectCollectionError(f"Invalid relative USD prim path {self.relative_prim_path!r}")
        values = (*self.local_position, *self.local_orientation, *self.local_scale)
        if not all(math.isfinite(value) for value in values):
            raise RigidObjectCollectionError(f"Child {self.identifier!r} pose must be finite")
        if not all(value > 0.0 for value in self.local_scale):
            raise RigidObjectCollectionError(f"Child {self.identifier!r} scale values must be greater than zero")
        norm = math.sqrt(sum(value * value for value in self.local_orientation))
        if not math.isclose(norm, 1.0, abs_tol=1.0e-5):
            raise RigidObjectCollectionError(f"Child {self.identifier!r} orientation must be normalized")


@dataclasses.dataclass(frozen=True)
class RigidObjectCollectionManifest:
    """Stable identifier/path table shared by config generation and runtime lookup."""

    usd_path: str
    default_prim_path: str
    children: tuple[CollectionRigidBodyDescriptor, ...]

    @property
    def child_ids(self) -> tuple[str, ...]:
        """Return stable children in deterministic collection order."""
        return tuple(child.identifier for child in self.children)

    def child(self, identifier: str) -> CollectionRigidBodyDescriptor:
        """Resolve one stable identifier or fail with the available identifiers."""
        for child in self.children:
            if child.identifier == identifier:
                return child
        raise RigidObjectCollectionError(
            f"Unknown collection child {identifier!r}; available children: {list(self.child_ids)}"
        )

    def object_id(self, identifier: str) -> int:
        """Resolve one stable identifier to its collection object index."""
        child = self.child(identifier)
        return self.children.index(child)

    def identifier_for_path(self, relative_prim_path: str) -> str:
        """Resolve a source-relative prim path to a stable identifier."""
        for child in self.children:
            if child.relative_prim_path == relative_prim_path:
                return child.identifier
        raise RigidObjectCollectionError(f"No collection child at relative path {relative_prim_path!r}")

    def as_mapping(self) -> dict[str, object]:
        """Return a JSON-compatible manifest for audit artifacts."""
        return {
            "usd_path": self.usd_path,
            "default_prim_path": self.default_prim_path,
            "children": [dataclasses.asdict(child) for child in self.children],
        }


@configclass
class MatterixRigidObjectCollectionCfg:
    """One source USD whose rigid descendants become one addressable collection."""

    usd_path: str = dataclasses.MISSING
    prim_path: str = "{ENV_REGEX_NS}/RigidObjects"
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    include_prim_path_regex: str | None = None
    exclude_prim_path_regex: str | None = None
    identifier_attribute: str | None = "matterix:identifier"
    collision_group: int = 0
    debug_vis: bool = False
    activate_contact_sensors: bool = False
    semantic_tags: list[tuple[str, str]] = dataclasses.field(default_factory=list)
    semantics: list[object] = dataclasses.field(default_factory=list)
    sensors: dict[str, object] = dataclasses.field(default_factory=dict)
    event_terms: dict[str, object] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if not self.usd_path.strip():
            raise RigidObjectCollectionError("usd_path must not be empty")
        if not self.prim_path.strip():
            raise RigidObjectCollectionError("prim_path must not be empty")
        if self.collision_group not in (0, -1):
            raise RigidObjectCollectionError("collision_group must be 0 or -1")
        if not all(math.isfinite(value) and value > 0.0 for value in self.scale):
            raise RigidObjectCollectionError("scale values must be finite and greater than zero")
        if not all(math.isfinite(value) for value in (*self.pos, *self.rot)):
            raise RigidObjectCollectionError("root pose values must be finite")
        norm = math.sqrt(sum(value * value for value in self.rot))
        if not math.isclose(norm, 1.0, abs_tol=1.0e-5):
            raise RigidObjectCollectionError("root orientation must be normalized")
        for pattern in (self.include_prim_path_regex, self.exclude_prim_path_regex):
            if pattern is not None:
                try:
                    re.compile(pattern)
                except re.error as error:
                    raise RigidObjectCollectionError(f"Invalid prim-path regex {pattern!r}: {error}") from error


RigidObjectCollectionCfg = MatterixRigidObjectCollectionCfg


def stable_child_identifier(relative_prim_path: str) -> str:
    """Derive a stable collection key from a source-relative USD prim path."""
    raw = "root" if relative_prim_path == "." else "__".join(relative_prim_path.split("/"))
    identifier = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_.-")
    if not identifier:
        raise RigidObjectCollectionError(f"Could not derive an identifier from {relative_prim_path!r}")
    if not identifier[0].isalpha():
        identifier = f"child_{identifier}"
    return identifier


def build_collection_manifest(
    usd_path: str,
    default_prim_path: str,
    children: Iterable[CollectionRigidBodyDescriptor],
) -> RigidObjectCollectionManifest:
    """Sort and validate discovered leaves for deterministic indexing."""
    ordered = tuple(sorted(children, key=lambda child: (child.relative_prim_path, child.identifier)))
    if not ordered:
        raise RigidObjectCollectionError(f"No enabled RigidBodyAPI descendants found in {usd_path!r}")
    identifiers = [child.identifier for child in ordered]
    if len(identifiers) != len(set(identifiers)):
        duplicates = sorted({name for name in identifiers if identifiers.count(name) > 1})
        raise RigidObjectCollectionError(f"Duplicate collection child identifiers: {duplicates}")
    paths = [child.relative_prim_path for child in ordered]
    if len(paths) != len(set(paths)):
        raise RigidObjectCollectionError("The same rigid-body prim path was discovered more than once")

    return RigidObjectCollectionManifest(
        usd_path=usd_path,
        default_prim_path=default_prim_path,
        children=ordered,
    )


def discover_collection_rigid_bodies(
    cfg: MatterixRigidObjectCollectionCfg,
) -> RigidObjectCollectionManifest:
    """Discover separately spawnable leaves using the legacy Orbit semantics."""
    return discover_legacy_leaf_rigid_bodies(cfg)


def _discover_composed_rigid_bodies(
    cfg: MatterixRigidObjectCollectionCfg,
) -> RigidObjectCollectionManifest:
    """Discover composed rigid bodies, including bodies in instance proxies."""
    try:
        from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError as error:
        raise RigidObjectCollectionError(
            "USD hierarchy discovery requires pxr from the Isaac Sim/Isaac Lab runtime"
        ) from error

    stage = Usd.Stage.Open(cfg.usd_path)
    if stage is None:
        raise RigidObjectCollectionError(f"Could not open nested USD asset {cfg.usd_path!r}")
    root = stage.GetDefaultPrim()
    if not root or not root.IsValid():
        raise RigidObjectCollectionError(f"Nested USD asset {cfg.usd_path!r} has no valid default prim")

    root_path = root.GetPath().pathString.rstrip("/")
    include = re.compile(cfg.include_prim_path_regex) if cfg.include_prim_path_regex else None
    exclude = re.compile(cfg.exclude_prim_path_regex) if cfg.exclude_prim_path_regex else None
    xforms = UsdGeom.XformCache()
    root_world_inverse = xforms.GetLocalToWorldTransform(root).GetInverse()
    children = []

    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsActive() or not prim.IsDefined() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        rigid_api = UsdPhysics.RigidBodyAPI(prim)
        enabled = rigid_api.GetRigidBodyEnabledAttr().Get()
        if enabled is False:
            continue
        absolute_path = prim.GetPath().pathString
        relative_path = "." if absolute_path == root_path else absolute_path[len(root_path) + 1 :]
        if include is not None and include.search(relative_path) is None:
            continue
        if exclude is not None and exclude.search(relative_path) is not None:
            continue

        identifier = None
        if cfg.identifier_attribute:
            attribute = prim.GetAttribute(cfg.identifier_attribute)
            if attribute and attribute.IsValid() and attribute.HasAuthoredValueOpinion():
                raw_identifier = attribute.Get()
                if raw_identifier is not None:
                    identifier = str(raw_identifier).strip()
        if not identifier:
            identifier = stable_child_identifier(relative_path)

        local_matrix = xforms.GetLocalToWorldTransform(prim) * root_world_inverse
        translation = local_matrix.ExtractTranslation()
        quaternion = local_matrix.ExtractRotationQuat()
        imaginary = quaternion.GetImaginary()
        orientation = _normalize_quaternion((
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
            float(quaternion.GetReal()),
        ))
        children.append(
            CollectionRigidBodyDescriptor(
                identifier=identifier,
                relative_prim_path=relative_path,
                local_position=(
                    float(translation[0]),
                    float(translation[1]),
                    float(translation[2]),
                ),
                local_orientation=orientation,
            )
        )
    return build_collection_manifest(
        cfg.usd_path,
        root_path,
        children,
    )


def discover_legacy_leaf_rigid_bodies(
    cfg: MatterixRigidObjectCollectionCfg,
) -> RigidObjectCollectionManifest:
    """Resolve composed rigid bodies back to separately spawnable leaf assets.

    This preserves the old Orbit behavior: referenced leaves are materialized as
    independent rigid objects.  The composed hierarchy is still used for robust
    transforms and stable identifiers instead of copying the old manual xform
    parsing.
    """
    try:
        from pxr import Gf, Usd, UsdGeom
    except ImportError as error:
        raise RigidObjectCollectionError(
            "USD hierarchy discovery requires pxr from the Isaac Sim/Isaac Lab runtime"
        ) from error

    stage = Usd.Stage.Open(cfg.usd_path)
    if stage is None:
        raise RigidObjectCollectionError(f"Could not open nested USD asset {cfg.usd_path!r}")
    root = stage.GetDefaultPrim()
    if not root or not root.IsValid():
        raise RigidObjectCollectionError(f"Nested USD asset {cfg.usd_path!r} has no valid default prim")

    composed = _discover_composed_rigid_bodies(cfg)
    root_layer = stage.GetRootLayer()
    root_path = root.GetPath().pathString.rstrip("/")
    xforms = UsdGeom.XformCache()
    root_world_inverse = xforms.GetLocalToWorldTransform(root).GetInverse()
    legacy_children = []
    has_top_level_geometry = any(
        prim.IsA(UsdGeom.Gprim) and any(spec.layer == root_layer for spec in prim.GetPrimStack())
        for prim in Usd.PrimRange(root)
    )

    for child in composed.children:
        absolute_path = root_path if child.relative_prim_path == "." else f"{root_path}/{child.relative_prim_path}"
        prim = stage.GetPrimAtPath(absolute_path)
        source_spec = _rigid_body_source_spec(prim)
        if source_spec is None:
            raise LegacyCollectionMaterializationUnsupported(
                f"Could not identify the authored RigidBodyAPI source for {child.relative_prim_path!r}"
            )

        source_layer = source_spec.layer
        source_usd_path = source_layer.realPath or source_layer.resolvedPath or source_layer.identifier
        source_stage = stage if source_layer == root_layer else Usd.Stage.Open(source_usd_path)
        if source_stage is None or not source_stage.GetDefaultPrim().IsValid():
            raise RigidObjectCollectionError(f"Could not open referenced rigid leaf asset {source_usd_path!r}")
        source_rigid_count = sum(
            1
            for source_prim in Usd.PrimRange(source_stage.GetDefaultPrim(), Usd.TraverseInstanceProxies())
            if source_prim.IsActive() and source_prim.IsDefined() and _has_enabled_rigid_body_api(source_prim)
        )
        if source_rigid_count != 1:
            raise LegacyCollectionMaterializationUnsupported(
                f"Leaf source {source_usd_path!r} contains {source_rigid_count} enabled rigid bodies; "
                "legacy leaf spawning requires exactly one rigid body per source USD"
            )

        if source_layer == root_layer:
            legacy_children.append(
                dataclasses.replace(
                    child,
                    relative_prim_path=".",
                    local_position=(0.0, 0.0, 0.0),
                    local_orientation=(0.0, 0.0, 0.0, 1.0),
                    source_usd_path=cfg.usd_path,
                    local_scale=(1.0, 1.0, 1.0),
                )
            )
            continue

        placement_prim = _reference_placement_prim(stage, prim, source_stage.GetDefaultPrim(), source_spec)
        local_matrix = xforms.GetLocalToWorldTransform(placement_prim) * root_world_inverse
        decomposed = Gf.Transform(local_matrix)
        translation = decomposed.GetTranslation()
        quaternion = decomposed.GetRotation().GetQuat()
        imaginary = quaternion.GetImaginary()
        local_scale = decomposed.GetScale()
        placement_path = placement_prim.GetPath().pathString
        relative_placement_path = "." if placement_path == root_path else placement_path[len(root_path) + 1 :]
        legacy_children.append(
            dataclasses.replace(
                child,
                relative_prim_path=relative_placement_path,
                local_position=(float(translation[0]), float(translation[1]), float(translation[2])),
                local_orientation=_normalize_quaternion((
                    float(imaginary[0]),
                    float(imaginary[1]),
                    float(imaginary[2]),
                    float(quaternion.GetReal()),
                )),
                source_usd_path=source_usd_path,
                local_scale=(float(local_scale[0]), float(local_scale[1]), float(local_scale[2])),
            )
        )

    if has_top_level_geometry and any(child.source_usd_path != cfg.usd_path for child in legacy_children):
        raise LegacyCollectionMaterializationUnsupported(
            "The top-level USD authors geometry outside referenced rigid leaves; "
            "separate leaf spawning would omit that geometry"
        )

    return build_collection_manifest(
        cfg.usd_path,
        root_path,
        legacy_children,
    )


def _rigid_body_source_spec(prim):
    """Return the strongest prim spec that authors PhysicsRigidBodyAPI."""
    for prim_spec in prim.GetPrimStack():
        if prim_spec.HasInfo("apiSchemas") and "PhysicsRigidBodyAPI" in str(prim_spec.GetInfo("apiSchemas")):
            return prim_spec
    return None


def _has_enabled_rigid_body_api(prim) -> bool:
    """Return whether a prim has an enabled composed rigid-body API."""
    from pxr import UsdPhysics

    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        return False
    enabled = UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get()
    return enabled is not False


def _reference_placement_prim(stage, composed_rigid_prim, source_default_prim, source_spec):
    """Map a source-layer default prim to its composed reference placement."""
    source_root_path = source_default_prim.GetPath().pathString.rstrip("/")
    source_rigid_path = source_spec.path.pathString
    if source_rigid_path == source_root_path:
        suffix = ""
    elif source_rigid_path.startswith(f"{source_root_path}/"):
        suffix = source_rigid_path[len(source_root_path) :]
    else:
        raise LegacyCollectionMaterializationUnsupported(
            f"Rigid source prim {source_rigid_path!r} is outside default prim {source_root_path!r}"
        )
    composed_path = composed_rigid_prim.GetPath().pathString
    if suffix and not composed_path.endswith(suffix):
        raise LegacyCollectionMaterializationUnsupported(
            f"Could not map composed rigid prim {composed_path!r} to source suffix {suffix!r}"
        )
    placement_path = composed_path[: -len(suffix)] if suffix else composed_path
    placement_prim = stage.GetPrimAtPath(placement_path)
    if not placement_prim or not placement_prim.IsValid():
        raise LegacyCollectionMaterializationUnsupported(f"Reference placement prim {placement_path!r} is invalid")
    return placement_prim


@dataclasses.dataclass(frozen=True)
class MaterializedRigidObjectCollection:
    """Legacy leaf collection config plus the stable child manifest."""

    collection: object
    manifest: RigidObjectCollectionManifest


def _load_isaaclab_types():
    from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
    from isaaclab.sim import UsdFileCfg
    from isaaclab.utils import configclass

    return RigidObjectCfg, RigidObjectCollectionCfg, UsdFileCfg, configclass


def materialize_rigid_object_collection(
    cfg: MatterixRigidObjectCollectionCfg,
    manifest: RigidObjectCollectionManifest | None = None,
) -> MaterializedRigidObjectCollection:
    """Materialize every discovered referenced leaf as an independent object."""
    if manifest is None:
        manifest = discover_collection_rigid_bodies(cfg)
    return _materialize_legacy_leaf_assets(cfg, manifest)


def _materialize_legacy_leaf_assets(
    cfg: MatterixRigidObjectCollectionCfg,
    manifest: RigidObjectCollectionManifest,
) -> MaterializedRigidObjectCollection:
    """Spawn every referenced leaf separately, matching the old Orbit model."""
    RigidObjectCfg, RigidObjectCollectionCfg, UsdFileCfg, configclass = _load_isaaclab_types()
    rigid_objects = {}
    for child in manifest.children:
        if not child.source_usd_path:
            raise LegacyCollectionMaterializationUnsupported(
                f"Child {child.identifier!r} has no separately spawnable source USD"
            )
        position, orientation = _compose_child_pose(cfg, child)
        scale = tuple(root * local for root, local in zip(cfg.scale, child.local_scale, strict=True))
        rigid_objects[child.identifier] = RigidObjectCfg(
            prim_path=f"{cfg.prim_path}/{child.identifier}",
            spawn=UsdFileCfg(
                usd_path=child.source_usd_path,
                scale=scale,
                activate_contact_sensors=cfg.activate_contact_sensors,
                semantic_tags=cfg.semantic_tags or None,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=position, rot=orientation),
            collision_group=cfg.collision_group,
            debug_vis=cfg.debug_vis,
        )
    collection = _build_collection_cfg(
        RigidObjectCollectionCfg,
        rigid_objects,
        semantics=cfg.semantics,
        event_terms=cfg.event_terms,
        configclass=configclass,
    )
    return MaterializedRigidObjectCollection(
        collection=collection,
        manifest=manifest,
    )


def _build_collection_cfg(
    base_type,
    rigid_objects: dict[str, object],
    *,
    semantics: list[object],
    event_terms: dict[str, object],
    configclass,
):
    """Create a native config subclass whose Matterix fields survive ``copy()``."""

    @configclass
    class MatterixRuntimeRigidObjectCollectionCfg(base_type):
        """Native collection configuration carrying Matterix semantics/events."""

        semantics: list[object] = dataclasses.field(default_factory=list)
        event_terms: dict[str, object] = dataclasses.field(default_factory=dict)

    return MatterixRuntimeRigidObjectCollectionCfg(
        rigid_objects=rigid_objects,
        semantics=list(semantics),
        event_terms=dict(event_terms),
    )


class RigidObjectCollectionView:
    """Name-based child access over an Isaac Lab ``RigidObjectCollection``."""

    def __init__(self, collection: object, manifest: RigidObjectCollectionManifest):
        body_names = tuple(collection.body_names)
        if body_names != manifest.child_ids:
            raise RigidObjectCollectionError(
                f"Runtime collection order {body_names} differs from manifest order {manifest.child_ids}"
            )
        self.collection = collection
        self.manifest = manifest

    @property
    def child_ids(self) -> tuple[str, ...]:
        """Return addressable stable child identifiers."""
        return self.manifest.child_ids

    def object_id(self, identifier: str) -> int:
        """Resolve a stable child identifier to the native object index."""
        return self.manifest.object_id(identifier)

    def child_pose_w(self, identifier: str):
        """Return all-environment world poses for one child."""
        body_link_pose_w = self.collection.data.body_link_pose_w
        if hasattr(body_link_pose_w, "torch"):
            body_link_pose_w = body_link_pose_w.torch
        return body_link_pose_w[:, self.object_id(identifier)]

    def write_child_pose_to_sim(self, identifier: str, body_poses: object, env_ids: Sequence[int] | None = None):
        """Write ``(num_envs, 1, 7)`` poses for one named child."""
        return self.collection.write_body_pose_to_sim_index(
            body_poses=body_poses,
            body_ids=[self.object_id(identifier)],
            env_ids=env_ids,
        )

    def write_child_velocity_to_sim(
        self,
        identifier: str,
        body_velocities: object,
        env_ids: Sequence[int] | None = None,
    ):
        """Write ``(num_envs, 1, 6)`` velocities for one named child."""
        return self.collection.write_body_velocity_to_sim_index(
            body_velocities=body_velocities,
            body_ids=[self.object_id(identifier)],
            env_ids=env_ids,
        )


def _compose_child_pose(
    cfg: MatterixRigidObjectCollectionCfg,
    child: CollectionRigidBodyDescriptor,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    scaled = tuple(value * scale for value, scale in zip(child.local_position, cfg.scale, strict=True))
    rotated = _rotate_vector(cfg.rot, scaled)
    position = tuple(root + offset for root, offset in zip(cfg.pos, rotated, strict=True))
    orientation = _normalize_quaternion(_multiply_quaternions(cfg.rot, child.local_orientation))
    return position, orientation


def _normalize_quaternion(
    values: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0 or not math.isfinite(norm):
        raise RigidObjectCollectionError("Quaternion must have a finite non-zero norm")
    return tuple(value / norm for value in values)


def _multiply_quaternions(
    lhs: tuple[float, float, float, float],
    rhs: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = lhs
    rx, ry, rz, rw = rhs
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _rotate_vector(
    quaternion: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    qx, qy, qz, qw = _normalize_quaternion(quaternion)
    vx, vy, vz = vector
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


__all__ = [
    "MaterializedRigidObjectCollection",
    "LEGACY_LEAF_MATERIALIZATION",
    "LegacyCollectionMaterializationUnsupported",
    "MatterixRigidObjectCollectionCfg",
    "CollectionRigidBodyDescriptor",
    "RigidObjectCollectionCfg",
    "RigidObjectCollectionError",
    "RigidObjectCollectionManifest",
    "RigidObjectCollectionView",
    "build_collection_manifest",
    "discover_collection_rigid_bodies",
    "discover_legacy_leaf_rigid_bodies",
    "materialize_rigid_object_collection",
    "stable_child_identifier",
]

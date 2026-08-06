from pathlib import Path


TASK_SOURCE = Path(__file__).parents[1] / "source/matterix_tasks/matterix_tasks/test_dev_tasks/test_ticket0c_small_vessel.py"


def test_task_does_not_override_materials_at_imported_asset_root() -> None:
    """Visual materials must come from per-prim asset bindings, not the task root."""
    source = TASK_SOURCE.read_text(encoding="utf-8")

    env_source = (
        TASK_SOURCE.parents[4]
        / "source"
        / "matterix"
        / "matterix"
        / "envs"
        / "matterix_base_env_cfg.py"
    ).read_text(encoding="utf-8")

    assert "GlassMdlCfg" not in source
    assert "self.spawn.visual_material_path" not in source
    assert "self.spawn.visual_material" not in source
    assert "The visual layer owns its per-prim material bindings" in source
    assert '"rtx.material.translucencyAsOpacity": True' in env_source

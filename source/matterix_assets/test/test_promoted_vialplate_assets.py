# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static checks for the promoted vial and holder payloads."""

import hashlib
import json
from pathlib import Path


MATTERIX_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = MATTERIX_ROOT / "source/matterix_assets/data/labware"
VIAL_ROOT = DATA_ROOT / "fisherbrand_03-339-21f"
HOLDER_ROOT = DATA_ROOT / "vialplate_3_5"
TASK_PATH = MATTERIX_ROOT / "source/matterix_tasks/matterix_tasks/test_dev_tasks/test_franka_vialplate.py"


def test_promoted_payloads_have_complete_static_contract():
    """Verify files, frame count, hashes, licenses, and official-only task paths."""
    required_files = [
        VIAL_ROOT / "fisherbrand_03-339-21f_z_up_fixed.usda",
        VIAL_ROOT / "files/fisherbrand_03-339-21f_z_up_fixed_mesh.usda",
        VIAL_ROOT / "files/Fisherbrand_Vial_Z_UP_FIXED.usdc",
        VIAL_ROOT / "provenance/LICENSE.asset.txt",
        VIAL_ROOT / "provenance/NOTICE.md",
        VIAL_ROOT / "provenance/provenance.yaml",
        HOLDER_ROOT / "3_5_vialplate_free_standing.usda",
        HOLDER_ROOT / "3_5_vialplate_free_standing_frames.usda",
        HOLDER_ROOT / "files/3_5_vialplate_free_standing_mesh.usda",
        HOLDER_ROOT / "holder-hole-frame-contract.json",
        HOLDER_ROOT / "provenance/LICENSE.asset.txt",
        HOLDER_ROOT / "provenance/NOTICE.md",
        HOLDER_ROOT / "provenance/asset_metadata.yaml",
        HOLDER_ROOT / "provenance/license_record_draft.txt",
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    assert not missing, f"missing promoted payload files: {missing}"

    contract_path = HOLDER_ROOT / "holder-hole-frame-contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    frames = contract["frames"]
    assert len(frames) == 15
    assert len({frame["name"] for frame in frames}) == 15
    selection = contract["initial_dynamic_vial_set"]
    assert selection["pick_vial"] == "hole_middle_center"
    assert len(selection["witness_vials"]) == 3

    metadata = (HOLDER_ROOT / "provenance/asset_metadata.yaml").read_text(encoding="utf-8")
    assert "original_license: Public Domain" in metadata
    assert "package_license: CC0-1.0-Universal" in metadata
    assert hashlib.sha256(contract_path.read_bytes()).hexdigest() in metadata

    holder_license = (HOLDER_ROOT / "provenance/license_record_draft.txt").read_text(encoding="utf-8")
    assert "https://3d.nih.gov/entries/3DPX-000429" in holder_license
    assert "Public Domain" in holder_license
    assert "CC BY 4.0" not in holder_license
    assert "Pending" not in holder_license

    vial_provenance = (VIAL_ROOT / "provenance/provenance.yaml").read_text(encoding="utf-8")
    assert "asset_status: promoted_to_official_matterix_data" in vial_provenance
    assert "canonical_visual_source_relative_path: not_packaged_in_official_data" in vial_provenance
    assert "creation_method: user_authored_self_modelled_from_official_specification_and_dimensions" in vial_provenance
    assert "license_status: CC0-1.0-Universal" in vial_provenance
    assert "manufacturer_cad_or_texture_copied: false" in vial_provenance

    vial_notice = (VIAL_ROOT / "provenance/NOTICE.md").read_text(encoding="utf-8")
    assert "This promoted Matterix asset" in vial_notice
    assert "The candidate is" not in vial_notice

    task = TASK_PATH.read_text(encoding="utf-8")
    assert "MATTERIX_PHASE_B_ASSETS_ROOT" not in task
    assert "MATTERIX_VIAL_USD" not in task
    assert "asset_workbench/phase_b_candidates" not in task

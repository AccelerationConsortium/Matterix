# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from .is_in_contact import IsInContact, IsInContactCfg, IsInContactManual, IsInContactPhysics, IsInContactPhysicsCfg

__semantics_cfg__ = [
    "IsInContactCfg",
    "IsInContactPhysicsCfg",
]

__semantics__ = [
    "IsInContactManual",
    "IsInContactPhysics",
]

__all__ = __semantics_cfg__ + __semantics__ + ["IsInContact"]

# Create a dictionary mapping class names to class objects
semantics_dict = {}

from .heat_transfer import heat_transfer_semantics_dict

semantics_dict.update(heat_transfer_semantics_dict)

for class_name in __semantics__:
    class_obj = globals()[class_name]
    semantics_dict[class_name] = class_obj

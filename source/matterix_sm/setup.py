# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'matterix_sm' python package."""

from setuptools import find_packages, setup

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "torch>=2.0.0",
    "numpy>=1.23.0",
]

# Try to include Isaac Lab by default for full functionality
# Fall back gracefully if not available (e.g., edge deployment)
try:
    import isaaclab  # noqa: F401

    # Isaac Lab is available, include it in base requirements
    INSTALL_REQUIRES.append("isaaclab")
except ImportError:
    # Isaac Lab not available - will use minimal install
    pass

# Optional dependencies for explicit control
EXTRAS_REQUIRE = {
    "full": ["isaaclab"],  # Explicitly request Isaac Lab: pip install -e .[full]
    "minimal": [],  # Explicitly request minimal: pip install -e .[minimal]
}

# Installation operation
setup(
    name="matterix_sm",
    author="Matterix Project Developers",
    maintainer="Matterix Project Developers",
    url="https://github.com/ac-rad/Matterix.git",
    version="0.1.0",
    description="State machine library for sequential action orchestration in robotic manipulation",
    keywords=["robotics", "state-machine", "manipulation", "matterix"],
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    packages=find_packages(),
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    zip_safe=False,
)

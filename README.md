# Matterix


[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)


**Matterix** is a multi-scale, GPU-accelerated robotic simulation framework designed to create high-fidelity digital twins of chemistry labs, thus accelerating workflow development. This multi-scale digital twin simulates robotic physical manipulation, powder and liquid dynamics, device functionalities, heat transfer, and basic chemical reaction kinetics. This is enabled by integrating realistic physics simulation and photorealistic rendering with a modular GPU-accelerated semantics engine, which models logical states and continuous behaviors to simulate chemistry workflows across different levels of abstraction.

## Key Features

The key features of Matterix are:

* **Multi-scale simulation with integrated PhysX and semantics engines**: supports robotics, soft bodies, liquids, powders, heat transfer, chemical kinetics, and device functions. Enables simulation of various vectorized logical events and continuous processes.
* **Digital twin design**: Easily create environments with multiple agents using an extensive asset library in wet lab domain.
* **Workflow automation**: Build and run experimental workflows with minimal effort.
* **Real-world deployment**: Deploy virtual workflows to physical robots and lab setups.


## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Install git lfs with `git lfs install`

- Clone this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):
    ```bash
    # Matterix has submodules
    git clone  --recurse-submodules https://github.com/ac-rad/Matterix.git
    cd Matterix
    git submodule foreach 'git lfs pull'
    ```

- If you are using the Isaac Sim python interpreter (i.e., **not** using Python venv or conda):
    ```bash
    # Ensure ${ISAACLAB_PATH} and ${ISAACSIM_PATH} are set in your terminal before running:
    ./matterix.sh -i
    ```

- If you are using conda that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # Activate your Isaac Lab Conda environment
    conda activate <isaaclab-conda-env-name>

    # Install Matterix in editable mode
    python -m pip install -e source/*

    # ─── Fallback: If Isaac Lab is not installed in the active env ───
    # Ensure ${ISAACLAB_PATH} and ${ISAACSIM_PATH} are set in your terminal before running:
    ./matterix.sh -p -m pip install -e source/*
    ```

- If you are using Python venv that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # Activate your virtual environment
    source <path-to-your-venv>/bin/activate  # On Linux/macOS
    # .\<path-to-your-venv>\Scripts\activate  # On Windows

    # Install Matterix in editable mode
    python -m pip install -e source/*
    ```




- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: It the task name changes, it may be necessary to update the search pattern `"Template-"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
        ```

    - Running a task with dummy agents:

        These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

        - Zero-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/zero_agent.py --task=<TASK_NAME>

            # example of an environment with two robot arms and several beakers:
            python scripts/zero_agent.py --task Matterix-Test-Beakers-Franka-v1 --num_envs 1
            # or
            ./matterix.sh -p scripts/zero_agent.py --task Matterix-Test-Beakers-Franka-v1 --num_envs 1

            ```
        - Random-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/random_agent.py --task=<TASK_NAME>

            # example of an environment with two robot arms and several beakers:
            python scripts/random_agent.py --task Matterix-Test-Beakers-Franka-v1 --num_envs 1
            # or
            ./matterix.sh -p scripts/random_agent.py --task Matterix-Test-Beakers-Franka-v1 --num_envs 1

            ```

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to `(1) add the absolute path to your Isaac Sim` and ` (1) add the absolute path to your Isaac Lab installation`.

If everything executes correctly, `settings.json` (and `launch.json` if missing) will be created in the .vscode directory.
The file contains the python paths to all the extensions provided by Isaac Sim, Isaac Lab, and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `source/matterix/matterix/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to the `source` directory of this project/repository.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/Matterix"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```

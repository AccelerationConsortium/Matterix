# Matterix State Machine (matterix_sm)

A standalone state machine library for sequential action orchestration in robotic manipulation tasks.

## Overview

`matterix_sm` provides a GPU-accelerated state machine system for orchestrating sequential robotic actions across parallel environments. It supports both primitive actions (move, gripper control) and compositional actions (pick, place sequences).

## Features

- **Vectorized execution**: Batch operations across multiple parallel environments on GPU
- **Hierarchical actions**: Primitive actions and compositional action sequences
- **Frame-based manipulation**: Standardized object interaction via manipulation frames
- **Config-driven design**: Define action sequences declaratively via configuration classes
- **Factory pattern**: Create actions from configs with automatic type dispatch

## Installation

### Default Installation (Recommended)
The package automatically detects if Isaac Lab is available and includes it:
```bash
pip install -e .
```

**Behavior**:
- If Isaac Lab is already installed → Full functionality (all actions work)
- If Isaac Lab is NOT installed → Minimal install (configs only, lazy loading for actions)

### Explicit Installation Modes

**Force full install** (explicitly install Isaac Lab):
```bash
pip install -e .[full]
```

**Force minimal install** (skip Isaac Lab even if available):
```bash
pip install -e .[minimal]
```

**Note**: Primitive actions (Move, MoveToFrame, etc.) require Isaac Lab at runtime. The package will provide helpful error messages if you try to use them without Isaac Lab installed.

## Quick Start

```python
from matterix_sm import StateMachine, PickObject

# Create state machine
sm = StateMachine(env)

# Define action sequence
pick_action = PickObject(
    object="beaker",
    asset="robot",
    device="cuda",
    timeout=100
)

sm.set_action_sequence([pick_action])

# Execute
while not (sm.action_sequence_success | sm.action_sequence_failure).all():
    action = sm.step()
    obs, reward, done, info = env.step(action)
```

## Architecture

### Action Hierarchy

```
ActionBase
├── PrimitiveAction
│   ├── Move
│   ├── MoveToFrame
│   ├── MoveRelative
│   ├── GripperAction
│   ├── OpenGripper
│   └── CloseGripper
└── CompositionalAction
    └── PickObject
```

### Core Components

- **StateMachine**: Orchestrates action sequences across parallel environments
- **WorkflowEnv**: Abstract interface for scene state access (objects, robots)
- **ActionBase**: Base class with factory pattern for creating actions from configs
- **Data structures**: Pose, ObjectState, RobotState

### Config-Driven Design

Actions are configured declaratively:

```python
from matterix_sm import PickObjectCfg

cfg = PickObjectCfg(
    object="beaker",
    asset="robot",
    num_envs=16,
    device="cuda",
    timeout=100
)

# Create action from config
action = PickObject.from_cfg(cfg)
```

## Dependencies

### Required (Minimal Install)
- **PyTorch** >= 2.0.0 - Tensor operations and GPU acceleration
- **NumPy** >= 1.23.0 - Numerical computations

### Optional (Full Install)
- **Isaac Lab** - Needed for primitive actions and full functionality:
  - IK solvers for robotic manipulation
  - Math utilities (frame transforms, rotations)
  - Environment integration (WorkflowEnv interface)
  - Enhanced `@configclass` decorator with validation

### Use Cases

**Development/Training** (install with `[full]`):
- Developing new actions and workflows
- Training policies in Isaac Sim
- Full access to all primitive actions (Move, MoveToFrame, etc.)

**Edge Deployment** (minimal install):
- Running trained policies on edge devices
- Using config classes for serialization
- Custom lightweight action implementations
- Scenarios where Isaac Sim cannot be installed

**Lazy Loading**: Isaac Lab dependencies are only imported when primitive actions are actually used, allowing minimal installs for edge deployment while providing helpful error messages if you try to use actions without Isaac Lab.

## License

BSD-3-Clause

## Contributing

Part of the [Matterix project](https://github.com/ac-rad/Matterix).

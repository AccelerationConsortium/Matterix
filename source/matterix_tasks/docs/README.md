# MATteRIX: Environment Suite

Using the core framework developed as part of MATteRIX, we provide various learning environments for robotics and lab automation research.
These environments follow the `gym.Env` API from OpenAI Gym version `0.21.0`. The environments are registered using
the Gym registry.

Each environment's name is composed of `Isaac-<Task>-<Robot>-v<X>`, where `<Task>` indicates the skill to learn
in the environment, `<Robot>` indicates the embodiment of the acting agent, and `<X>` represents the version of
the environment (which can be used to suggest different observation or action spaces).

The environments are configured using either Python classes (wrapped using `configclass` decorator) or through
YAML files. The template structure of the environment is always put at the same level as the environment file
itself. However, its various instances are included in directories within the environment directory itself.
For more information, please refer to: https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/docs/README.md

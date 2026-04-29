# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from .primitive_semantics import semantics_dict
from .semantics import SemanticPredicate, Semantics, SemanticState, SemanticStateTransition
from .semantics_cfg import semantic_info


def initialize_semantics(obj: type, name: str):
    """
    Initializes the semantics for an object by initializing the semantic list.
    Args:
        obj: The object to initialize semantics for (RigidObject, ArticulatedObject)
        name: class name
    """

    obj.semantic_list = []
    # obj.name = name


class SemanticManager:
    def __init__(self, env):
        self.env = env
        print("[sem manager] env device", self.env.device)

        # loop through self.cfg.semantics
        self.env.semantic_list: list[Semantics] = []

        self.full_semantic_states: dict[str, Semantics] = {}
        self.full_semantic_state_transitions: dict[str, Semantics] = {}
        self.full_semantic_predicates: dict[str, Semantics] = {}

        # environment level semantics
        if self.env.cfg.semantics is not None:
            for semantic_cfg in self.env.cfg.semantics:
                try:
                    self.env.semantic_list.append(
                        semantics_dict[semantic_cfg.type](
                            self.env, cfg=semantic_cfg, parent_asset=None, name=f"env/{semantic_cfg.type}"
                        )
                    )
                    self.add_env_semantics(self.env.semantic_list[-1])
                    print(f"[INFO] Initializing semantic {self.env.semantic_list[-1].name}.")
                except KeyError:
                    raise KeyError(
                        f"Class '{semantic_cfg.type}' not found in semantics_dict. semantics_dict keys are:"
                        f" {semantics_dict.keys()}"
                    )

        # asset level semantics (rigid objects, articulations, and static objects)
        # First process objects that have runtime .cfg attribute (RigidObject, Articulation)
        for object_name in self.env.scene.keys():
            object_val = self.env.scene[object_name]
            if object_val is not None:
                initialize_semantics(object_val, object_name)
                # check if object has cfg attribute with semantics
                if hasattr(object_val, "cfg") and hasattr(object_val.cfg, "semantics"):
                    for semantic_cfg in object_val.cfg.semantics:
                        object_val.semantic_list.append(
                            semantics_dict[semantic_cfg.type](
                                self.env,
                                cfg=semantic_cfg,
                                parent_asset=object_val,
                                name=f"{object_name}/{semantic_cfg.type}",
                            )
                        )
                        print(f"[INFO] Initializing semantic {object_val.semantic_list[-1].name}.")
                        self.add_env_semantics(object_val.semantic_list[-1])

        # Process objects from env cfg (for static objects that don't have runtime .cfg)
        if hasattr(self.env.cfg, "objects"):
            for object_name, object_cfg in self.env.cfg.objects.items():
                if hasattr(object_cfg, "semantics") and object_cfg.semantics:
                    # Check if object exists in scene
                    if object_name in self.env.scene.keys():
                        object_val = self.env.scene[object_name]
                        if object_val is not None and not hasattr(object_val, "cfg"):
                            # Static object without runtime cfg - initialize from env cfg
                            initialize_semantics(object_val, object_name)
                            for semantic_cfg in object_cfg.semantics:
                                object_val.semantic_list.append(
                                    semantics_dict[semantic_cfg.type](
                                        self.env,
                                        cfg=semantic_cfg,
                                        parent_asset=object_val,
                                        name=f"{object_name}/{semantic_cfg.type}",
                                    )
                                )
                                print(
                                    f"[INFO] Initializing semantic {object_val.semantic_list[-1].name} (from env cfg)."
                                )
                                self.add_env_semantics(object_val.semantic_list[-1])

    def initialize(self):
        # do post initialization // adding this later
        for semantic in self.full_semantic_states.values():
            semantic.init()
        for semantic in self.full_semantic_state_transitions.values():
            semantic.init()
        for semantic in self.full_semantic_predicates.values():
            semantic.init()

    def add_env_semantics(self, semantics_object: Semantics):
        if isinstance(semantics_object, SemanticState):
            self.full_semantic_states[semantics_object.name] = semantics_object
        elif isinstance(semantics_object, SemanticStateTransition):
            self.full_semantic_state_transitions[semantics_object.name] = semantics_object
        elif isinstance(semantics_object, SemanticPredicate):
            self.full_semantic_predicates[semantics_object.name] = semantics_object
        else:
            raise TypeError(
                f"Semantic object with name: {semantics_object.name} and type:{semantics_object.__class__.__name__} is"
                " not among the valid types and inheritance."
            )

    def step(self, semantic_actions: list[semantic_info] = None):
        # set the semantic states provided by semantic actions
        if semantic_actions is not None:
            for semantic_action in semantic_actions:
                semantic_name = f"{semantic_action.asset_name}/{semantic_action.type}"
                try:
                    print("semantic_name", semantic_name)
                    print("self.full_semantic_predicates", self.full_semantic_predicates.items())
                    self.full_semantic_predicates[semantic_name].set_value(semantic_action)
                except KeyError:
                    raise KeyError(
                        f"semantic name '{semantic_name}' not found in full_semantic_predicates in semantic manager."
                        f" full_semantic_predicates keys are: {self.full_semantic_predicates.keys()}"
                    )

        # step predicate semantics
        for semantic in self.full_semantic_predicates.values():
            semantic.step()
        # step state transition semantics
        for semantic in self.full_semantic_state_transitions.values():
            semantic.step()
        # step semantics and physics states
        for semantic in self.full_semantic_states.values():
            semantic.step()

    def reset(self, env_ids):
        if env_ids is None:
            env_ids = ...

        for semantic in self.full_semantic_predicates.values():
            semantic.reset(env_ids=env_ids)
        # reset state transition semantics
        for semantic in self.full_semantic_state_transitions.values():
            semantic.reset(env_ids=env_ids)
        # reset semantics and physics states
        for semantic in self.full_semantic_states.values():
            semantic.reset(env_ids=env_ids)

# Copyright (c) 2022-2026, The Matterix Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Frame buffer + mp4 writer used by ``MatterixBaseEnv``.

The recorder is bound to a single env at construction. The env exposes thin
``start_recording`` / ``save_video`` / ``stop_recording`` methods that forward
here; all validation, fps derivation, and moviepy I/O live in this class.
"""

from __future__ import annotations

import numpy as np
import os
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matterix.envs.matterix_base_env import MatterixBaseEnv


class RealTimeVideoRecorder:
    """Owns the frame buffer for one ``MatterixBaseEnv``.

    Idle by default. ``start()`` opens a buffer; subsequent ``capture_step()``
    calls (driven from the env's ``step()``) append frames; ``save(path)`` flushes
    them to an mp4 and returns to idle.
    """

    def __init__(self, env: MatterixBaseEnv) -> None:
        self.env = env
        self.frames: list[np.ndarray] | None = None

    @property
    def is_recording(self) -> bool:
        return self.frames is not None

    def start(self) -> None:
        """Open the buffer. Raises if render_mode is wrong, warns on double-start."""
        if self.env.render_mode != "rgb_array":
            raise RuntimeError(
                "Cannot start video recording: env.render_mode is "
                f"{self.env.render_mode!r}, but recording requires 'rgb_array'. "
                "render_mode is fixed at construction, so set it at the gym.make call:\n"
                "    env = gym.make(task, cfg=cfg, render_mode='rgb_array')"
            )
        if self.frames is not None:
            warnings.warn(
                "start_recording() called while recording is already in progress; "
                "previous frames discarded. Call save_video() or stop_recording() "
                "first to avoid this warning."
            )
        self.frames = []

    def capture_step(self) -> None:
        """If recording, render one frame from the env and append it."""
        if self.frames is None:
            return
        frame = self.env.render()
        if frame is not None:
            self.frames.append(frame)

    def save(self, path: str) -> None:
        """Flush the buffer to ``path`` as an mp4 and return to idle.

        fps is derived from the env's ``cfg.sim.dt * cfg.decimation`` so playback
        matches sim time. Raises if recording was never started.
        """
        if self.frames is None:
            raise RuntimeError(
                "save_video() called but recording was never started. "
                "Call env.start_recording() before stepping the environment."
            )
        frames = self.frames
        self.frames = None
        if not frames:
            return
        fps = getattr(self.env.cfg, "_recording_fps", None)
        if fps is None:
            fps = int(round(1.0 / (self.env.cfg.sim.dt * self.env.cfg.sim.render_interval)))
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

        abs_path = os.path.abspath(path)
        parent = os.path.dirname(abs_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        ImageSequenceClip(frames, fps=fps).write_videofile(abs_path, logger=None)

    def stop(self) -> None:
        """Discard the buffer without saving."""
        self.frames = None

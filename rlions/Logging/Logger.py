#  RLIonS - Reinforcement Learning Ion Shuttling Compiler
#  Copyright (C) 2026 Maximilian Schier, Lea Richtmann
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import Dict, Optional, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class ImageObject:
    """
    Image data in RGB8 and shape (height, width, 3)
    """
    rgb_array: np.ndarray


@dataclass
class VideoObject:
    """
    Video frame data in shape (number_frames, height, width, channels)
    """
    frames: np.ndarray

    """
    Frames per second
    """
    fps: int


_active_instance: Optional['Logger'] = None

LogDictType = Dict[str, Union[None, float, ImageObject, VideoObject]]


class Logger:
    def __init__(self):
        self._active_exception = None
    
    def __enter__(self):
        global _active_instance
        self.open()
        _active_instance = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _active_instance
        self._active_exception = None if exc_type is None else (exc_type, exc_val, exc_tb)
        self.close()
        self._active_exception = None
        _active_instance = None

    def open(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def log_dict(self, metrics: LogDictType, step: int):
        raise NotImplementedError


def log_dict(metrics: LogDictType, step: int):
    if not _active_instance:
        raise ValueError("No active logger")

    _active_instance.log_dict(metrics, step)

import numpy as np

from typing import List, Any
from abc import ABC, ABCMeta, abstractmethod

class Padder(ABC):
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def __call__(self, frames: np.ndarray = None, annotations: List[Any] = None, target_segment_size: int = None):
        pass
    
class ValuePadder(Padder):
    def __init__(self, frames_padding_value: Any, annotations_padding_value: Any):
        self.frames_padding_value = frames_padding_value
        self.annotations_padding_value = annotations_padding_value
        
    def __call__(self, frames: np.ndarray = None, annotations: List[Any] = None, target_segment_size: int = None):
        assert target_segment_size != None
        
        frames_padded = self.__pad_frames(frames, target_segment_size) if frames is not None else None
        annotations_padded = self.__pad_annotations(annotations, target_segment_size) if annotations is not None else None
        
        return frames_padded, annotations_padded
    
    def __pad_frames(self, frames: np.ndarray, target_segment_size: int):
        frames_padded = np.full((target_segment_size, *frames.shape[1:]), self.frames_padding_value, dtype=frames.dtype)
        frames_padded[:frames.shape[0]] = frames
        
        return frames_padded
    
    def __pad_annotations(self, annotations: List[Any], target_segment_size: int):
        annotations_padded = [self.annotations_padding_value] * target_segment_size
        annotations_padded[:len(annotations)] = annotations
        
        return annotations_padded
    
class LastValuePadder(Padder):
    def __init__(self):
        pass
    
    def __call__(self, frames: np.ndarray = None, annotations: List[Any] = None, target_segment_size: int = None):
        assert target_segment_size != None
        
        frames_padded = self.__pad_frames(frames, target_segment_size) if frames is not None else None
        annotations_padded = self.__pad_annotations(annotations, target_segment_size) if annotations is not None else None
        
        return frames_padded, annotations_padded
    
    def __pad_frames(self, frames: np.ndarray, target_segment_size: int):
        pad_length = target_segment_size - frames.shape[0]
        
        if pad_length <= 0:
            return frames  # No need to pad
        
        last_frame = frames[-1:]  # Correctly extract the last frame
        frames_padded = np.vstack([frames] + [last_frame] * pad_length)  # Stack repeated frames

        return frames_padded
    
    def __pad_annotations(self, annotations: List[Any], target_segment_size: int):
        annotations_padded = [annotations[-1]] * (target_segment_size - len(annotations))
        annotations_padded = annotations + annotations_padded
        
        return annotations_padded
import os

from abc import ABC, ABCMeta, abstractmethod

class Annotations(ABC):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, annotations_dir_path: str, id: str):
        pass
    
    @abstractmethod
    def get_id(self):
        pass
    
    @abstractmethod
    def __getitem__(self, index: int | slice):
        """
        Get the annotation(s) of the video file corresponding to the given frame(s) index / indices.
        Note that even if an index is given the annotations will be returned in a batch format (Number of frames, Height, Width, Channels).
        """
        pass
    
PREFILL_VALUE = "nothing"
MAX_OVERFLOW_VALUE = 100
    
class AnnotationsFromFrameLevelTxtFileAnnotations(Annotations):
    def __init__(self, annotations_dir_path, id, prefill_value = PREFILL_VALUE, max_overflow_value = MAX_OVERFLOW_VALUE):
        self.annotations_dir_path = annotations_dir_path
        self.id = id
        self.prefill_value = prefill_value
        self.max_overflow_value = max_overflow_value
        
    def get_id(self):
        return self.id
    
    def __len__(self):
        with open(self.path, 'r') as f:
            lines = f.readlines()
            
        return len(lines)
    
    def __getitem__(self, index: int | slice):
        if isinstance(index, int):
            if index < 0 or index >= self.__len__() + self.max_overflow_value:
                raise IndexError("Index out of bounds")
            return self.__get_annotation(index)
        elif isinstance(index, slice):
            start, stop, step = index.indices(self.__len__() + self.max_overflow_value)
            return self.__get_annotations(start, stop, step)
        else:
            raise TypeError("Index must be an integer or slice")
        
    def __get_annotation(self, index: int):
        with open(os.path.join(self.annotations_dir_path, self.id), 'r') as f:
            lines = f.readlines()
        
        if index >= len(lines):
            return self.prefill_value
            
        # each line is an annotation for a frame
        return lines[index]
        
    def __get_annotations(self, start: int, stop: int, step: int):
        with open(os.path.join(self.annotations_dir_path, self.id), 'r') as f:
            lines = f.readlines()
            
        # each line is an annotation for a frame
        
        annotations = []
        for i in range(start, stop, step):
            if i >= len(lines):
                annotations.append(self.prefill_value)
            else:
                annotations.append(lines[i])
        
        return annotations
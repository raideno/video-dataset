import os
import csv

from typing import Union
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
    
class UndefinedAnnotationsException(Exception):
    """
    Raised when no valid annotations file is found for a given video ID.
    """
    
    def __init__(self, message: str, id: str):
        super().__init__(message)
        
        self.message = message
        self.id = id
    
FALLBACK_ANNOTATION = "nothing"
ALLOWED_INDEX_OVERFLOW = 100
    
class AnnotationsFromFrameLevelTxtFileAnnotations(Annotations):
    def __init__(self, annotations_dir_path, id, fallback_annotation = FALLBACK_ANNOTATION, max_overflow_value = ALLOWED_INDEX_OVERFLOW):
        self.annotations_dir_path = annotations_dir_path
        self.id = id
        self.fallback_annotation = fallback_annotation
        self.max_overflow_value = max_overflow_value
        
        if not self.__does_annotations_file_exists():
            raise UndefinedAnnotationsException(f"No valid annotations file found for {id} in {annotations_dir_path}", self.id)
        
    def __does_annotations_file_exists(self):
        return os.path.exists(os.path.join(self.annotations_dir_path, f"{self.id}.txt"))
        
    def get_id(self):
        return self.id
    
    def __len__(self):
        with open(os.path.join(self.annotations_dir_path, f"{self.id}.txt"), 'r') as f:
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
        with open(os.path.join(self.annotations_dir_path, f"{self.id}.txt"), 'r') as f:
            lines = f.readlines()
        
        if index >= len(lines):
            return self.fallback_annotation
            
        # each line is an annotation for a frame
        return lines[index]
        
    def __get_annotations(self, start: int, stop: int, step: int):
        with open(os.path.join(self.annotations_dir_path, f"{self.id}.txt"), 'r') as f:
            lines = f.readlines()
            
        # each line is an annotation for a frame
        
        annotations = []
        for i in range(start, stop, step):
            if i >= len(lines):
                annotations.append(self.fallback_annotation)
            else:
                annotations.append(lines[i])
        
        return annotations
    
DEFAULT_CSV_DELIMITER = ';'

class AnnotationsFromSegmentLevelCsvFileAnnotations(Annotations):
    def __init__(self, annotations_dir_path: str, id: str, fps: int, fallback_annotation=FALLBACK_ANNOTATION, max_overflow_value=ALLOWED_INDEX_OVERFLOW, delimiter=DEFAULT_CSV_DELIMITER):
        self.annotations_dir_path = annotations_dir_path
        self.id = id
        self.fps = fps
        self.fallback_annotation = fallback_annotation
        self.max_overflow_value = max_overflow_value
        self.delimiter = delimiter
        
        if not self.__does_annotations_file_exists():
            raise UndefinedAnnotationsException(f"No valid annotations file found for {id} in {annotations_dir_path}", self.id)
        
        self.annotations = self.__load_annotations()
        
    def __does_annotations_file_exists(self):
        return os.path.exists(os.path.join(self.annotations_dir_path, f"{self.id}.csv"))
        
    def __load_annotations(self):
        """
        Given a csv file with annotations for segments, load the annotations into a list of frame level annotations.
        """
        annotations = []
        max_frame = 0
        
        with open(os.path.join(self.annotations_dir_path, f"{self.id}.csv"), newline='') as file:
            csv_reader = csv.DictReader(file, delimiter=self.delimiter)
            
            for row in csv_reader:
                start_frame = int(float(row['starting-timestamp']) * self.fps / 1000)
                end_frame = int(float(row['ending-timestamp']) * self.fps / 1000)
                
                if end_frame > max_frame:
                    max_frame = end_frame
                
                while len(annotations) < start_frame:
                    annotations.append(self.fallback_annotation)
                
                annotations.extend([row['action']] * (end_frame - start_frame + 1))
            
            return annotations
        
    def get_id(self):
        return self.id
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index: Union[int, slice]):
        if isinstance(index, int):
            if index < 0 or index >= len(self) + self.max_overflow_value:
                raise IndexError("Index out of bounds")
            return self.__get_annotation(index)
        elif isinstance(index, slice):
            start, stop, step = index.indices(len(self) + self.max_overflow_value)
            return self.__get_annotations(start, stop, step)
        else:
            raise TypeError("Index must be an integer or slice")
        
    def __get_annotation(self, index: int):
        if index >= len(self.annotations):
            return self.fallback_annotation
        return self.annotations[index]
    
    def __get_annotations(self, start: int, stop: int, step: int):
        return [self.__get_annotation(i) for i in range(start, stop, step)]
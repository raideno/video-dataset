import os
import bisect
import itertools

from enum import IntEnum
from typing import Type, Any, Tuple, Dict, List, Optional, Callable
from pydantic import BaseModel, Field, FilePath, DirectoryPath, PositiveInt, NonNegativeInt, field_validator

from video_dataset.video import Video
from video_dataset.padder import Padder
from video_dataset.utils import better_listdir
from video_dataset.annotations import Annotations

class VideoShapeComponents(IntEnum):
    TIME = 0
    HEIGHT = 1
    WIDTH = 2
    CHANNELS = 3
    
DEFAULT_VIDEO_SHAPE = (VideoShapeComponents.TIME, VideoShapeComponents.HEIGHT, VideoShapeComponents.WIDTH, VideoShapeComponents.CHANNELS)

class VideoDatasetConfig(BaseModel):
    annotations_dir: DirectoryPath
    videos_dir: DirectoryPath
    video_processor: Type[Video]
    annotations_processor: Type[Annotations]
    segment_size: PositiveInt
    verbose: bool = True
    video_extension: str = 'mp4'
    annotations_extension: str = 'csv'
    video_shape: Tuple[NonNegativeInt, NonNegativeInt, NonNegativeInt, NonNegativeInt] = Field(default=DEFAULT_VIDEO_SHAPE)
    step: Optional[PositiveInt] = 1
    
    video_processor_kwargs: Optional[Dict[str, Any]] = {}
    annotations_processor_kwargs: Optional[Dict[str, Any]] = {}
    ids_file: Optional[FilePath] = None
    frames_transform: Optional[Callable] = None
    annotations_transform: Optional[Callable] = None
    
    padder: Optional[Any] = None
    
    overlap: Optional[NonNegativeInt] = 0

    @field_validator("video_processor")
    def check_video_processor(cls, v):
        if not issubclass(v, Video):
            raise ValueError("Video processor must inherit from Video class.")
        return v

    @field_validator("annotations_processor")
    def check_annotations_processor(cls, v):
        if not issubclass(v, Annotations):
            raise ValueError("Annotations processor must inherit from Annotations class.")
        return v

    @field_validator("video_shape")
    def check_video_shape(cls, v):
        if not isinstance(v, tuple) or len(v) != 4 or len(set(v)) != 4:
            raise ValueError("Video shape must have exactly 4 unique components.")
        return v

    @field_validator("video_processor_kwargs", "annotations_processor_kwargs", mode="before")
    def check_kwargs(cls, v):
        if v is not None and not isinstance(v, dict):
            raise ValueError("Processor kwargs must be a dictionary or None.")
        return v

    @field_validator("frames_transform", "annotations_transform", mode="before")
    def check_transform_callable(cls, v):
        if v is not None and not callable(v):
            raise ValueError("Transform must be callable or None.")
        return v
    
    # @field_validator("padder")
    # def check_padder(cls, v):
    #     if v is not None and not isinstance(v, Padder):
    #         raise ValueError("Padder must be an instance of a subclass of Padder.")
    #     return v
    
class VideoDataset():
    """
    A dataset class for loading and processing videos and their corresponding annotations.

    The class allows for flexible video and annotation handling, including:
    - Video and annotation files stored in separate directories.
    - Videos processed in either frame format or video format.
    - Support for custom annotations formats through object-oriented principles.
    - The ability to apply transformations to both video frames and annotations.
    - Custom segmentation of video frames into smaller segments of specified size.

    Attributes:
        annotations_dir (str): Path to the directory containing annotation files.
        videos_dir (str): Path to the directory containing video files.
        segment_size (int): The size of each segment (in frames) when splitting videos.
        frames_transform (callable, optional): A transformation function to apply to video frames.
        annotations_transform (callable, optional): A transformation function to apply to annotations.
        ids (List[str], optional): A list of specific video/annotation IDs to load (if None, all are loaded).
        verbose (bool): Whether to print detailed logs for operations (default is True).
        video_processor (Type[Video]): A class implementing the video processing logic.
        annotations_processor (Type[Annotations]): A class implementing the annotation processing logic.
        videos_paths (List[str]): List of video file paths.
        annotations_paths (List[str]): List of annotation file paths.
        videos (List[Video]): List of processed video objects.
        annotations (List[Annotations]): List of processed annotation objects.

    Note:
        - The videos in the `videos_dir` and annotations in the `annotations_dir` must have the same base names (different extensions allowed).
        - The class supports segmentation of videos into smaller frame chunks to optimize training.
        - Custom formats can be supported by implementing corresponding processors for videos and annotations.
    """
    def __init__(self, **kwargs):
        configuration = VideoDatasetConfig(**kwargs)
        
        self.__dict__.update(configuration.model_dump())

        if self.ids_file is None:
            self.ids = list(map(lambda file_name: os.path.splitext(file_name)[0], better_listdir(self.videos_dir)))
        else:
            with open(self.ids_file, "r") as file:
                self.ids = file.read().splitlines()
            
        self.videos, self.annotations = self.__prepare_videos_and_annotations()
        
        self.__segment_size_check()
        
    def __prepare_videos_and_annotations(self):
        return \
            list(map(lambda id: self.video_processor(self.videos_dir, id, **self.video_processor_kwargs), self.ids)), \
            list(map(lambda id: self.annotations_processor(self.annotations_dir, id, **self.annotations_processor_kwargs), self.ids))
        
    def __segment_size_check(self):
        if self.padder is None:
            for index, video in enumerate(self.videos):
                remaining_segments = len(video) % self.segment_size
                if remaining_segments != 0:
                    if self.verbose:
                        print(f"[warning]: {remaining_segments} frames will be lost, because video {index} has {len(video)} frames, which is not divisible by segment size {self.segment_size}. consider using a padder.")

    def __len__(self):
        return sum([(max(0, len(video) - self.overlap) // (self.segment_size - self.overlap)) for video in self.videos])

    def __getitem__(self, virtual_video_index):
        video_index, starting_frame_number_in_video = self.__translate_virtual_video_index_to_video_index(virtual_video_index)
        
        starting_frame = starting_frame_number_in_video
        ending_frame = starting_frame_number_in_video + self.segment_size
        
        frames= self.videos[video_index][starting_frame:ending_frame:self.step]
        annotations = self.annotations[video_index][starting_frame:ending_frame:self.step]
        
        # NOTE: we expect the video_processor to return a numpy array of the frames in the DEFAULT_VIDEO_SHAPE format.
        frames = frames.transpose(self.video_shape)
        
        if self.padder is not None:
            frames, annotations = self.padder(frames, annotations,  self.segment_size // self.step)
        
        if self.frames_transform is not None:
            frames = self.frames_transform(frames)
        
        if self.annotations_transform is not None:
            annotations = self.annotations_transform(annotations)
            
        return frames, annotations
    
    def __translate_virtual_video_index_to_video_index(self, virtual_video_index):
        video_index = 0
        
        videos_cropped_frames_number = [len(video) // self.segment_size for video in self.videos]
        
        cumulative_videos_cropped_frames_number = list(itertools.accumulate(videos_cropped_frames_number))
        
        video_index = bisect.bisect_right(cumulative_videos_cropped_frames_number, virtual_video_index)
        
        if video_index >= len(videos_cropped_frames_number):
            raise ValueError("Virtual video index is out of range.")

        previous_frames = 0 if video_index == 0 else cumulative_videos_cropped_frames_number[video_index - 1]
        segment_index_within_video = virtual_video_index - previous_frames
        starting_frame_number_in_video = segment_index_within_video * (self.segment_size - self.overlap)
        
        return video_index, starting_frame_number_in_video
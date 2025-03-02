import os
import bisect
import itertools

import numpy as np

from enum import IntEnum
from typing import Type, Any, Tuple

from video_dataset.video import Video
from video_dataset.utils import better_listdir
from video_dataset.annotations import Annotations

class VideoShapeComponents(IntEnum):
    TIME = 0
    HEIGHT = 1
    WIDTH = 2
    CHANNELS = 3
    
DEFAULT_VIDEO_SHAPE = (VideoShapeComponents.TIME, VideoShapeComponents.HEIGHT, VideoShapeComponents.WIDTH, VideoShapeComponents.CHANNELS)
    
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

    Methods:
        __len__(): Returns the total number of video segments in the dataset.
        __getitem__(virtual_video_index): Fetches a specific segment (video frames and annotations) by index.
        __expand_ids(): Expands the provided video and annotation IDs to full file paths.
        __segment_size_check(): Checks if the video segment size divides evenly into the total frames of each video.
        __translate_virtual_video_index_to_video_index(virtual_video_index): Translates a virtual video index into a specific video index and frame number.

    Note:
        - The videos in the `videos_dir` and annotations in the `annotations_dir` must have the same base names (different extensions allowed).
        - The class supports segmentation of videos into smaller frame chunks to optimize training.
        - Custom formats can be supported by implementing corresponding processors for videos and annotations.
    """
    # TODO: refactor the class into using kwargs rather than params just like this
    def __init__(
        self,
        # --- --- ---
        annotations_dir: str,
        videos_dir: str,
        # --- --- ---
        video_processor: Type[Video],
        annotations_processor: Type[Annotations],
        # --- --- ---
        segment_size: int,
        # --- --- ---
        video_processor_kwargs: dict[str, Any] = None,
        annotations_processor_kwargs: dict[str, Any] = None,
        # --- --- ---
        ids_file: list[str] = None,
        # --- --- ---
        frames_transform = None,
        annotations_transform = None,
        # --- --- ---
        verbose: bool = True,
        # --- --- ---
        video_extension: str = 'mp4',
        annotations_extension: str = 'csv',
        # --- --- ---
        video_shape: Tuple[int, int, int, int] = DEFAULT_VIDEO_SHAPE,
        # --- --- ---
        step: int = None
    ):
        """
        Note: the videos in the videos_dir and the annotations in the annotations_dir must have the same name, they can have different extensions of course.
        """
        self.annotations_dir = annotations_dir
        self.videos_dir = videos_dir
        self.segment_size = segment_size
        self.frames_transform = frames_transform
        self.annotations_transform = annotations_transform
        
        self.video_processor = video_processor
        self.annotations_processor = annotations_processor
        
        self.verbose = verbose
        
        self.video_extension = video_extension 
        self.annotations_extension = annotations_extension
        
        self.video_processor_kwargs = video_processor_kwargs or {}
        self.annotations_processor_kwargs = annotations_processor_kwargs or {}
        
        self.ids_file = ids_file
        
        self.video_shape = video_shape
        
        self.step = step
        
        self.__params_check()
        
        if self.ids_file is None:
            self.ids = list(map(lambda file_name: os.path.splitext(file_name)[0], better_listdir(self.videos_dir)))
        else:
            with open(self.ids_file, "r") as file:
                self.ids = file.read().splitlines()
            
        # self.__ids_check()
        
        self.videos, self.annotations = self.__prepare_videos_and_annotations()
        
        self.__segment_size_check()
        
    # TODO: it should check the params rather than the self.*
    def __params_check(self):
        if not os.path.exists(self.annotations_dir):
            raise ValueError(f"Annotations directory {self.annotations_dir} does not exist.")
        
        if not os.path.exists(self.videos_dir):
            raise ValueError(f"Videos directory {self.videos_dir} does not exist.")
        
        if not isinstance(self.video_processor, type):
            raise ValueError("Video processor must be a class.")
        
        if not isinstance(self.annotations_processor, type):
            raise ValueError("Annotations processor must be a class.")
        
        if not issubclass(self.video_processor, Video):
            raise ValueError("Video processor must inherit from the Video class.")
        
        if not issubclass(self.annotations_processor, Annotations):
            raise ValueError("Annotations processor must inherit from the Annotations class.")
        
        if not isinstance(self.segment_size, int):
            raise ValueError("Segment size must be an integer.")
        
        if self.segment_size <= 0:
            raise ValueError("Segment size must be a positive integer.")
        
        if self.ids_file is not None and not os.path.exists(self.ids_file):
            raise ValueError("ids_file does not exist.")
        
        if self.frames_transform is not None and not callable(self.frames_transform):
            raise ValueError("Frames transform must be a callable function.")
        
        if self.annotations_transform is not None and not callable(self.annotations_transform):
            raise ValueError("Annotations transform must be a callable function.")
        
        if not isinstance(self.verbose, bool):
            raise ValueError("Verbose must be a boolean.")
        
        if self.video_shape is None:
            raise ValueError("Video shape must not be None.")
            
        if not isinstance(self.video_shape, tuple):
            raise ValueError("Video shape must be a tuple.")
        
        if len(self.video_shape) != 4:
            raise ValueError("Video shape must have exactly 4 components.")
        
        if not all([isinstance(component, VideoShapeComponents) for component in self.video_shape]):
            raise ValueError("Video shape components must be of type VideoShapeComponents.")
        
        if len(set(self.video_shape)) != len(self.video_shape):
            raise ValueError("Video shape components must be unique.")
        
    def __prepare_videos_and_annotations(self):
        return \
            list(map(lambda id: self.video_processor(self.videos_dir, id, **self.video_processor_kwargs), self.ids)), \
            list(map(lambda id: self.annotations_processor(self.annotations_dir, id, **self.annotations_processor_kwargs), self.ids))
        
    def __segment_size_check(self):
        for index, video in enumerate(self.videos):
            remaining_segments = len(video) % self.segment_size
            if remaining_segments != 0:
                if self.verbose:
                    print(f"[warning]: {remaining_segments} frames will be lost, because video {index} has {len(video)} frames, which is not divisible by segment size {self.segment_size}.")

    def __len__(self):
        return sum([len(video) // self.segment_size for video in self.videos])    

    def __getitem__(self, virtual_video_index):
        video_index, starting_frame_number_in_video = self.__translate_virtual_video_index_to_video_index(virtual_video_index)
        
        starting_frame = starting_frame_number_in_video
        ending_frame = starting_frame_number_in_video + self.segment_size
        
        frames= self.videos[video_index][starting_frame:ending_frame:self.step]
        annotations = self.annotations[video_index][starting_frame:ending_frame:self.step]
        
        # NOTE: we expect the video_processor to return a numpy array of the frames in the DEFAULT_VIDEO_SHAPE format.
        frames = frames.transpose(self.video_shape)
        
        if self.frames_transform is not None:
            frames = self.frames_transform(frames)
        
        if self.annotations_transform is not None:
            annotations = self.annotations_transform(annotations)
            
        return frames, annotations
    
    def __translate_virtual_video_index_to_video_index(self, virtual_video_index):
        video_index = 0
        
        # videos_cropped_frames_number = [len(video_frames_annotations) // self.segment_size for video_frames_annotations in self.annotations]
        videos_cropped_frames_number = [len(video) // self.segment_size for video in self.videos]
        
        # cumulative_videos_cropped_frames_number = np.cumsum(videos_cropped_frames_number)
        cumulative_videos_cropped_frames_number = list(itertools.accumulate(videos_cropped_frames_number))
        
        # video_index = np.searchsorted(cumulative_videos_cropped_frames_number, virtual_video_index, side='right')
        video_index = bisect.bisect_right(cumulative_videos_cropped_frames_number, virtual_video_index)
        
        if video_index >= len(videos_cropped_frames_number):
            raise ValueError("Virtual video index is out of range.")

        previous_frames = 0 if video_index == 0 else cumulative_videos_cropped_frames_number[video_index - 1]
        segment_index_within_video = virtual_video_index - previous_frames
        starting_frame_number_in_video = segment_index_within_video * self.segment_size
        
        return video_index, starting_frame_number_in_video
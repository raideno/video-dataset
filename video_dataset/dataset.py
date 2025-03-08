import os
import bisect
import itertools

from enum import IntEnum
from typing import Type, Any, Tuple, Dict, List, Optional, Callable
from pydantic import BaseModel, Field, FilePath, DirectoryPath, PositiveInt, NonNegativeInt, field_validator, model_validator

from video_dataset.padder import Padder
from video_dataset.utils import better_listdir
from video_dataset.video import Video, UndefinedVideoException
from video_dataset.annotations import Annotations, UndefinedAnnotationsException

class VideoShapeComponents(IntEnum):
    TIME = 0
    HEIGHT = 1
    WIDTH = 2
    CHANNELS = 3
    
DEFAULT_VIDEO_SHAPE = (VideoShapeComponents.TIME, VideoShapeComponents.HEIGHT, VideoShapeComponents.WIDTH, VideoShapeComponents.CHANNELS)

class VideoDatasetConfiguration(BaseModel):
    annotations_dir: DirectoryPath
    videos_dir: DirectoryPath
    video_processor: Type[Video]
    annotations_processor: Type[Annotations]
    segment_size: int
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
    
    allow_undefined_annotations: bool = False
    
    padder: Optional[Any] = None
    
    overlap: Optional[NonNegativeInt] = 0
    
    load_videos: Optional[bool] = True
    load_annotations: Optional[bool] = True

    @model_validator(mode='before')
    @classmethod
    def adjust_loading_videos_on_segment_size(cls, values):
        segment_size = values.get('segment_size')
        
        load_videos_set = 'load_videos' in values
        
        # NOTE: if segment_size is -1 and loading flags are not explicitly set, set them to False just to prevent unnecessary unwanted loading times.
        if segment_size == VideoDataset.FULL_VIDEO_SEGMENT and not load_videos_set:
            values['load_videos'] = False
        
        return values

    @field_validator("segment_size")
    def check_segment_size(cls, v):
        if v < VideoDataset.FULL_VIDEO_SEGMENT:
            raise ValueError("segment_size must be bigger or equal to -1.")
        return v

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
    FULL_VIDEO_SEGMENT = -1
    
    def __init__(self, **kwargs):
        configuration = VideoDatasetConfiguration(**kwargs)
        
        self.__dict__.update(configuration.model_dump())

        if self.ids_file is None:
            self.ids = list(map(lambda file_name: os.path.splitext(file_name)[0], better_listdir(self.videos_dir)))
        else:
            with open(self.ids_file, "r") as file:
                self.ids = file.read().splitlines()
                
        self.videos, self.annotations = self.__prepare_videos_and_annotations()
        
        self.__segment_size_check()
        
    def __prepare_videos_and_annotations(self):
        videos = []
        annotations = []
        
        for id in self.ids:
            video = self.video_processor(self.videos_dir, id, **self.video_processor_kwargs)
                
            try:
                annotation = self.annotations_processor(self.annotations_dir, id, **self.annotations_processor_kwargs)
            except UndefinedAnnotationsException as exception:
                if self.allow_undefined_annotations:
                    if self.verbose:
                        print(f"[warning]: {exception.message}")
                    annotation = None
                else:
                    raise exception
            
            videos.append(video)
            annotations.append(annotation)
            
        return videos, annotations
        
    def __segment_size_check(self):
        if self.padder is None and self.segment_size != VideoDataset.FULL_VIDEO_SEGMENT:
            for index, video in enumerate(self.videos):
                remaining_segments = len(video) % self.segment_size
                if remaining_segments != 0:
                    if self.verbose:
                        print(f"[warning]: {remaining_segments} frames will be lost, because video {index} has {len(video)} frames, which is not divisible by segment size {self.segment_size}. consider using a padder.")

    def __len__(self):
        if self.segment_size == VideoDataset.FULL_VIDEO_SEGMENT:
            return len(self.videos)
        else:
            return sum([(max(0, len(video) - self.overlap) // (self.segment_size - self.overlap)) for video in self.videos])
        
    def __getitem__(self, virtual_video_index):
        if self.segment_size == VideoDataset.FULL_VIDEO_SEGMENT:
            video_index = virtual_video_index
            
            frames = self.__getitem_frames__(video_index, 0) if self.load_videos else None
            annotations = self.__getitem_annotations__(video_index, 0) if self.load_annotations else None
            
            return frames, annotations, (video_index, self.videos[video_index].get_id(), 0)
        else:
            video_index, starting_frame_number_in_video = self.__translate_virtual_video_index_to_video_index(virtual_video_index)
            
            frames = self.__getitem_frames__(video_index, starting_frame_number_in_video) if self.load_videos else None
            annotations = self.__getitem_annotations__(video_index, starting_frame_number_in_video) if self.load_annotations else None
        
            return frames, annotations, (video_index, self.videos[video_index].get_id(), starting_frame_number_in_video)
    
    def __getitem_frames__(self, video_index, starting_frame_number_in_video):
        if self.segment_size == VideoDataset.FULL_VIDEO_SEGMENT:
            frames = self.videos[video_index][0:]
            
            frames = frames.transpose(self.video_shape)
            
            if self.frames_transform is not None:
                frames = self.frames_transform(frames)
                
            return frames
        else:
            starting_frame = starting_frame_number_in_video
            ending_frame = starting_frame_number_in_video + self.segment_size
            
            frames = self.videos[video_index][starting_frame:ending_frame:self.step]
            
            # NOTE: we expect the video_processor to return a numpy array of the frames in the DEFAULT_VIDEO_SHAPE format.
            frames = frames.transpose(self.video_shape)
            
            if self.padder is not None:
                frames, _ = self.padder(frames=frames, annotations=None,  target_segment_size=self.segment_size // self.step)
            
            if self.frames_transform is not None:
                frames = self.frames_transform(frames)
            
            return frames
    
    def __getitem_annotations__(self, video_index, starting_frame_number_in_video):
        if self.segment_size == VideoDataset.FULL_VIDEO_SEGMENT:
            video_annotations = self.annotations[video_index]
            
            if video_annotations is None and self.allow_undefined_annotations:
                return None
            
            else:
                annotations = video_annotations[0:]
                
                if self.annotations_transform is not None:
                    annotations = self.annotations_transform(annotations)
                    
                return annotations
        else:
            starting_frame = starting_frame_number_in_video
            ending_frame = starting_frame_number_in_video + self.segment_size
            
            video_annotations = self.annotations[video_index]
            
            if video_annotations is None and self.allow_undefined_annotations:
                return None
            else:
                annotations = video_annotations[starting_frame:ending_frame:self.step]
                
                if self.padder is not None:
                    _, annotations = self.padder(frames=None, annotations=annotations,  target_segment_size=self.segment_size // self.step)
                
                if self.annotations_transform is not None:
                    annotations = self.annotations_transform(annotations)
                    
                return annotations
    
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
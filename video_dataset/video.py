import os
import cv2

import numpy as np

from PIL import Image
from video_dataset.utils import better_listdir
from abc import ABC, ABCMeta, abstractmethod

class Video(ABC):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, videos_dir_path: str, id: str):
        pass
    
    @abstractmethod
    def get_id(self):
        pass
    
    @abstractmethod
    def __len__(self):
        pass
    
    @abstractmethod
    def __getitem__(self, index: int | slice):
        """
        Get the frame(s) in the video file.
        Note that even if an index is given the frames will be returned in a batch format (Number of frames, Height, Width, Channels).
        """
        pass
    
class UndefinedVideoException(Exception):
    """
    Raised when no valid video file is found for a given video ID.
    """
    
    def __init__(self, message: str, id: str):
        super().__init__(message)
        
        self.message = message
        self.id = id

STARTING_INDEX = 1

class VideoFromVideoFramesDirectory(Video):
    def __init__(self, videos_dir_path, id, starting_index = STARTING_INDEX):
        super().__init__(videos_dir_path, id)
        
        self.id = id
        self.videos_dir_path = videos_dir_path
        self.starting_index = starting_index
        
        if not self.__does_video_video_exists():
            raise UndefinedVideoException(f"No valid video file found for {id} in {videos_dir_path}", self.id)
        
    def __does_video_video_exists(self):
        return os.path.exists(os.path.join(self.videos_dir_path, self.id))
    
    def get_id(self):
        return self.id
    
    def __len__(self):
        return len(better_listdir(os.path.join(self.videos_dir_path, self.id)))
    
    def __getitem__(self, index: int | slice):
        if isinstance(index, int):
            if index < 0 or index >= self.__len__():
                raise IndexError("Index out of bounds")
            return self.__get_frame(index)
        elif isinstance(index, slice):
            start, stop, step = index.indices(self.__len__())
            return self.__get_frames(start, stop, step)
        else:
            raise TypeError("Index must be an integer or slice")
        
    def __get_frame(self, index: int):
        image_path = os.path.join(self.videos_dir_path, self.id, f"img_{(index + self.starting_index):05d}.jpg")
        
        # NOTE: PIL return the image in shape (width, height, channels), converting it to a numpy array will give it in shape (height, width, channels)
        return np.array(Image.open(image_path).convert("RGB"))
    
    def __get_frames(self, start: int, stop: int, step: int):
        frames = []
        
        for i in range(start, stop, step):
            frame = self.__get_frame(i)
            frames.append(frame)
        
        # NOTE: will be of shape (number of frames, height, width, channels)
        return np.array(frames)
    
class VideoFromVideoFile(Video):
    SUPPORTED_VIDEO_EXTENSIONS = ["mp4", "avi", "mkv", "mov", "webm"]
    
    def __init__(self, videos_dir_path, id, video_extension=None):
        super().__init__(videos_dir_path, id)
        
        self.id = id
        self.videos_dir_path = videos_dir_path
        self.video_extension = video_extension or VideoFromVideoFile.__is_video_file(self.videos_dir_path, self.id)

        if not self.video_extension:
            raise UndefinedVideoException(f"No valid video file found for {id} in {videos_dir_path}", self.id)

        self.video_path = os.path.join(self.videos_dir_path, f"{self.id}.{self.video_extension}")
        self.cached_number_of_frames = self.__cache_number_of_frames()
        
    @staticmethod
    def __is_video_file(videos_dir_path, id):
        for extension in VideoFromVideoFile.SUPPORTED_VIDEO_EXTENSIONS:
            if os.path.exists(os.path.join(videos_dir_path, f"{id}.{extension}")):
                return extension
        return None
        
    def get_id(self):
        return self.id
    
    def __cache_number_of_frames(self):
        """Opens video temporarily to get frame count."""
        video = cv2.VideoCapture(os.path.join(self.videos_dir_path, f"{self.id}.{self.video_extension}"))
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.release()
        return num_frames

    def __len__(self):
        return self.cached_number_of_frames

    def __getitem__(self, index: int | slice):
        if isinstance(index, int):
            if index < 0 or index >= self.__len__():
                raise IndexError("Index out of bounds")
            return self.__get_frame(index)
        elif isinstance(index, slice):
            start, stop, step = index.indices(self.__len__())
            return self.__get_frames(start, stop, step)
        else:
            raise TypeError("Index must be an integer or slice")

    def __get_frame(self, index: int):
        """Opens video, retrieves a single frame, and immediately closes it."""
        with cv2.VideoCapture(self.videos_dir_path) as video:
            video.set(cv2.CAP_PROP_POS_FRAMES, index)
            # NOTE: return in the shape (height, width, channels)
            ret, frame = video.read()
        
        if not ret:
            raise Exception(f"Could not read frame at index {index}")
        
        return np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def __get_frames(self, start: int, stop: int, step: int):
        """Reads multiple frames in a single open-close cycle."""
        frames = []
        
        with cv2.VideoCapture(self.videos_dir_path) as video:
            video.set(cv2.CAP_PROP_POS_FRAMES, start)
            
            for i in range(start, stop, step):
                # NOTE: return in the shape (height, width, channels)
                ret, frame = video.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # NOTE: will be of shape (number of frames, height, width, channels)
        return np.array(frames)
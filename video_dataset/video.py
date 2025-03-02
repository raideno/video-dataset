import os
import cv2

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

class VideoFromVideoFramesDirectory(Video):
    def __init__(self, videos_dir_path, id):
        super().__init__(videos_dir_path, id)
        
        self.id = id
        self.videos_dir_path = videos_dir_path
    
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
        image_path = os.path.join(self.videos_dir_path, self.id, f"img_{index:05d}.jpg")
        
        return Image.open(image_path).convert("RGB")
    
    def __get_frames(self, start: int, stop: int, step: int):
        frames = []
        
        for i in range(start, stop, step):
            frame = self.__get_frame(i)
            frames.append(frame)
        
        return frames
    
VIDEO_EXTENSION = "mp4"
    
class VideoFromVideoFile(Video):
    # TODO: it should be able to support multiple video extensions by providing a function for the video_extension or a sort of global video reader by providing * & thus be able to read anything.
    # NOTE: when setting it to * be carful if two videos with the same name but different format are available
    def __init__(self, videos_dir_path, id, video_extension = VIDEO_EXTENSION):
        super().__init__(videos_dir_path, id)
        
        self.id = id
        self.videos_dir_path = videos_dir_path
        self.video_extension = video_extension
        self.cached_number_of_frames = self.__cache_number_of_frames()
        
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
            ret, frame = video.read()
        
        if not ret:
            raise Exception(f"Could not read frame at index {index}")
        
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __get_frames(self, start: int, stop: int, step: int):
        """Reads multiple frames in a single open-close cycle."""
        frames = []
        
        with cv2.VideoCapture(self.videos_dir_path) as video:
            video.set(cv2.CAP_PROP_POS_FRAMES, start)
            
            for i in range(start, stop, step):
                ret, frame = video.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        return frames
import os
import pytest
import tempfile

import numpy as np

from PIL import Image
from typing import Tuple, Type
from dataclasses import dataclass

from enum import IntEnum

from video_dataset.dataset import VideoDataset
from video_dataset.video import VideoFromVideoFramesDirectory
from video_dataset.annotations import AnnotationsFromFrameLevelTxtFileAnnotations

DEFAUlT_VIDEO_FPS = 8
DEFAULT_VIDEO_WIDTH = 8
DEFAULT_VIDEO_HEIGHT = 8
DEFAULT_VIDEO_NUMBER_OF_CHANNELS = 3

@dataclass
class DatasetConfiguration:
    videos_directory_path: str
    annotations_directory_path: str
    ids_file_path: str | None

def create_frame_level_video(videos_directory_path, id, number_of_frames, width=DEFAULT_VIDEO_WIDTH, height=DEFAULT_VIDEO_HEIGHT):
    video_directory_path = os.path.join(videos_directory_path, id)

    os.makedirs(video_directory_path, exist_ok=True)
    
    for i in range(number_of_frames):
        frame = np.random.randint(0, 255, (height, width, DEFAULT_VIDEO_NUMBER_OF_CHANNELS), dtype=np.uint8)
        
        image = Image.fromarray(frame)
        
        image_file_name = f"img_{(i + 1):05d}.jpg"
        
        image_path = os.path.join(video_directory_path, image_file_name)
        
        image.save(image_path)
        
    return video_directory_path

def create_segment_level_annotations_csv_file(annotations_directory_path, id, number_of_segments):
    separator = ";"
    annotations_file_path = os.path.join(annotations_directory_path, f"{id}.csv")
    
    with open(annotations_file_path, "w") as file:
        file.write(f"starting-timestamp{separator}ending-timestamp{separator}action\n")
        
        for i in range(number_of_segments):
            start_frame = i * 10 + 1
            end_frame = (i + 1) * 10
            label = np.random.randint(0, 2)
            file.write(f"{start_frame}{separator}{end_frame}{separator}{label}\n")
    
    return annotations_file_path

def create_complete_video(videos_directory_path, id, number_of_frames, width=DEFAULT_VIDEO_WIDTH, height=DEFAULT_VIDEO_HEIGHT):
    video_directory_path = os.path.join(videos_directory_path, id)
    os.makedirs(video_directory_path, exist_ok=True)
    
    video_path = os.path.join(video_directory_path, f"{id}.mp4")
    
    # NOTE: create video frames
    frames = [np.random.randint(0, 255, (height, width, DEFAULT_VIDEO_NUMBER_OF_CHANNELS), dtype=np.uint8) 
              for _ in range(number_of_frames)]
    
    # NOTE: convert frames to video using OpenCV
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, DEFAUlT_VIDEO_FPS, (width, height))
    
    for frame in frames:
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    video_writer.release()
    
    return video_path

def create_frame_level_annotations_txt_file(annotations_directory_path, id, number_of_frames):
    annotations_file_path = os.path.join(annotations_directory_path, f"{id}.txt")
    
    with open(annotations_file_path, "w") as file:
        for i in range(number_of_frames):
            file.write(f"{(i + 1):05d} {np.random.randint(0, 2)}\n")
    
    return annotations_file_path

class AnnotationType(IntEnum):
    TXT_FRAME_LEVEL = 0
    CSV_SEGMENT_LEVEL = 1
    
class VideoType(IntEnum):
    FRAME_LEVEL = 0
    COMPLETE_VIDEO = 1 

def setup_test_data(
    number_of_samples,
    number_of_frames,
    number_of_frames_variance,
    with_ids_file=False,
    annotations_type: AnnotationType = AnnotationType.TXT_FRAME_LEVEL,
    video_type: VideoType = VideoType.FRAME_LEVEL
) -> Tuple[DatasetConfiguration, tempfile.TemporaryDirectory]:
    temporary_directory = tempfile.TemporaryDirectory()
    
    videos_directory_path = os.path.join(temporary_directory.name, "videos")
    annotations_directory_path = os.path.join(temporary_directory.name, "annotations")
    
    os.makedirs(videos_directory_path, exist_ok=True)
    os.makedirs(annotations_directory_path, exist_ok=True)
    
    ids = []

    for i in range(number_of_samples):
        id = f"video_{i}"
        
        number_of_frames_ = int(number_of_frames + np.random.randint(
            -number_of_frames_variance * number_of_frames, 
            number_of_frames_variance * number_of_frames
        ))
        
        if video_type == VideoType.FRAME_LEVEL:
            create_frame_level_video(videos_directory_path, id, number_of_frames_)
        elif video_type == VideoType.COMPLETE_VIDEO:
            create_complete_video(videos_directory_path, id, number_of_frames_)
        
        if annotations_type == AnnotationType.CSV_SEGMENT_LEVEL:
            create_segment_level_annotations_csv_file(annotations_directory_path, id, number_of_frames_ // 10)
        elif annotations_type == AnnotationType.TXT_FRAME_LEVEL:
            create_frame_level_annotations_txt_file(annotations_directory_path, id, number_of_frames_)
        
        ids.append(id)
    
    ids_file_path = os.path.join(temporary_directory.name, "ids.txt") if with_ids_file else None
    
    if with_ids_file:
        with open(ids_file_path, "w") as file:
            for id in ids:
                file.write(f"{id}\n")
    
    # NOTE: return the temporary directory to ensure cleanup
    return DatasetConfiguration(
        videos_directory_path=videos_directory_path,
        annotations_directory_path=annotations_directory_path,
        ids_file_path=ids_file_path,
    ), temporary_directory

@pytest.fixture
def setup_small_test_data():
    dataset_configuration, temporary_directory = setup_test_data(
        number_of_samples=8,
        number_of_frames=32,
        number_of_frames_variance=0.1
    )
    
    yield dataset_configuration, temporary_directory

    temporary_directory.cleanup()
    
def initialize_dataset_from_configuration(dataset_configuration: DatasetConfiguration, segment_size: int):
    return VideoDataset(
        annotations_dir=dataset_configuration.annotations_directory_path,
        videos_dir=dataset_configuration.videos_directory_path,
        video_processor=VideoFromVideoFramesDirectory,
        annotations_processor=AnnotationsFromFrameLevelTxtFileAnnotations,
        ids_file=dataset_configuration.ids_file_path,
        segment_size=segment_size
    )
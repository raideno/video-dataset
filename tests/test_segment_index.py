import pytest
import numpy as np
from tests.helpers import setup_test_data, initialize_dataset_from_configuration, VideoType, AnnotationType
from tests.helpers import DEFAULT_VIDEO_HEIGHT, DEFAULT_VIDEO_WIDTH, DEFAULT_VIDEO_NUMBER_OF_CHANNELS
from video_dataset.dataset import VideoDataset
from video_dataset.video import VideoFromVideoFramesDirectory
from video_dataset.annotations import AnnotationsFromSegmentLevelCsvFileAnnotations, AnnotationsFromFrameLevelTxtFileAnnotations

@pytest.fixture
def setup_simple_segment_data():
    """Setup test data for segment index tests"""
    dataset_configuration, temporary_directory = setup_test_data(
        number_of_samples=1,  # Single video to simplify test
        number_of_frames=30,  # 30 frames to create 3 segments of size 10
        number_of_frames_variance=0.1,  # Very small variance to avoid error,
        annotations_type=AnnotationType.TXT_FRAME_LEVEL,
        video_type=VideoType.FRAME_LEVEL
    )
    
    yield dataset_configuration, temporary_directory
    
    temporary_directory.cleanup()


@pytest.fixture
def setup_overlap_segment_data():
    """Setup test data for testing overlap segments"""
    dataset_configuration, temporary_directory = setup_test_data(
        number_of_samples=1,
        number_of_frames=40,
        number_of_frames_variance=0.1, # Very small variance to avoid error
        annotations_type=AnnotationType.TXT_FRAME_LEVEL,
        video_type=VideoType.FRAME_LEVEL
    )
    
    yield dataset_configuration, temporary_directory
    
    temporary_directory.cleanup()


@pytest.fixture
def setup_full_video_segment_data():
    """Setup test data for full video segment tests"""
    dataset_configuration, temporary_directory = setup_test_data(
        number_of_samples=2,
        number_of_frames=30,
        number_of_frames_variance=0.1,  # Very small variance to avoid error
        annotations_type=AnnotationType.TXT_FRAME_LEVEL,
        video_type=VideoType.FRAME_LEVEL
    )
    
    yield dataset_configuration, temporary_directory
    
    temporary_directory.cleanup()


def test_segment_index_in_return_transform(setup_simple_segment_data):
    """Test that segment_index is correctly calculated and passed to return_transform"""
    dataset_configuration, temporary_directory = setup_simple_segment_data
    
    segment_size = 10
    collected_segment_indices = []
    
    # Define a custom return_transform function to capture segment indices
    def custom_return_transform(data_dict):
        collected_segment_indices.append(data_dict["segment_index"])
        return data_dict["frames"], data_dict["annotations"]
    
    # Initialize dataset with the custom transform
    dataset = VideoDataset(
        annotations_dir=dataset_configuration.annotations_directory_path,
        videos_dir=dataset_configuration.videos_directory_path,
        video_processor=VideoFromVideoFramesDirectory,
        annotations_processor=AnnotationsFromFrameLevelTxtFileAnnotations,
        ids_file=dataset_configuration.ids_file_path,
        segment_size=segment_size,
        return_transform=custom_return_transform
    )
    
    # Retrieve all segments from the dataset
    for i in range(len(dataset)):
        _ = dataset[i]
    
    # Check if segment indices are correct (should be 0, 1, 2 for a 30-frame video with segment_size=10)
    assert len(collected_segment_indices) == 3
    assert collected_segment_indices == [0, 1, 2]


def test_segment_index_with_overlap(setup_overlap_segment_data):
    """Test that segment_index is correctly calculated when overlap is used"""
    dataset_configuration, temporary_directory = setup_overlap_segment_data
    
    segment_size = 10
    overlap = 5
    collected_segment_indices = []
    collected_start_frames = []
    
    # Define a custom return_transform function to capture data
    def custom_return_transform(data_dict):
        collected_segment_indices.append(data_dict["segment_index"])
        collected_start_frames.append(data_dict["starting_frame_number_in_video"])
        return data_dict["frames"], data_dict["annotations"]
    
    # Initialize dataset with overlap
    dataset = VideoDataset(
        annotations_dir=dataset_configuration.annotations_directory_path,
        videos_dir=dataset_configuration.videos_directory_path,
        video_processor=VideoFromVideoFramesDirectory,
        annotations_processor=AnnotationsFromFrameLevelTxtFileAnnotations,
        ids_file=dataset_configuration.ids_file_path,
        segment_size=segment_size,
        overlap=overlap,
        return_transform=custom_return_transform
    )
    
    # Retrieve all segments from the dataset
    for i in range(len(dataset)):
        _ = dataset[i]
    
    # With segment_size=10 and overlap=5, a 40-frame video should have 7 segments
    # starting at frames 0, 5, 10, 15, 20, 25, 30
    # Segment indices should be 0, 1, 2, 3, 4, 5, 6
    expected_segment_indices = [0, 1, 2, 3, 4, 5, 6]
    expected_start_frames = [0, 5, 10, 15, 20, 25, 30]
    
    assert len(collected_segment_indices) == len(expected_segment_indices)
    assert collected_segment_indices == expected_segment_indices
    assert collected_start_frames == expected_start_frames


def test_full_video_segment_index(setup_full_video_segment_data):
    """Test that segment_index is correctly set to 0 when using FULL_VIDEO_SEGMENT mode"""
    dataset_configuration, temporary_directory = setup_full_video_segment_data
    
    segment_index_value = None
    
    # Define a custom return_transform function to capture segment index
    def custom_return_transform(data_dict):
        nonlocal segment_index_value
        segment_index_value = data_dict["segment_index"]
        return data_dict["frames"], data_dict["annotations"]
    
    # Initialize dataset with FULL_VIDEO_SEGMENT
    dataset = VideoDataset(
        annotations_dir=dataset_configuration.annotations_directory_path,
        videos_dir=dataset_configuration.videos_directory_path,
        video_processor=VideoFromVideoFramesDirectory,
        annotations_processor=AnnotationsFromFrameLevelTxtFileAnnotations,
        ids_file=dataset_configuration.ids_file_path,
        segment_size=VideoDataset.FULL_VIDEO_SEGMENT,
        return_transform=custom_return_transform
    )
    
    # Retrieve the first video
    _ = dataset[0]
    
    # In FULL_VIDEO_SEGMENT mode, segment_index should be 0
    assert segment_index_value == 0
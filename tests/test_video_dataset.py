import pytest

from tests.helpers import setup_small_test_data, initialize_dataset_from_configuration
from tests.helpers import DEFAULT_VIDEO_HEIGHT, DEFAULT_VIDEO_WIDTH, DEFAULT_VIDEO_NUMBER_OF_CHANNELS

from video_dataset.video import VideoFromVideoFramesDirectory
from video_dataset.dataset import VideoDataset, DEFAULT_VIDEO_SHAPE
from video_dataset.annotations import AnnotationsFromFrameLevelTxtFileAnnotations
 
    
def test_initialization(setup_small_test_data):
    dataset_configuration, _ = setup_small_test_data

    dataset = initialize_dataset_from_configuration(dataset_configuration=dataset_configuration, segment_size=12)


def test_video_sample_retrieval(setup_small_test_data):
    dataset_configuration, _ = setup_small_test_data
    
    segment_size = 12
    
    dataset = initialize_dataset_from_configuration(dataset_configuration=dataset_configuration, segment_size=segment_size)
    
    frames, labels = dataset[10]
    
    assert frames.shape == (segment_size, DEFAULT_VIDEO_HEIGHT, DEFAULT_VIDEO_WIDTH, DEFAULT_VIDEO_NUMBER_OF_CHANNELS)
    assert len(labels) == segment_size
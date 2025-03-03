import pytest

from video_dataset import VideoDataset
from video_dataset.video import VideoFromVideoFramesDirectory
from video_dataset.annotations import AnnotationsFromSegmentLevelCsvFileAnnotations, AnnotationsFromFrameLevelTxtFileAnnotations

from tests.helpers import setup_test_data, VideoType, AnnotationType, DEFAUlT_VIDEO_FPS, DEFAULT_VIDEO_HEIGHT, DEFAULT_VIDEO_WIDTH, DEFAULT_VIDEO_NUMBER_OF_CHANNELS

@pytest.fixture
def setup_csv_annotations_dataset():
    dataset_configuration, temporary_directory = setup_test_data(
        number_of_samples=8,
        number_of_frames=32,
        number_of_frames_variance=0.1,
        annotations_type=AnnotationType.CSV_SEGMENT_LEVEL,
        video_type=VideoType.FRAME_LEVEL
    )
    
    yield dataset_configuration, temporary_directory

    temporary_directory.cleanup()
    
@pytest.fixture
def setup_txt_annotations_dataset():
    dataset_configuration, temporary_directory = setup_test_data(
        number_of_samples=8,
        number_of_frames=32,
        number_of_frames_variance=0.1,
        annotations_type=AnnotationType.TXT_FRAME_LEVEL,
        video_type=VideoType.FRAME_LEVEL
    )
    
    yield dataset_configuration, temporary_directory

    temporary_directory.cleanup()
    
def test_csv_annotations(setup_csv_annotations_dataset):
    segment_size = 8
    
    dataset_configuration, temporary_directory = setup_csv_annotations_dataset
    
    dataset = VideoDataset(
        annotations_dir=dataset_configuration.annotations_directory_path,
        videos_dir=dataset_configuration.videos_directory_path,
        video_processor=VideoFromVideoFramesDirectory,
        annotations_processor=AnnotationsFromSegmentLevelCsvFileAnnotations,
        ids_file=dataset_configuration.ids_file_path,
        segment_size=segment_size,
        annotations_processor_kwargs={ 'fps': DEFAUlT_VIDEO_FPS}
    )
    
    for i in range(len(dataset)):
        video, annotations = dataset[i]
        
        print(video.shape)
        
        assert video.shape == (segment_size, DEFAULT_VIDEO_HEIGHT, DEFAULT_VIDEO_WIDTH, DEFAULT_VIDEO_NUMBER_OF_CHANNELS)
        assert len(annotations) == segment_size
        
    temporary_directory.cleanup()

def test_txt_annotations(setup_txt_annotations_dataset):
    segment_size = 8
    
    dataset_configuration, temporary_directory = setup_txt_annotations_dataset
    
    dataset = VideoDataset(
        annotations_dir=dataset_configuration.annotations_directory_path,
        videos_dir=dataset_configuration.videos_directory_path,
        video_processor=VideoFromVideoFramesDirectory,
        annotations_processor=AnnotationsFromFrameLevelTxtFileAnnotations,
        ids_file=dataset_configuration.ids_file_path,
        segment_size=segment_size,
    )
    
    for i in range(len(dataset)):
        video, annotations = dataset[i]
        
        assert video.shape == (segment_size, DEFAULT_VIDEO_HEIGHT, DEFAULT_VIDEO_WIDTH, DEFAULT_VIDEO_NUMBER_OF_CHANNELS)
        assert len(annotations) == segment_size
        
    temporary_directory.cleanup()
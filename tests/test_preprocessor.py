import os
import pytest
import shutil

from unittest.mock import patch, call
from video_dataset.preprocessor import extract_frames_from_videos  # Assuming the function is in a file called extract_frames.py

@pytest.fixture
def setup_test_dirs():
    """Create and clean up test directories"""
    # Setup test directories
    videos_dir = "test_videos"
    output_dir = "test_output"
    
    # Create test directories if they don't exist
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create some dummy video files
    video_files = ["video1.mp4", "video2.avi", "video3.mkv"]
    for video in video_files:
        with open(os.path.join(videos_dir, video), 'w') as f:
            f.write("dummy content")
    
    yield videos_dir, output_dir, video_files
    
    # Cleanup after test
    shutil.rmtree(videos_dir)
    shutil.rmtree(output_dir)

@pytest.fixture
def setup_test_dirs_with_spaces():
    """Create and clean up test directories including files with spaces"""
    videos_dir = "test_videos_spaces"
    output_dir = "test_output_spaces"
    
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    video_files = ["video with spaces.mp4", "another video.avi"]
    for video in video_files:
        with open(os.path.join(videos_dir, video), 'w') as f:
            f.write("dummy content")
    
    yield videos_dir, output_dir, video_files
    
    shutil.rmtree(videos_dir)
    shutil.rmtree(output_dir)

@patch('os.system')
@patch('os.makedirs')
def test_basic_extraction(mock_makedirs, mock_system, setup_test_dirs):
    """Test basic frame extraction functionality"""
    videos_dir, output_dir, _ = setup_test_dirs
    
    extract_frames_from_videos(videos_dir, output_dir)
    
    # Check if directories were created for each video
    assert mock_makedirs.call_count == 3
    
    # Check if ffmpeg was called for each video
    assert mock_system.call_count == 3
    
    # Check the correct ffmpeg commands were executed
    mock_system.assert_any_call(f"ffmpeg -i {os.path.join(videos_dir, 'video1.mp4')} {os.path.join(output_dir, 'video1')}/img_%05d.jpg")
    mock_system.assert_any_call(f"ffmpeg -i {os.path.join(videos_dir, 'video2.avi')} {os.path.join(output_dir, 'video2')}/img_%05d.jpg")
    mock_system.assert_any_call(f"ffmpeg -i {os.path.join(videos_dir, 'video3.mkv')} {os.path.join(output_dir, 'video3')}/img_%05d.jpg")

@patch('os.system')
def test_custom_output_extension(mock_system, setup_test_dirs):
    """Test extraction with custom output extension"""
    videos_dir, output_dir, _ = setup_test_dirs
    
    extract_frames_from_videos(videos_dir, output_dir, output_extension="png")
    
    # Check the correct ffmpeg commands were executed with png extension
    for call_args in mock_system.call_args_list:
        assert call_args[0][0].endswith(".png")

@patch('os.system')
@patch('os.path.exists')
def test_skip_existing_directories(mock_exists, mock_system, setup_test_dirs):
    """Test skipping videos with existing output directories"""
    videos_dir, output_dir, _ = setup_test_dirs
    
    # Simulate that output directories for all videos already exist
    mock_exists.return_value = True
    
    extract_frames_from_videos(videos_dir, output_dir)
    
    # Check that ffmpeg was not called
    mock_system.assert_not_called()

@patch('os.system')
@patch('builtins.print')
def test_verbose_output(mock_print, mock_system, setup_test_dirs):
    """Test verbose output messages"""
    videos_dir, output_dir, _ = setup_test_dirs
    
    extract_frames_from_videos(videos_dir, output_dir, verbose=True)
    
    # Check that print was called for each extraction
    assert mock_print.call_count == 6  # 2 prints per video (start and complete)

@patch('os.system')
@patch('builtins.print')
def test_non_verbose_output(mock_print, mock_system, setup_test_dirs):
    """Test non-verbose mode (no output messages)"""
    videos_dir, output_dir, _ = setup_test_dirs
    
    extract_frames_from_videos(videos_dir, output_dir, verbose=False)
    
    # Check that print was not called
    mock_print.assert_not_called()

def test_empty_videos_directory(setup_test_dirs):
    """Test behavior with empty videos directory"""
    videos_dir, output_dir, _ = setup_test_dirs
    
    # Remove all video files
    for file in os.listdir(videos_dir):
        os.remove(os.path.join(videos_dir, file))
    
    # Should run without errors even if directory is empty
    extract_frames_from_videos(videos_dir, output_dir)
    
    # No directories should be created in output_dir
    assert len(os.listdir(output_dir)) == 0

@patch('os.system')
def test_ffmpeg_command_construction(mock_system, setup_test_dirs_with_spaces):
    """Test that ffmpeg commands are constructed correctly with spaces in paths"""
    videos_dir, output_dir, video_files = setup_test_dirs_with_spaces
    
    extract_frames_from_videos(videos_dir, output_dir)
    
    # Check that the command for the video with spaces is correctly formed
    for video in video_files:
        video_name = os.path.splitext(os.path.basename(video))[0]
        video_path = os.path.join(videos_dir, video)
        output_path = os.path.join(output_dir, video_name)
        mock_system.assert_any_call(f"ffmpeg -i {video_path} {output_path}/img_%05d.jpg")

@patch('os.listdir')
def test_sorted_video_processing(mock_listdir, setup_test_dirs):
    """Test that videos are processed in sorted order"""
    videos_dir, output_dir, _ = setup_test_dirs
    
    # Return videos in unsorted order
    mock_listdir.return_value = ["video3.mkv", "video1.mp4", "video2.avi"]
    
    with patch('os.system') as mock_system:
        extract_frames_from_videos(videos_dir, output_dir)
        
        # Check that the videos were processed in sorted order
        calls = mock_system.call_args_list
        assert "video1.mp4" in calls[0][0][0]
        assert "video2.avi" in calls[1][0][0]
        assert "video3.mkv" in calls[2][0][0]
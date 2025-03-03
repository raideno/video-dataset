import os

def extract_frames_from_videos(videos_dir: str, output_dir: str, output_extension: str = "jpg", verbose: bool = True):
    """
    Extract frames from all video files in the specified directory and save them as image files.
    Note that you must have ffmpeg already installed on your computer for this function to work properly.
    
    Parameters:
        videos_dir (str): The directory containing the video files to process.
        output_dir (str): The directory where the extracted frames will be saved.
        output_extension (str): The image format for saving the frames (default is "jpg").
        verbose (bool): If True, prints progress messages during extraction (default is True).
        
    Returns:
        None
        
    This method iterates over all video files in the specified `videos_dir`, creates a subfolder for each video 
    in the `output_dir`, and extracts frames from each video using ffmpeg. The frames are saved as images with 
    the specified `output_extension` in the respective video subfolders.
    """
    videos = sorted([os.path.join(videos_dir, video_file_name) for video_file_name in os.listdir(videos_dir)])
    
    for video in videos:
        video_name = os.path.splitext(os.path.basename(video))[0]
        
        output_video_dir = os.path.join(output_dir, video_name)
        
        if os.path.exists(output_video_dir):
            if verbose:
                print(f"[INFO]: frames for \"{video_name}\" already exist. skipping extraction.")
            continue
        
        os.makedirs(output_video_dir)
        
        if verbose:
            print(f"[INFO]: extracting frames from {video_name}...")
        
        os.system(f"ffmpeg -i {video} {output_video_dir}/img_%05d.{output_extension}")
        
        if verbose:
            print(f"[INFO]: frames from {video_name} extracted successfully.")
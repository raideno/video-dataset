# Video Dataset

This is a python library to create a video dataset. The project is inspired from [Video-Dataset-Loading-Pytorch
](https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch) but with a lot of additional features and modifications.

The goal is to have a very moldable and customizable video dataset that can be reused in all possible video dataset situations.

## Installation

```bash
pip install video-dataset
```

## Supported Dataset Structures

### Raw Videos

```txt
- your-dataset
- - videos
- - - video-1.mp4
- - - video-2.mp3
- - - ...
- - annotations
- - - video-1.csv
- - - video-2.csv
- - - ...
training_ids.txt
testing_ids.txt
validation_ids.txt
```

### Videos Frames

```txt
- your-dataset
- - videos
- - - video-1
- - - - img_00001.jpg
- - - - img_00002.jpg
- - - - img_00003.jpg
- - - - ...
- - - video-2
- - - ...
- - annotations
- - - video-1.csv
- - - video-2.csv
- - - ...
training_ids.txt
testing_ids.txt
validation_ids.txt
```

### Text Annotations

## Usage

# Todos

- [ ] Explore for faster methods of reading images to speed up the loading as 0.6 seconds is really too slow in my opinion.

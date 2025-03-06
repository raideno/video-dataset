# Video Dataset

This is a python library to create a video dataset. The project is inspired from [Video-Dataset-Loading-Pytorch
](https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch) but with a lot of additional features and modifications.

The goal is to have a very moldable and customizable video dataset that can be reused in all possible video dataset situations.

## Installation

```bash
pip install video-dataset
```

## Dataset Structures

The General dataset structure is the one specified below. One global directory where there is a sub directory for the videos and another one for the annotations, ids files are optional.

```txt
- your-dataset
- - videos
- - - video-1
- - - video-2
- - - ...
- - annotations
- - - video-1
- - - video-2
- - - ...
training_ids.txt
testing_ids.txt
validation_ids.txt
```

An important thing is that each the video must be named (except for the extension) the same way as it's corresponding annotation in order for the VideoDataset to correctly detect it.

When defining a video-dataset multiple components need to be defined:

- **videos_dir:** The path were the videos are stored.
- **annotations_dir** The path were the annotations are stored.
- **segment_size:** The desired number of frames per video $*_1$.
- **video_processor:** Will be in charge to read the video $*_2$.
- **annotations_processor:** Will be in charge to read the annotations $*_2$.

$*_1$: Suppose your videos contain 100 frames and you put `segment_size=10`; from each video you'll have 10 sub videos of 10 frames each. You an also consider the whole video by putting `segment_size=-1`.

$*_2:$ In the package, a number of predefined video and annotations processor are available and cover practically any case you can encounter, but it is also possible to defined a custom video or annotation processor and use it with the video-dataset.

## Video Processors

The Dataset supports multiple video formats, all the supported formats are presented below:

### Raw Video Representation

In this format each element in the videos directory need to be a video file (with any of the supported video extensions).

For example:

```txt
- your-dataset
- - videos
- - - video-1.mp4
- - - video-2.mp4
- - - ...
- - annotations
- - - ...
```

The corresponding VideoDataset:

```python
from video_dataset import VideoDataset
from video_dataset.video import VideoFromVideoFile
from video_dataset.annotations import AnnotationsFromFrameLevelTxtFileAnnotations

video_processor: Type[Video]
annotations_processor: Type[Annotations]

dataset = VideoDataset(
    videos_dir="./dataset/videos",
    annotations_dir="./dataset/annotations",
    segment_size=32,
    video_processor=VideoFromVideoFile,
    annotations_processor=AnnotationsFromFrameLevelTxtFileAnnotations,
)
```

### Frame Level Video Representation

Having the elements of the videos directory as raw videos can be quite slow when loading the videos, an alternative approach is that each element of the videos directory is a directory it self with the name of the video and the content of the directory is images where each image represent a single frame of the video.

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
- - - ...
```

The corresponding VideoDataset:

```python
from video_dataset import VideoDataset
from video_dataset.video import VideoFromVideoFramesDirectory
from video_dataset.annotations import AnnotationsFromFrameLevelTxtFileAnnotations

video_processor: Type[Video]
annotations_processor: Type[Annotations]

dataset = VideoDataset(
    videos_dir="./dataset/videos",
    annotations_dir="./dataset/annotations",
    segment_size=32,
    video_processor=VideoFromVideoFramesDirectory,
    annotations_processor=AnnotationsFromFrameLevelTxtFileAnnotations,
)
```

This significantly reduces video loading time but at the cost of storage space.

### Custom Processor

In order to create a custom video processor you basically need to create a class that implements the `Video` class as follow:

```python
from video_dataset.video import Video

class CustomVideoProcessor(Video):
    def __init__(self, videos_dir_path: str, id: str):
        ...

    def get_id(self):
        return self.id

    def __len__(self):
        ...

    def __getitem__(self, index: int | slice):
        """
        Return the corresponding video frame(s) requested by the index.
        """
        ...
```

## Annotations Processors

Your video annotations files can be in multiple formats.

### Whole Video Annotations

A single csv or txt file describing the classes / labels of all the videos.

- [ ] Implementation. Coming Soon..

### Frame By Frame Annotations

Each video have a corresponding txt file where each line in the file correspond to a class / label / annotation of a frame in the video.

```txt
eating
eating
eating
eating
eating
eating
eating
...
```

The corresponding VideoDataset:

```python
from video_dataset import VideoDataset
from video_dataset.video import VideoFromVideoFile
from video_dataset.annotations import AnnotationsFromFrameLevelTxtFileAnnotations

video_processor: Type[Video]
annotations_processor: Type[Annotations]

dataset = VideoDataset(
    videos_dir="./dataset/videos",
    annotations_dir="./dataset/annotations",
    segment_size=32,
    video_processor=VideoFromVideoFile,
    annotations_processor=AnnotationsFromFrameLevelTxtFileAnnotations,
)
```

### Segment Level Annotations

Each video has a corresponding csv file with the following structure:

| acton   | starting-timestamp | duration |
| ------- | ------------------ | -------- |
| eating  | 0                  | 4000     |
| dancing | 4000               | 6000     |
| eating  | 10000              | 8000     |

The corresponding VideoDataset:

```python
from video_dataset import VideoDataset
from video_dataset.video import VideoFromVideoFile
from video_dataset.annotations import AnnotationsFromSegmentLevelCsvFileAnnotations

video_processor: Type[Video]
annotations_processor: Type[Annotations]

dataset = VideoDataset(
    videos_dir="./dataset/videos",
    annotations_dir="./dataset/annotations",
    segment_size=32,
    video_processor=VideoFromVideoFile,
    annotations_processor=AnnotationsFromSegmentLevelCsvFileAnnotations,
)
```

### Custom Processor

In order to create a custom annotations processor you basically need to create a class that implements the `Annotations` class as follow:

```python
from video_dataset.annotations import Annotations

class CustomAnnotationsProcessor(Annotations):
    def __init__(self, annotations_dir_path: str, id: str):
        ...

    def get_id(self):
        return self.id

    @abstractmethod
    def __getitem__(self, index: int | slice):
        """
        Get the annotation(s) of the video file corresponding to the given frame(s) index / indices.
        Note that even if an index is given the annotations will be returned in a batch format (Number of frames, Height, Width, Channels).
        """
        ...
```

## Contributions

All contributions are welcome, just open a pull request.

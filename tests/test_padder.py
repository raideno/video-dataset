import numpy as np
import pytest

from video_dataset.padder import  ValuePadder, LastValuePadder

@pytest.mark.parametrize("padder_class,frames_padding_value,annotations_padding_value,expected_last_value", [
    (ValuePadder, 0, -1, 0),
    (LastValuePadder, None, None, None),
])
def test_padder(padder_class, frames_padding_value, annotations_padding_value, expected_last_value):
    original_segment_size = 5
    target_segment_size = 8

    frames = np.random.randint(0, 255, (original_segment_size, 8, 8, 1), dtype=np.uint8)
    annotations = list(range(original_segment_size))

    if padder_class == ValuePadder:
        padder = padder_class(frames_padding_value, annotations_padding_value)
    else:
        padder = padder_class()

    padded_frames, padded_annotations = padder(frames, annotations, target_segment_size)

    if padder_class == ValuePadder:
        assert np.all(padded_frames[original_segment_size:] == frames_padding_value), "ValuePadder frames padding incorrect!"
        assert padded_annotations[original_segment_size:] == [annotations_padding_value] * (target_segment_size - original_segment_size), "ValuePadder annotations padding incorrect!"
    
    elif padder_class == LastValuePadder:
        assert np.array_equal(padded_frames[original_segment_size:], np.tile(frames[-1], (target_segment_size - original_segment_size, 1, 1, 1))), "LastValuePadder frames padding incorrect!"
        assert padded_annotations[original_segment_size:] == [annotations[-1]] * (target_segment_size - original_segment_size), "LastValuePadder annotations padding incorrect!"
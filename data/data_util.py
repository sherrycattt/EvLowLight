import glob
import os
import os.path as osp
import time
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib as mpl
import numpy as np
import torch
import tqdm
from PIL import Image
from basicsr.data.transforms import mod_crop
from basicsr.utils import imwrite
from basicsr.utils import scandir
from pathlib2 import Path

try:
    import h5py
except ImportError:
    raise ImportError('Please install h5py to enable LmdbBackend.')

time_fmt = "%Y-%m-%d %H:%M:%S"
TIMESTAMP_COLUMN = 2
X_COLUMN = 0
Y_COLUMN = 1
POLARITY_COLUMN = 3

HEIGHT = 480
WIDTH = 854
NUM_BIN = 15
FRAME_FPS = 25


def read_img_seq(path, mod_size=1, return_imgname=False, load_size=None):
    if isinstance(path, list):
        img_paths = path
    else:
        img_paths = sorted(list(scandir(path, full_path=True)))
    # the same as imfrombytes in basicsr.data.data_util.py
    imgs = [cv2.imread(v).astype(np.float32) / 255. for v in img_paths]

    if load_size:
        imgs = [cv2.resize(img, tuple(load_size), cv2.INTER_CUBIC) for img in imgs]

    if mod_size > 1:
        imgs = [mod_crop(img, mod_size) for img in imgs]

    # imgs = img2tensor(imgs, bgr2rgb=True, float32=True)
    # imgs = torch.stack(imgs, dim=0)

    if return_imgname:
        imgnames = [osp.splitext(osp.basename(path))[0] for path in img_paths]
        return imgs, imgnames
    else:
        return imgs


def save_events(events, file, is_dict=False):
    """Save events to ".npy" file.

    In the "events" array columns correspond to: x, y, timestamp, polarity.

    We store:
    (1) x,y coordinates with uint16 precision.
    (2) timestamp with float32 precision.
    (3) polarity with binary precision, by converting it to {0,1} representation.

    """
    if (0 > events[:, X_COLUMN]).any() or (events[:, X_COLUMN] > 2 ** 16 - 1).any():
        raise ValueError("Coordinates should be in [0; 2**16-1].")
    if (0 > events[:, Y_COLUMN]).any() or (events[:, Y_COLUMN] > 2 ** 16 - 1).any():
        raise ValueError("Coordinates should be in [0; 2**16-1].")
    if ((events[:, POLARITY_COLUMN] != -1) & (events[:, POLARITY_COLUMN] != 1)).any():
        raise ValueError("Polarity should be in {-1,1}.")
    events = np.copy(events)
    if is_dict:
        x, y, timestamp, polarity = np.hsplit(events, events.shape[1])
        polarity = (polarity + 1) / 2
        events = dict(
            x=x.astype(np.uint16),
            y=y.astype(np.uint16),
            timestamp=timestamp.astype(np.float32),
            polarity=polarity.astype(np.bool),
        )
        if file.endswith('.npz'):
            np.savez(file, *events)
        elif file.endswith('.h5'):
            with h5py.File(file, 'w') as f:
                f.create_dataset('events', data=events)
        else:
            raise ValueError
    else:
        (x, y, timestamp, polarity) = (
            events[:, X_COLUMN], events[:, Y_COLUMN], events[:, TIMESTAMP_COLUMN], events[:, POLARITY_COLUMN])
        events = np.stack((
            timestamp.reshape((-1,)),
            x.reshape((-1,)),
            y.reshape((-1,)),
            polarity.reshape((-1,)),
        ), axis=-1)
        if file.endswith('.txt'):
            np.savetxt(file, events, fmt="%f %d %d %d")
        elif file.endswith('.h5'):
            with h5py.File(file, 'w') as f:
                f.create_dataset('events', data=events)
        else:
            raise ValueError


def load_events(file, is_dict=False):
    """Load events to ".npz" file.

    See "save_events" function description.
    """
    file = str(file)
    f = None
    if not is_dict:
        if file.endswith('.txt'):
            try:
                tmp = np.loadtxt(file)
            except ValueError:
                tmp = np.loadtxt(file, skiprows=1)
        elif file.endswith('.h5'):
            f = h5py.File(file, 'r')
            tmp = f['events'][()]
        else:
            raise ValueError

        (x, y, timestamp, polarity) = (tmp[:, 1], tmp[:, 2], tmp[:, 0], tmp[:, 3])
    else:
        if file.endswith('.h5'):
            f = h5py.File(file, 'r')
            tmp = f['events'][()]
            (x, y, timestamp, polarity) = (tmp["x"][()], tmp["y"][()], tmp["t"][()], tmp["p"][()])
            ev_rect_file = Path(file).parent / 'rectify_map.h5'
            if ev_rect_file.exists():
                with h5py.File(str(ev_rect_file), 'r') as h5_rect:
                    rectify_ev_map = h5_rect['rectify_map'][()]
                    assert rectify_ev_map.shape == (480, 640, 2), rectify_ev_map.shape
                    assert x.max() < 640
                    assert y.max() < 480
                    xy_rect = rectify_ev_map[y, x]
                    x = xy_rect[:, 0]
                    y = xy_rect[:, 1]
            if 't_offset' in f:
                t_offset = int(f['t_offset'][()])
                timestamp = timestamp + t_offset
        elif file.endswith('.npz'):
            tmp = np.load(file, allow_pickle=True)
            (x, y, timestamp, polarity) = (tmp["x"], tmp["y"], tmp["t"], tmp["p"])
        else:
            raise ValueError
    if f is not None:
        f.close()

    if np.mean(polarity) < 1.0 and np.min(polarity) >= 0.0:
        polarity = polarity * 2 - 1
    events = np.stack((
        x.astype(np.float64).reshape((-1,)),
        y.astype(np.float64).reshape((-1,)),
        timestamp.astype(np.float64).reshape((-1,)),
        polarity.astype(np.float32).reshape((-1,)),
    ), axis=-1)

    events = events[events[:, X_COLUMN].argsort()]  # First sort doesn't need to be stable.
    events = events[events[:, Y_COLUMN].argsort(kind='mergesort')]
    events = events[events[:, TIMESTAMP_COLUMN].argsort(kind='mergesort')]

    return events


def plot_points_on_background(y, x, background, points_color=[0, 0, 255]):
    """Return PIL image with overlayed points.
    Args:
        x, y : numpy vectors with points coordinates (might be empty).
        background: (height x width x 3) torch tensor.
        color: color of points [red, green, blue] uint8.
    """
    if x.size == 0:
        return background
    background = np.array(background)
    if not (len(background.shape) == 3 and background.shape[-1] == 3):
        raise ValueError("background should be (height x width x color).")
    height, width, _ = background.shape
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    if not (x_min >= 0 and y_min >= 0 and x_max < width and y_max < height):
        raise ValueError('points coordinates are outsize of "background" ' "boundries.")
    background[y, x, :] = points_color
    return background


class EventSequence(object):
    """Stores events in oldes-first order."""

    def __init__(
            self, features, image_height=None, image_width=None, start_time=None, end_time=None, is_flip=False,
    ):
        """Returns object of EventSequence class.

        Args:
            features: numpy array with events softed in oldest-first order. Inside,
                      rows correspond to individual events and columns to event
                      features (x, y, timestamp, polarity)

            image_height, image_width: widht and height of the event sensor.
                                       Note, that it can not be inferred
                                       directly from the events, because
                                       events are spares.
            start_time, end_time: start and end times of the event sequence.
                                  If they are not provided, this function inferrs
                                  them from the events. Note, that it can not be
                                  inferred from the events when there is no motion.
        """
        self._features = features
        self._image_width = image_width or int(features[:, X_COLUMN].max() + 1)
        self._image_height = image_height or int(features[:, Y_COLUMN].max() + 1)
        self._start_time = start_time or features[0, TIMESTAMP_COLUMN]
        self._end_time = end_time or features[-1, TIMESTAMP_COLUMN]
        if is_flip:
            self.flip_vertically()
            # self.flip_horizontally()

    def __len__(self):
        return self._features.shape[0]

    def is_self_consistent(self):
        return (
                self.are_spatial_coordinates_within_range()
                and self.are_timestamps_ascending()
                and self.are_polarities_one_and_minus_one()
                and self.are_timestamps_within_range()
        )

    def are_spatial_coordinates_within_range(self):
        x = self._features[:, X_COLUMN]
        y = self._features[:, Y_COLUMN]
        return np.all((x >= 0) & (x < self._image_width)) and np.all(
            (y >= 0) & (y < self._image_height)
        )

    def are_timestamps_ascending(self):
        timestamp = self._features[:, TIMESTAMP_COLUMN]
        return np.all((timestamp[1:] - timestamp[:-1]) >= 0)

    def are_timestamps_within_range(self):
        timestamp = self._features[:, TIMESTAMP_COLUMN]
        return np.all((timestamp <= self.end_time()) & (timestamp >= self.start_time()))

    def are_polarities_one_and_minus_one(self):
        polarity = self._features[:, POLARITY_COLUMN]
        return np.all((polarity == -1) | (polarity == 1))

    def flip_horizontally(self):
        self._features[:, X_COLUMN] = (
                self._image_width - 1 - self._features[:, X_COLUMN]
        )

    def flip_vertically(self):
        self._features[:, Y_COLUMN] = (
                self._image_height - 1 - self._features[:, Y_COLUMN]
        )

    def reverse(self):
        """Reverse temporal direction of the event stream.

        Polarities of the events reversed.

                          (-)       (+)
        --------|----------|---------|------------|----> time
           t_start        t_1       t_2        t_end

                          (+)       (-)
        --------|----------|---------|------------|----> time
                0    (t_end-t_2) (t_end-t_1) (t_end-t_start)

        """
        if len(self) == 0:
            return
        self._features[:, TIMESTAMP_COLUMN] = (
                self._end_time - self._features[:, TIMESTAMP_COLUMN]
        )
        self._features[:, POLARITY_COLUMN] = -self._features[:, POLARITY_COLUMN]
        self._start_time, self._end_time = 0, self._end_time - self._start_time
        # Flip rows of the 'features' matrix, since it is sorted in oldest first.
        self._features = np.copy(np.flipud(self._features))

    def duration(self):
        return self.end_time() - self.start_time()

    def start_time(self):
        return self._start_time

    def end_time(self):
        return self._end_time

    def min_timestamp(self):
        return self._features[:, TIMESTAMP_COLUMN].min()

    def max_timestamp(self):
        return self._features[:, TIMESTAMP_COLUMN].max()

    def filter_by_polarity(self, polarity, make_deep_copy=True):
        mask = self._features[:, POLARITY_COLUMN] == polarity
        return self.filter_by_mask(mask, make_deep_copy)

    def copy(self):
        return EventSequence(
            features=np.copy(self._features),
            image_height=self._image_height,
            image_width=self._image_width,
            start_time=self._start_time,
            end_time=self._end_time,
        )

    def filter_by_mask(self, mask, make_deep_copy=True):
        if make_deep_copy:
            return EventSequence(
                features=np.copy(self._features[mask]),
                image_height=self._image_height,
                image_width=self._image_width,
                start_time=self._start_time,
                end_time=self._end_time,
            )
        else:
            return EventSequence(
                features=self._features[mask],
                image_height=self._image_height,
                image_width=self._image_width,
                start_time=self._start_time,
                end_time=self._end_time,
            )

    def filter_by_timestamp(self, start_time, duration, make_deep_copy=True):
        """Returns event sequence filtered by the timestamp.

        The new sequence includes event in [start_time, start_time+duration).
        """
        end_time = start_time + duration
        mask = (start_time <= self._features[:, TIMESTAMP_COLUMN]) & (
                end_time > self._features[:, TIMESTAMP_COLUMN]
        )

        event_sequence = self.filter_by_mask(mask, make_deep_copy)
        event_sequence._start_time = start_time
        event_sequence._end_time = start_time + duration
        return event_sequence

    def filter_by_window(self, start_x, crop_width, start_y, crop_height, make_deep_copy=True):
        """Returns event sequence filtered by the timestamp.

        The new sequence includes event in [start_time, start_time+duration).
        """
        x = self._features[:, X_COLUMN]
        y = self._features[:, Y_COLUMN]

        # Cropping (+- 2 for safety reasons)
        x_mask = (x >= start_x - 2) & (x < start_x + crop_width + 2)
        y_mask = (y >= start_y - 2) & (y < start_y + crop_height + 2)
        mask = x_mask & y_mask

        event_sequence = self.filter_by_mask(mask, make_deep_copy)
        event_sequence._image_height = crop_height
        event_sequence._image_width = crop_width
        return event_sequence

    def to_image(self, background=None):
        """Visualizes stream of event as a PIL image.

        The pixel is shown as red if dominant polarity of pixel's
        events is 1, as blue if dominant polarity of pixel's
        events is -1 and white if pixel does not recieve any events,
        or it's events does not have dominant polarity.

        Args:
            background: is PIL image.
        """
        polarity = self._features[:, POLARITY_COLUMN].astype(np.int8) == 1
        x_negative = self._features[~polarity, X_COLUMN].astype(np.int32)
        y_negative = self._features[~polarity, Y_COLUMN].astype(np.int32)
        x_positive = self._features[polarity, X_COLUMN].astype(np.int32)
        y_positive = self._features[polarity, Y_COLUMN].astype(np.int32)

        positive_histogram, _, _ = np.histogram2d(
            x_positive,
            y_positive,
            bins=(self._image_width, self._image_height),
            range=[[0, self._image_width], [0, self._image_height]],
        )
        negative_histogram, _, _ = np.histogram2d(
            x_negative,
            y_negative,
            bins=(self._image_width, self._image_height),
            range=[[0, self._image_width], [0, self._image_height]],
        )

        red = np.transpose(positive_histogram > negative_histogram)
        blue = np.transpose(positive_histogram < negative_histogram)

        if background is None:
            height, width = red.shape
            background = Image.fromarray(
                np.full((height, width, 3), 255, dtype=np.uint8)
            )
        y, x = np.nonzero(red)
        points_on_background = plot_points_on_background(
            y, x, background, [255, 0, 0]
        )
        y, x = np.nonzero(blue)
        points_on_background = plot_points_on_background(
            y, x, points_on_background, [0, 0, 255]
        )
        return points_on_background

    def _advance_index_to_timestamp(self, timestamp, start_index=0):
        """Returns index of the first event with timestamp > "timestamp" from "start_index"."""
        index = start_index
        while index < len(self):
            if self._features[index, TIMESTAMP_COLUMN] >= timestamp:
                return index
            index += 1
        return min(index, len(self) - 1)

    def split_in_two(self, timestamp):
        """Returns two sequences from splitting the original sequence in two."""
        if not (self.start_time() <= timestamp <= self.end_time()):
            raise ValueError(
                '"timestamps" should be between start and end of the sequence.'
            )
        first_sequence_duration = timestamp - self.start_time()
        second_sequence_duration = self.end_time() - timestamp
        first_sequence = self.filter_by_timestamp(
            self.start_time(), first_sequence_duration
        )
        second_sequence = self.filter_by_timestamp(timestamp, second_sequence_duration)
        return first_sequence, second_sequence

    def make_iterator_over_splits(self, number_of_splits):
        """Returns iterator over splits in two.

        E.g, if "number_of_splits" = 3, than the iterator will output
        (t_start->t_0, t_0->t_end)
        (t_start->t_1, t_1->t_end)
        (t_start->t_2, t_2->t_end)

        ---|------|------|------|------|--->
         t_start  t0     t1    t2     t_end

        t0 = (t_end - t_start) / (number_of_splits + 1), and ect.
        """
        start_time = self.start_time()
        end_time = self.end_time()
        split_timestamps = np.linspace(start_time, end_time, number_of_splits + 2)[1:-1]

        for split_timestamp in split_timestamps:
            left_events, right_events = self.split_in_two(split_timestamp)
            yield left_events, right_events

    def make_sequential_iterator(self, timestamps, delta_t_us=None, delta_cnt=None):
        """Returns iterator over sub-sequences of events.
        """
        if len(timestamps) < 2:
            raise ValueError("There should be at least two timestamps")
        if delta_cnt is not None:
            delta_cnt = int(delta_cnt)
            iterator_list = []
            start_timestamp = timestamps[0]
            start_index = self._advance_index_to_timestamp(start_timestamp)

            for end_timestamp in timestamps[1:]:
                assert start_timestamp < end_timestamp
                middle_timestamp = (start_timestamp + end_timestamp) / 2.
                middle_index = self._advance_index_to_timestamp(middle_timestamp, start_index)
                iterator_list.append(max(0, middle_index - delta_cnt))
                iterator_list.append(middle_index)
                start_index = middle_index
                start_timestamp = end_timestamp

            for current_idx in iterator_list:
                assert not self._features[current_idx:(current_idx + delta_cnt), :].size == 0
                yield EventSequence(
                    features=np.copy(self._features[current_idx:(current_idx + delta_cnt), :]),
                    image_height=self._image_height,
                    image_width=self._image_width,
                    start_time=self._features[current_idx, TIMESTAMP_COLUMN],
                    end_time=self._features[current_idx + delta_cnt, TIMESTAMP_COLUMN],
                )

        elif delta_t_us is not None:
            iterator_list = []
            iterator_list_end = []
            start_index = 0
            start_index2 = 0

            start_timestamp = timestamps[0]
            for end_timestamp in timestamps[1:]:
                assert start_timestamp < end_timestamp
                middle_timestamp = (start_timestamp + end_timestamp) / 2.

                middle_index = self._advance_index_to_timestamp(middle_timestamp - delta_t_us, start_index)
                iterator_list.append(middle_index)
                start_index = middle_index

                middle_index2 = self._advance_index_to_timestamp(middle_timestamp, start_index2)
                iterator_list_end.append(middle_index2)
                start_index2 = middle_index2

                middle_index3 = self._advance_index_to_timestamp(middle_timestamp, middle_index2)
                iterator_list.append(middle_index3)

                middle_index3 = self._advance_index_to_timestamp(middle_timestamp + delta_t_us, middle_index3)
                iterator_list_end.append(middle_index3)

                start_timestamp = end_timestamp

            for current_idx, end_index in zip(iterator_list, iterator_list_end):
                assert not self._features[current_idx:end_index, :].size == 0
                yield EventSequence(
                    features=np.copy(self._features[current_idx:end_index, :]),
                    image_height=self._image_height,
                    image_width=self._image_width,
                    start_time=self._features[current_idx, TIMESTAMP_COLUMN],
                    end_time=self._features[end_index, TIMESTAMP_COLUMN],
                )
        else:
            start_timestamp = timestamps[0]
            start_index = self._advance_index_to_timestamp(start_timestamp)

            for end_timestamp in timestamps[1:]:
                middle_index = self._advance_index_to_timestamp(end_timestamp, start_index)
                assert self._features[start_index:middle_index, :].size != 0
                yield EventSequence(
                    features=np.copy(self._features[start_index:middle_index, :]),
                    image_height=self._image_height,
                    image_width=self._image_width,
                    start_time=start_timestamp,
                    end_time=end_timestamp,
                )
                start_index = middle_index
                start_timestamp = end_timestamp

    def to_file(self, filename):
        """Saves event sequences from to npz.

        Args:
            folder: folder where events will be saved in the files events_000000.npz,
                    events_000001.npz, etc.
            timestamps: iterator that outputs event sequences.
        """
        save_events(self._features, filename)

    def to_folder(self, folder, timestamps, event_file_template="{:06d}", delta_t_us=None):
        """Saves event sequences from to npz.

        Args:
            folder: folder where events will be saved in the files events_000000.npz,
                    events_000001.npz, etc.
            timestamps: iterator that outputs event sequences.
        """
        event_iterator = self.make_sequential_iterator(timestamps, delta_t_us)
        for sequence_index, sequence in enumerate(event_iterator):
            filename = os.path.join(folder, event_file_template.format(sequence_index))
            save_events(sequence._features, filename)

    @classmethod
    def from_folder(
            cls,
            folder,
            image_height=None,
            image_width=None,
            event_file_template="{:06d}.npz",
            is_dict=False,
            is_flip=False,
            return_iterator=False,
    ):
        filename_iterator = sorted(glob.glob(
            os.path.join(folder, event_file_template)
        ))
        filenames = [filename for filename in filename_iterator]
        if return_iterator:
            for filename in filename_iterator:
                yield cls.from_files([filename], image_height, image_width, is_dict=is_dict, is_flip=is_flip)
        else:
            return cls.from_files(filenames, image_height, image_width, is_dict=is_dict, is_flip=is_flip)

    @classmethod
    def from_files(
            cls,
            list_of_filenames,
            image_height=None,
            image_width=None,
            start_time=None,
            end_time=None,
            is_dict=False,
            is_flip=False,
    ):
        """Reads event sequence from numpy file list."""
        if len(list_of_filenames) > 1:
            features_list = []
            for f in tqdm.tqdm(list_of_filenames):
                if os.stat(f).st_size == 0:
                    continue
                features_list += [load_events(f, is_dict)]  # for filename in list_of_filenames]
            features = np.concatenate(features_list)
        else:
            features = load_events(list_of_filenames[0], is_dict)

        return EventSequence(features, image_height, image_width, start_time, end_time, is_flip=is_flip)

    def to_voxel_grid(self, nb_of_time_bins=5, remapping_maps=None, normalize=True):
        """Returns voxel grid representation of event steam.

        In voxel grid representation, temporal dimension is
        discretized into "nb_of_time_bins" bins. The events fir
        polarities are interpolated between two near-by bins
        using bilinear interpolation and summed up.

        If event stream is empty, voxel grid will be empty.
        """
        C, H, W = nb_of_time_bins, self._image_height, self._image_width
        voxel_grid = torch.zeros(C, H, W, dtype=torch.float32, device='cpu', requires_grad=False)

        features = torch.from_numpy(self._features)
        x = features[:, X_COLUMN]
        y = features[:, Y_COLUMN]
        polarity = features[:, POLARITY_COLUMN].float()
        t = features[:, TIMESTAMP_COLUMN].float()

        # Convert timestamps to [0, nb_of_time_bins] range.
        t = (t - self.start_time()) * (C - 1) / self.duration()

        if remapping_maps is not None:
            remapping_maps = torch.from_numpy(remapping_maps)
            x, y = remapping_maps[:, y, x]

        left_t, right_t = t.floor(), t.floor() + 1
        left_x, right_x = x.floor(), x.floor() + 1
        left_y, right_y = y.floor(), y.floor() + 1

        for lim_x in [left_x, right_x]:
            for lim_y in [left_y, right_y]:
                for lim_t in [left_t, right_t]:
                    mask = (0 <= lim_x) & (0 <= lim_y) & (0 <= lim_t) & (lim_x < W) & (lim_y < H) & (lim_t < C)
                    lin_idx = lim_x.long() + lim_y.long() * W + lim_t.long() * W * H
                    weight = polarity * (1 - (lim_x - x).abs()) * (1 - (lim_y - y).abs()) * (1 - (lim_t - t).abs())
                    voxel_grid.put_(lin_idx[mask], weight[mask].float(), accumulate=True)  # eraft
        return voxel_grid.permute(1, 2, 0).contiguous().cpu().numpy()

    def visualize_event(self):
        # print(events.shape)
        polarity = self._features[:, POLARITY_COLUMN].astype(np.int8) == 1
        x = self._features[::, X_COLUMN].astype(np.uint64)
        y = self._features[::, Y_COLUMN].astype(np.uint64)
        stack = np.zeros((self._image_height, self._image_width))
        p_col = np.where(polarity, 1, -1)
        np.add.at(stack, (y, x), p_col)

        norm = mpl.colors.Normalize(vmin=-3, vmax=3, clip=True)
        color = mpl.cm.get_cmap("bwr")
        n = norm(stack)
        c = color(n)
        return np.uint8(c * 255)[:, :, 0:3]


def voxel2tensor(voxels, float32=True):
    def _totensor(voxel, float32):
        voxel = torch.from_numpy(voxel.transpose(2, 0, 1))
        if float32:
            voxel = voxel.float()
        return voxel

    if isinstance(voxels, list):
        return [_totensor(img, float32) for img in voxels]
    else:
        return _totensor(voxels, float32)


def even_from_clip(path, iterate_index=None,
                   time_tmpl='timestamp.txt', frame_subdir=None, image_height=None, image_width=None,
                   delta_t_us=100000,  # delta_t_us: 100000   # [100000], null
                   delta_cnt=None,  # delta_cnt: null   # [null],  400000
                   is_dict=False,
                   is_flip=False,
                   ):
    event_sequence = EventSequence.from_files(
        [str(path)], image_height or HEIGHT, image_width or WIDTH, is_dict=is_dict, is_flip=is_flip)

    # Load and compute timestamps
    if time_tmpl is not None:
        timestamps = np.loadtxt(str(Path(path).parent / time_tmpl), 'float32')
        if timestamps[-1] < 1000.0 and delta_t_us:
            delta_t_us = float(delta_t_us) / 1000000.0
    elif frame_subdir is not None:
        paths = sorted(list(scandir(str(Path(path).parent).replace('/events', f"/{frame_subdir}"), full_path=True)))
        num_frame = len(paths)
        timestamps = np.linspace(0, num_frame * (1000000 / FRAME_FPS), num_frame, endpoint=False)
    else:
        raise ValueError

    if iterate_index is not None:
        timestamps = [timestamps[i] for i in iterate_index]
    print(f"Creating event iterator of length {len(timestamps)}")

    return event_sequence.make_sequential_iterator(timestamps, delta_t_us=delta_t_us, delta_cnt=delta_cnt)


def read_voxel_seq(path, mod_size=1, return_imgname=False, suffix=None,
                   iterate_index=None, time_tmpl='timestamp.txt', frame_subdir=None,
                   real_size=None, num_bin=None, remapping_maps=None, load_size=None,
                   is_dict=False, is_flip=False):
    if isinstance(path, list):
        event_paths = path
    else:
        event_paths = sorted(list(scandir(path, full_path=True, suffix=suffix)))
    if event_paths[0].endswith('.npy'):
        voxels = [np.load(p).astype(np.float32) for p in event_paths]
    else:
        real_w, real_h = (WIDTH, HEIGHT) or real_size
        if len(event_paths) == 1 and (not isinstance(path, list)):
            seqs = even_from_clip(path=event_paths[0],
                                  iterate_index=iterate_index, time_tmpl=time_tmpl, frame_subdir=frame_subdir,
                                  image_height=real_h,
                                  image_width=real_w,
                                  is_dict=is_dict, is_flip=is_flip)
        else:
            seqs = [EventSequence.from_files(
                [p], image_height=real_h, image_width=real_w, is_dict=is_dict, is_flip=is_flip
            ) for p in event_paths]
        voxels = [seq.to_voxel_grid(num_bin or NUM_BIN, remapping_maps=remapping_maps) for seq in seqs]

    if load_size:
        voxels = [cv2.resize(img, tuple(load_size), cv2.INTER_CUBIC) for img in voxels]

    if mod_size > 1:
        voxels = [mod_crop(img, mod_size) for img in voxels]

    # voxels = voxel2tensor(voxels)
    # voxels = torch.stack(voxels, dim=0)

    if return_imgname:
        imgnames = [osp.splitext(osp.basename(path))[0] for path in event_paths]
        return voxels, imgnames
    else:
        return voxels


def write_voxel_seq(data_path, folder, key_pattern='{:05d}',
                    save_npy=True, save_png=False, new_dir='voxels',
                    filename_tmpl="{}.h5",
                    image_height=None, image_width=None, iterate_index=None,
                    ):
    start_time = time.time()
    print(f'Start saving {folder} at {datetime.fromtimestamp(start_time).strftime(time_fmt)}')
    seqs = even_from_clip(path=Path(data_path) / folder / filename_tmpl.format(folder),
                          iterate_index=iterate_index,
                          image_height=image_height, image_width=image_width,
                          )
    outs = []
    for idx, seq in enumerate(seqs):
        voxel_grid = seq.to_voxel_grid(NUM_BIN)

        voxel_root = Path(str(data_path).replace('events', new_dir)) / folder

        if save_npy:
            os.makedirs(str(voxel_root), exist_ok=True)
            np.save(str(voxel_root / key_pattern.format(idx)), voxel_grid)

        if save_png:
            imwrite(seq.to_image(), file_path=str(voxel_root / f"event_{key_pattern.format(idx)}.png"))

        outs.append((
            f"{folder}/{key_pattern.format(idx)}",  # key
            voxel_grid,  # data
            f"({seq._image_height},{seq._image_width},{NUM_BIN})",  # shape
            f"({seq.start_time()},{seq.end_time()})",  # meta_info
        ))
    return folder, outs

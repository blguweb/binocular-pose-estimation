from pathlib import Path

import yaml
import numpy as np

import cv2
import hdf5plugin
import h5py

import dataset_constants
from event_slicer import EventSlicer


class DSEC(object):
    def __init__(self, seq_path):
        self._events_time_horizon = int(0.030 * 1e6)

        assert self._events_time_horizon <= 100000, 'duration must be no higher than 100 ms'

        seq_path = Path(seq_path)
        # load disparity timestamps
        disp_dir = seq_path / 'disparity'
        self.timestamps = np.loadtxt(disp_dir / 'timestamps.txt', dtype='int64')
        # load disparity paths
        ev_disp_dir = disp_dir / 'event'
        self.disp_gt_pathstrings = sorted([str(entry) for entry in ev_disp_dir.iterdir() if str(entry.name).endswith('.png')])
        # load baseline
        calibration_path = seq_path / 'calibration' / 'cam_to_cam.yaml'
        with open(str(calibration_path)) as file:
            calibration_dict = yaml.load(file, Loader=yaml.FullLoader)
        calibration_matrix = np.asarray(calibration_dict['disparity_to_depth']['cams_03'])
        assert np.all(calibration_matrix[2, :3] == 0) and np.all(calibration_matrix[3, :2] == 0) and calibration_matrix[3, 3] == 0
        self.baseline = calibration_matrix[2, 3] / calibration_matrix[3, 2]

        assert len(self.disp_gt_pathstrings) == self.timestamps.size
        # Remove first disparity path and corresponding timestamp.
        # This is necessary because we do not have events before the first disparity map.
        assert int(Path(self.disp_gt_pathstrings[0]).stem) == 0
        self.disp_gt_pathstrings.pop(0)
        self.timestamps = self.timestamps[1:]

        self.rectify_ev_maps = dict()
        self.event_slicers = dict()

        ev_dir = seq_path / 'events'
        for location in ['left', 'right']:
            ev_dir_location = ev_dir / location
            self.event_slicers[location] = EventSlicer(str(ev_dir_location / 'events.h5'))

            with h5py.File(str(ev_dir_location / 'rectify_map.h5'), 'r') as h5_rect:
                self.rectify_ev_maps[location] = h5_rect['rectify_map'][()]

    @staticmethod
    def get_disparity_map(filepath: Path):
        assert filepath.is_file()
        disp_16bit = cv2.imread(str(filepath), cv2.IMREAD_ANYDEPTH)
        invalid_disparity = (disp_16bit == dataset_constants.INVALID_DISPARITY)
        disparity_image = disp_16bit.astype('float32') / dataset_constants.DISPARITY_MULTIPLIER
        disparity_image[invalid_disparity] = float('inf')

        return disparity_image

    def rectify_events(self, x: np.ndarray, y: np.ndarray, location: str):
        assert location in ['left', 'right']
        # From distorted to undistorted
        rectify_map = self.rectify_ev_maps[location]
        assert rectify_map.shape == (dataset_constants.IMAGE_HEIGHT, dataset_constants.IMAGE_WIDTH, 2), rectify_map.shape
        assert x.max() < dataset_constants.IMAGE_WIDTH
        assert y.max() < dataset_constants.IMAGE_HEIGHT
        return rectify_map[y, x]

    def __len__(self):
        return len(self.disp_gt_pathstrings)

    def get_example(self, index):
        ts_end = self.timestamps[index]
        ts_start = ts_end - self._events_time_horizon

        event_sequence = {}
        for location in ['left', 'right']:
            event_data = self.event_slicers[location].get_events(ts_start, ts_end)

            p = event_data['p']
            t = event_data['t']
            x = event_data['x']
            y = event_data['y']

            xy_rect = self.rectify_events(x, y, location)
            x_rect = np.clip(np.round(xy_rect[:, 0]), 0, dataset_constants.IMAGE_WIDTH-1)
            y_rect = np.clip(np.round(xy_rect[:, 1]), 0, dataset_constants.IMAGE_HEIGHT-1)

            events_array = np.stack([t/1e6, x_rect, y_rect, p.astype(np.int32)*2 - 1], axis=-1)

            event_sequence[location] = events_array

        return {
            'left': {
                'event_sequence': event_sequence['left'],
                'disparity_image': self.get_disparity_map(Path(self.disp_gt_pathstrings[index]))
            },
            'right': {
                'event_sequence': event_sequence['right']
            },
            'baseline': self.baseline
        }


if __name__ == '__main__':
    dataset = DSEC('zurich_city_04_c')
    print(len(dataset))
    for index in range(len(dataset)):
        data = dataset.get_example(index)
        print(data['left']['event_sequence'].shape)
        print(data['left']['disparity_image'].shape)
        print(data['right']['event_sequence'].shape)
        print(data['baseline'])

        break
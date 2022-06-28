# src: https://raw.githubusercontent.com/oidelima/Deepfake-Detection/master/src/dataloaders/dataset.py
# modified by cocoaaa
# 2022-02-09
import os
from sklearn.model_selection import train_test_split

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from .default_path import PathDict
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.core.debugger import set_trace as breakpoint
np.random.seed(420)


FLOW = True

class VideoDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            dataset (str): Name of dataset. Defaults to 'ucf101'.
            datatype (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
                options: 'train', 'validation', 'test'
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
            preprocess (bool): Determines whether to preprocess dataset. Default is False.
            extract_every (int): extract every N frames in the video into an image
    """

    def __init__(self,
                 dataset='celeb-df',
                 datatype='train',
                 clip_len=16,
                 preprocess=False,
                 extract_every=4):
        self.data_root, self.out_dir = PathDict.get_root_and_output_dirs(dataset)
        self.clip_len = clip_len
        self.datatype = datatype
        datatype_dir = self.out_dir/datatype

    # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 256#128#256# 128
        self.resize_width = 256#128#256#171
        self.crop_size = 224#112#112
        self.extract_every = extract_every
        # if not self.check_integrity():
        #     raise RuntimeError('Dataset not found or corrupted.' +
        #                        ' You need to download it from official website.')

        if preprocess: # or (not self.check_preprocess())
            print('Preprocessing of {} dataset, this will take long, but it will be done only once.'.format(dataset))
            self.preprocess()


        # Obtain all the filenames of files inside all the class folders
        # Going through each class folder one at a time
        self.fnames, labels = [], []
        for label in sorted(os.listdir(datatype_dir)):
            for fname in os.listdir(os.path.join(datatype_dir, label)):
                self.fnames.append(os.path.join(datatype_dir, label, fname))
                labels.append(label)



        assert len(labels) == len(self.fnames)
        print('Number of {} videos: {:d}'.format(datatype, len(self.fnames)))

        # Prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # Convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int) # one label for each video file (e.g. Celeb-real, Celeb-Synthesis, Youtube-Real)

        if dataset == "ucf101":
            if not os.path.exists('dataloaders/ucf_labels.txt'):
                with open('dataloaders/ucf_labels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id+1) + ' ' + label + '\n')

        elif dataset == 'hmdb51':
            if not os.path.exists('dataloaders/hmdb_labels.txt'):
                with open('dataloaders/hmdb_labels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id+1) + ' ' + label + '\n')

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # Loading and preprocessing.
        buffer = self.load_frames(self.fnames[index])
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        label = np.array(self.label_array[index])

        if self.datatype == 'test':
            # Perform data augmentation
            buffer = self.randomflip(buffer)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer), torch.from_numpy(label)

    def check_integrity(self):
        if not os.path.exists(self.data_root):
            return False
        else:
            return True

    def check_preprocess(self):
        # TODO: Check image size in output_dir
        if not os.path.exists(self.out_dir):
            return False
        elif not os.path.exists(os.path.join(self.out_dir, 'train')):
            return False

        for ii, video_class in enumerate(os.listdir(os.path.join(self.out_dir, 'train'))):
            for video in os.listdir(os.path.join(self.out_dir, 'train', video_class)):
                video_name = os.path.join(os.path.join(self.out_dir, 'train', video_class, video),
                                          sorted(os.listdir(os.path.join(self.out_dir, 'train', video_class, video)))[0])
                image = cv2.imread(video_name)
                if np.shape(image)[0] != 128 or np.shape(image)[1] != 171:
                    return False
                else:
                    break

            if ii == 10:
                break

        return True

    def preprocess(self):

        if not self.out_dir.exists():
            self.out_dir.mkdir(parents=True)
            (self.out_dir/'train').mkdir()
            (self.out_dir, 'validation').mkdir()
            (self.out_dir, 'test').mkdir()

        # Split train/val/test sets <-- here is the keypoint
        for label_dir in self.data_root.iterdir():
            if not label_dir.is_dir():
                continue

            label = label_dir.stem
            video_fps = [fn for fn in label_dir.iterdir()]

            train_and_valid, test_fps = train_test_split(video_fps, test_size=0.2, random_state=42)
            train_fps, val_fps = train_test_split(train_and_valid, test_size=0.2, random_state=42)

            train_label_dir = self.out_dir/'train'/label
            val_label_dir = self.out_dir/'validation'/label
            test_label_dir = self.out_dir/'test'/label
            # make output dirs if not exist
            out_dirs = [train_label_dir, val_label_dir, test_label_dir]
            for out_dir in out_dirs:
                if not out_dir.exists():
                    out_dir.mkdir(parents=True)

            # process extracting images from videos
            for video_fp in tqdm(train_fps):
                self.process_video(video_fp, label, train_label_dir)

            for video_fp in tqdm(val_fps):
                self.process_video(video_fp, label, val_label_dir)

            for video_fp in tqdm(test_fps):
                self.process_video(video_fp, label, test_label_dir)

            print('Preprocessing finished: ', label_dir.stem)

    print('Image extraction from all videos finished.')

    def process_video(self, video_fp, label, save_dir):
        # Initialize a VideoCapture object to read video data into a numpy array
        # print(video_fn)
        video_fn = video_fp.name
        capture = cv2.VideoCapture(str(self.data_root/label/video_fn)) #essentially str(video_fp)

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # print("Height = ", frame_height)
        # print("Width = ", frame_width)
        # print("Frame count = ", frame_count)
        # breakpoint()
        if frame_count < 16:
            print("Video = ", video_fp)
            print("Frame count too small: ", frame_count)
            return

        # if not save_dir/video_fn:
        #     os.mkdir(os.path.join(save_dir, video_fp))

        # Make sure splited video has at least 16 frames
        # todo: should be dealt as a while loop
        if frame_count // self.extract_every <= 16:
            self.extract_every -= 1
            if frame_count // self.extract_every <= 16:
                self.extract_every -= 1
                if frame_count // self.extract_every <= 16:
                    self.extract_every -= 1

        count = 0
        i = 0
        retaining = True
        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            if frame is None:
                continue

            if count % self.extract_every == 0:
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                _ = cv2.imwrite(filename= str(save_dir/f'{video_fn}_{str(i).zfill(2)}.jpg'),
                                      img=frame)
                i += 1
                count += 1

        # Release the VideoCapture once it is no longer needed
        capture.release()

    def compute_TVL1(self, prev, curr, bound=15):
        """Compute the TV-L1 optical flow."""

        TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
        flow = TVL1.calc(prev, curr, None)
        flow = np.clip(flow, -20,20) #default values are +20 and -20
        #print(flow)
        assert flow.dtype == np.float32

        flow = (flow + bound) * (255.0 / (2*bound))
        flow = np.round(flow).astype(int)
        flow[flow >= 255] = 255
        flow[flow <= 0] = 0

        return flow

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer


    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame

        return buffer

    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering


        time_index = np.random.randint(buffer.shape[0] - clip_len)

        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames

        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]


        return buffer


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    train_data = VideoDataset(dataset='celeb-df', split='test', clip_len=8, preprocess=False)
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=4)

    for i, sample in enumerate(train_loader):
        inputs = sample[0]
        labels = sample[1]
        print(inputs.size())
        print(labels)

        if i == 1:
            break
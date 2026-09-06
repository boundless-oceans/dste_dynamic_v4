# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import torch
import torch.utils.data as data

from PIL import Image
import os
import random
import numpy as np
from numpy.random import randint

import json

# class VideoRecord(object):
#     def __init__(self, row):
#         self._data = row

#     @property
#     def path(self):
#         return self._data[0]

#     @property
#     def num_frames(self):
#         return int(self._data[1])

#     @property
#     def label(self):
#         return int(self._data[2])

def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data["labels"]:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map

def get_video_names_and_annotations(data):
    video_names = []
    annotations = []
    subset = next(iter(data["database"].values()))["subset"]
    for key, value in data["database"].items():
        # video_names.append(key.split("_")[0])       #   ego
        # video_names.append(key.split("^")[0])       # hci
        video_names.append(key.split("^")[0])    # ipnhand
        annotations.append(value["annotations"])
    return video_names, annotations, subset

def load_annotation_data(annotation_path):
    with open(annotation_path, "r") as data_file:
        return json.load(data_file)

def make_dataset(dataset_path, annotation_path):
    data = load_annotation_data(annotation_path)
    video_names, annotations, subset = get_video_names_and_annotations(data)
    class_to_index = get_class_labels(data)
    index_to_class = {}
    for name, index in class_to_index.items():
        index_to_class[index] = name

    dataset = []

    print(subset + " dataset is loading...")

    for i in range(len(video_names)):
        if i % 1000 == 0:
            print("dataset loading [{}/{}]".format(i, len(video_names)))
        
        video_path = os.path.join(dataset_path, video_names[i])
        
        if not os.path.exists(video_path):
            print("video path not exist!!!", video_path)
            continue

        start_frame = int(float(annotations[i]["start_frame"]))
        end_frame = int(float(annotations[i]["end_frame"]))
        n_frames = end_frame - start_frame + 1
        sample = {
            "video_path": video_path,
            "segment": [start_frame, end_frame],
            "start_frame": start_frame,
            "end_frame": end_frame,
            "n_frames": n_frames,
            "video_id": i
        }
        sample["label"] = class_to_index[annotations[i]["label"]]
        sample["frame_indices"] = list(range(start_frame, end_frame + 1))
        dataset.append(sample)

    print(subset + " dataset loading over\n")
    return dataset, index_to_class


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None, 
                 random_shift=True, test_mode=False,
                 remove_missing=False, 
                 multi_clip_test=False,
                 dense_sample=False, twice_sample=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.remove_missing = remove_missing
        self.multi_clip_test = multi_clip_test
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.twice_sample = twice_sample  # twice sample for more validation
        if self.dense_sample:
            print('=> Using dense sample for the dataset...')
        if self.twice_sample:
            print('=> Using twice sample for the dataset...')

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        # self._parse_list()
        self.data, self.class_names = make_dataset(root_path, list_file)

        # 训练水平翻转：镜像会左右颠倒（视觉上左手变右手），缓解左右手样本不均衡。
        # 关键：只有能自动识别出"向左/向右"配对时才翻转，否则镜像后的方向类会
        # 携带错误标签（静默错误监督），宁可不翻。
        self._lr_pair = self._find_lr_pair()
        if self.random_shift and not self.test_mode:
            if self._lr_pair is None:
                print('WARNING: 无法自动识别"左右方向"类别配对，训练将【不做水平翻转】。')
                print('  class_names =', self.class_names)
                print('  如需翻转，请在 _find_lr_pair 中补充对真实类名的识别规则。')
            else:
                n0 = self.class_names.get(self._lr_pair[0], self._lr_pair[0])
                n1 = self.class_names.get(self._lr_pair[1], self._lr_pair[1])
                print('=> 训练水平翻转已启用：翻转时会把 类别{}[{}] <-> 类别{}[{}] 的标签对调。'
                      .format(self._lr_pair[0], n0, self._lr_pair[1], n1))
        self._do_hflip = self.random_shift and not self.test_mode and self._lr_pair is not None

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            try:
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
            except Exception:
                print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]
        # elif self.modality == 'Flow':
        #     if self.image_tmpl == 'flow_{}_{:05d}.jpg':  # ucf
        #         x_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('x', idx))).convert(
        #             'L')
        #         y_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('y', idx))).convert(
        #             'L')
        #     elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':  # something v1 flow
        #         x_img = Image.open(os.path.join(self.root_path, '{:06d}'.format(int(directory)), self.image_tmpl.
        #                                         format(int(directory), 'x', idx))).convert('L')
        #         y_img = Image.open(os.path.join(self.root_path, '{:06d}'.format(int(directory)), self.image_tmpl.
        #                                         format(int(directory), 'y', idx))).convert('L')
        #     else:
        #         try:
        #             # idx_skip = 1 + (idx-1)*5
        #             flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert(
        #                 'RGB')
        #         except Exception:
        #             print('error loading flow file:',
        #                   os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
        #             flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')
        #         # the input flow file is RGB image with (flow_x, flow_y, blank) for each channel
        #         flow_x, flow_y, _ = flow.split()
        #         x_img = flow_x.convert('L')
        #         y_img = flow_y.convert('L')

        #     return [x_img, y_img]

    # def _parse_list(self):
        # check the frame number is large >3:
        # tmp = [x.strip().split(' ') for x in open(self.list_file)]
        # if not self.test_mode or self.remove_missing:
        #     tmp = [item for item in tmp if int(item[1]) >= 3]
        # self.video_list = [VideoRecord(item) for item in tmp]

        # if self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
        #     for v in self.video_list:
        #         v._data[1] = int(v._data[1]) / 2
        # print('video number:%d' % (len(self.video_list)))

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record['n_frames'] - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record['n_frames'] for idx in range(self.num_segments)]
            return np.array(offsets) + record['start_frame']
        else:  # normal sample
            average_duration = (record['n_frames'] - self.new_length + 1) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                                  size=self.num_segments)
            elif record['n_frames'] > self.num_segments:
                offsets = np.sort(randint(record['n_frames'] - self.new_length + 1, size=self.num_segments))
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + record['start_frame']

    def _get_val_indices(self, record):
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record['n_frames'] - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record['n_frames'] for idx in range(self.num_segments)]
            return np.array(offsets) + record['start_frame']
        else:
            if record['n_frames'] > self.num_segments + self.new_length - 1:
                tick = (record['n_frames'] - self.new_length + 1) / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + record['start_frame']

    def _get_test_indices(self, record):
        if self.dense_sample:
            sample_pos = max(1, 1 + record['n_frames'] - 64)
            t_stride = 64 // self.num_segments
            start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) % record['n_frames'] for idx in range(self.num_segments)]
            return np.array(offsets) + record['start_frame']
        elif self.twice_sample:
            tick = (record['n_frames'] - self.new_length + 1) / float(self.num_segments)

            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)] +
                               [int(tick * x) for x in range(self.num_segments)])

            return offsets + record['start_frame']
        else:
            tick = (record['n_frames'] - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            return offsets + record['start_frame']

    def __getitem__(self, index):
        # record = self.video_list[index]
        record = self.data[index]
        # check this is a legit video folder

        # if self.image_tmpl == 'flow_{}_{:05d}.jpg':
        #     file_name = self.image_tmpl.format('x', 1)
        #     full_path = os.path.join(self.root_path, record.path, file_name)
        # elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
        #     file_name = self.image_tmpl.format(int(record.path), 'x', 1)
        #     full_path = os.path.join(self.root_path, '{:06d}'.format(int(record.path)), file_name)
        # else:
        #     file_name = self.image_tmpl.format(1)
        #     full_path = os.path.join(self.root_path, record.path, file_name)

        # while not os.path.exists(full_path):
        #     print('################## Not Found:', os.path.join(self.root_path, record.path, file_name))
        #     index = np.random.randint(len(self.video_list))
        #     record = self.video_list[index]
        #     if self.image_tmpl == 'flow_{}_{:05d}.jpg':
        #         file_name = self.image_tmpl.format('x', 1)
        #         full_path = os.path.join(self.root_path, record.path, file_name)
        #     elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
        #         file_name = self.image_tmpl.format(int(record.path), 'x', 1)
        #         full_path = os.path.join(self.root_path, '{:06d}'.format(int(record.path)), file_name)
        #     else:
        #         file_name = self.image_tmpl.format(1)
        #         full_path = os.path.join(self.root_path, record.path, file_name)

        if not self.test_mode:  # training or validation set
            if self.random_shift:  # training set
                segment_indices = self._sample_indices(record)
            else:  # validation set
                segment_indices = self._get_val_indices(record)
        else:  # test set
            # for mulitple clip test, use random sampling;
            # for single clip test, use middle sampling
            if self.multi_clip_test:
                # segment_indices = self._sample_indices(record)
                n = ((record['n_frames'] + self.num_segments - 1) // self.num_segments) * self.num_segments    
                average_duration = n // self.num_segments    
                offsets_list = []
                for _ in range (10):                                                 
                    offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments) 
                    offsets_list.append(offsets)
                offsets = np.concatenate(offsets_list)
                segment_indices = offsets * record['n_frames'] // n + record['start_frame']
            else:
                segment_indices = self._get_test_indices(record)
        # if not self.test_mode:
        #     segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        # else:
        #     segment_indices = self._get_test_indices(record)
        return self.get(record, segment_indices)

    def _find_lr_pair(self):
        """自动识别'向左/向右'两个类别索引（供翻转时对调），找不到返回 None。
        支持类名为 G05/G06 编码或含 left/right 的命名。"""
        names = self.class_names  # {index: label_name}
        if not names:
            return None
        idxs = []
        for i, nm in names.items():
            low = (nm or '').lower()
            if 'left' in low or 'right' in low or low in ('g05', 'g06'):
                idxs.append(i)
        return (idxs[0], idxs[1]) if len(idxs) == 2 else None

    def get(self, record, indices):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):        # new_length = 1
                seg_imgs = self._load_image(record['video_path'], p)
                images.extend(seg_imgs)
                if p < record['end_frame']:
                    p += 1

        process_data = self.transform(images)
        label = record['label']

        # 训练水平翻转：整段 8 帧一起镜像，方向类标签对调
        if self._do_hflip and random.random() < 0.5:
            process_data = torch.flip(process_data, dims=[-1])
            if self._lr_pair is not None and label in self._lr_pair:
                label = self._lr_pair[1] if label == self._lr_pair[0] else self._lr_pair[0]

        return process_data, label

    def __len__(self):
        return len(self.data)

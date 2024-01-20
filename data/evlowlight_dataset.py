from os import path as osp
from pathlib import Path

import torch
from basicsr.data.data_util import generate_frame_indices
from basicsr.utils import get_root_logger, scandir, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from data.data_util import (
    write_voxel_seq, read_voxel_seq, read_img_seq, voxel2tensor
)


@DATASET_REGISTRY.register()
class EventVideoTestDataset(data.Dataset):
    def __init__(self, opt):
        super(EventVideoTestDataset, self).__init__()
        self.opt = opt
        self.data_info = {
            'all_folder': [], 'folder': [], 'idx': [], 'idx_num': [], 'start_idx': [], 'border': [],
        }
        data_info = {
            'all_idx_num': [], 'all_shape': [], 'all_start_idx': [],
            'all_folder': [], 'folder': [], 'idx': [], 'idx_num': [], 'start_idx': [], 'border': [],
        }
        self.data_keys = opt['data_keys']
        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)
        self.scale = opt['scale']
        self.img_load_size = opt.get('load_size', None)  # (640, 480), (692, 520), (1920, 1080)
        self.ev_load_size = [i // self.scale for i in self.img_load_size] if self.img_load_size else None
        self.real_ev_size = opt.get('real_ev_size', None)
        self.filename_tmpl = opt.get('filename_tmpl', "05d")  # "{:05d}.jpg"
        self.filename_tmpl_ev = opt.get('filename_tmpl_ev', "output.h5")  # "{:05d}.jpg"
        self.num_frame = opt.get('num_frame', 3)
        self.mod_size = 1  # processing of size is asigned to the model

        logger = get_root_logger()
        self.center_frame_only = opt.get('center_frame_only', False)

        if 'meta_info_file' in opt:
            with open(opt['meta_info_file'], 'r') as fin:
                for line in fin:
                    folder, idx_num, shape, start_idx = line.split(' ')
                    idx_num = opt.get('max_frame', None) or int(idx_num)
                    start_idx = opt.get('start_idx', None) or int(start_idx)
                    data_info['all_folder'].append(folder)
                    data_info['all_idx_num'].append(idx_num)
                    data_info['all_shape'].append([int(i) for i in shape[1:-1].split(',')])
                    data_info['all_start_idx'].append(start_idx)

                    data_info['folder'].extend([folder for i in range(idx_num)])
                    data_info['idx'].extend(
                        [f'{i}/{start_idx + idx_num}' for i in range(start_idx, start_idx + idx_num)])
                    data_info['idx_num'].extend([idx_num for i in range(idx_num)])
                    data_info['start_idx'].extend([start_idx for i in range(idx_num)])
                    border_l = [0] * idx_num
                    for i in range(self.opt['num_frame'] // 2):
                        border_l[i] = 1
                        border_l[idx_num - i - 1] = 1
                    data_info['border'].extend(border_l)

        for client_key in self.data_keys:
            root = Path(opt.get(f"dataroot_{client_key}"))
            setattr(self, f"{client_key}_root", root)

        self.ev_file_ext = opt.get('ev_file_ext', '.npy')
        self.data_info.update(**{'lq_path': [], 'gt_path': [], 'ev_path': [], })

        logger.info(f'Generate data info for VideoTestDataset - {opt["name"]}')
        if len(data_info['all_folder']) == 0:
            data_info['all_folder'] = [p.stem for p in sorted(list(Path(self.lq_root).iterdir()))]

        self.imgs_lq, self.imgs_gt, self.imgs_ev = {}, {}, {}
        for f_idx, folder in enumerate(data_info['all_folder']):
            # get frame list for lq and gt
            img_paths_lq = sorted(list(scandir(osp.join(self.lq_root, folder), full_path=True)))
            if 'gt' in self.data_keys:
                img_paths_gt = sorted(list(scandir(osp.join(self.gt_root, folder), full_path=True)))
                assert len(img_paths_lq) == len(img_paths_gt), (
                    f'Different number of images in lq ({len(img_paths_lq)})'
                    f' and gt folders ({len(img_paths_gt)})')
            if 'meta_info_file' in opt:
                start_idx = data_info['all_start_idx'][f_idx]
                idx_num = data_info['all_idx_num'][f_idx]
                shape = data_info['all_shape'][f_idx]
                real_ev_size = (shape[1], shape[0])
            else:
                start_idx = opt.get('start_idx', 0)
                idx_num = opt.get('max_frame', len(img_paths_lq))
                real_ev_size = self.real_ev_size

            iterate_index = [int(i) for i in range(start_idx, start_idx + idx_num)]
            iterate_index_ev = [int(i) for i in range(start_idx * 2, (start_idx + idx_num - 1) * 2)]

            try:
                img_paths_ev = sorted(list(
                    scandir(osp.join(self.ev_root, folder), full_path=True, suffix=self.ev_file_ext)))
                if not len(img_paths_ev) == len(iterate_index_ev):
                    print(f"find {len(img_paths_ev)} but we needs {len(iterate_index_ev)} events, reindexing ... ")
                    img_paths_ev = [img_paths_ev[i] for i in iterate_index_ev]
            except Exception:
                logger.info(f"Creating event dataset in {self.ev_root}")
                write_voxel_seq(Path(self.ev_root).parent / 'events', folder,
                                new_dir=Path(self.ev_root).stem,
                                filename_tmpl=self.filename_tmpl_ev,
                                image_height=real_ev_size[1] if real_ev_size else None,
                                image_width=real_ev_size[0] if real_ev_size else None,
                                iterate_index=iterate_index,
                                )
                img_paths_ev = sorted(list(
                    scandir(osp.join(self.ev_root, folder), full_path=True, suffix=self.ev_file_ext)))
                assert len(iterate_index_ev) == len(img_paths_ev)

            img_paths_lq = [img_paths_lq[i] for i in iterate_index]
            if 'gt' in self.data_keys:
                img_paths_gt = [img_paths_gt[i] for i in iterate_index]

            # [************************** Found data *******************************]
            start_idx = 0
            idx_num = len(img_paths_lq)
            self.data_info['all_folder'].append(folder)
            self.data_info['start_idx'].extend([start_idx for i in range(start_idx, start_idx + idx_num)])
            self.data_info['idx_num'].extend([idx_num for i in range(start_idx, start_idx + idx_num)])
            self.data_info['folder'].extend([folder] * idx_num)

            for i in range(start_idx, start_idx + idx_num):
                self.data_info['idx'].append(f'{i}/{start_idx + idx_num}')
            border_l = [0] * idx_num
            for i in range(self.opt['num_frame'] // 2):
                border_l[i] = 1
                border_l[idx_num - i - 1] = 1
            self.data_info['border'].extend(border_l)

            self.data_info['lq_path'].extend(img_paths_lq)
            if 'gt' in self.data_keys:
                self.data_info['gt_path'].extend(img_paths_gt)
            self.data_info['ev_path'].extend(img_paths_ev)
            # cache data or save the frame list
            self.imgs_lq[folder] = img_paths_lq
            if 'gt' in self.data_keys:
                self.imgs_gt[folder] = img_paths_gt
            self.imgs_ev[folder] = img_paths_ev

    def _transform(self, data):
        for key in self.data_keys:
            if key.lower().startswith('ev'):
                data[key] = voxel2tensor(data[key])
                data[key] = torch.stack(data[key], dim=0)
            else:
                data[key] = img2tensor(data[key])
                data[key] = torch.stack(data[key], dim=0)
                if self.mean is not None or self.std is not None:
                    data[key] = normalize(data[key], self.mean, self.std, inplace=True)
        return data

    @staticmethod
    def generate_event_indices(crt_idx, max_frame_num, num_frames, padding='reflection'):
        assert padding in ('replicate', 'reflection', 'reflection_circle', 'circle'), f'Wrong padding mode: {padding}.'

        max_frame_num = (max_frame_num - 1) * 2 - 1  # start from 0
        num_pad = num_frames - 1
        crt_idx = crt_idx * 2

        indices = []
        for i in range(crt_idx - num_pad, crt_idx + num_pad):
            if i < 0:
                if padding == 'replicate':
                    pad_idx = 0
                elif padding == 'reflection':
                    pad_idx = -(i + 1)
                elif padding == 'reflection_circle':
                    pad_idx = crt_idx + num_pad - (i + 1)
                else:
                    pad_idx = num_pad * 2 + i
            elif i > max_frame_num:
                if padding == 'replicate':
                    pad_idx = max_frame_num
                elif padding == 'reflection':
                    pad_idx = max_frame_num * 2 - (i - 1)
                elif padding == 'reflection_circle':
                    pad_idx = (crt_idx - num_pad) - (i - max_frame_num)
                else:
                    pad_idx = i - num_pad * 2
            else:
                pad_idx = i
            indices.append(pad_idx)
        return indices

    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        idx = int(self.data_info['idx'][index].split('/')[0])
        idx_num = int(self.data_info['idx_num'][index])
        lq_path = self.data_info['lq_path'][index]

        data = {
            'folder': folder,  # folder name
            'idx': f'{idx}/{idx_num}',  # e.g., 0/99
            'border': self.data_info['border'][index],  # 1 for border, 0 for non-border
            'lq_path': lq_path  # center frame
        }

        select_idx = generate_frame_indices(idx, idx_num, self.opt['num_frame'], padding=self.opt['padding'])
        select_idx_ev = self.generate_event_indices(idx, idx_num, self.opt['num_frame'],
                                                    padding=self.opt['padding'])
        if 'lq' in self.data_keys:
            data['lq'] = read_img_seq([self.imgs_lq[folder][i] for i in select_idx], load_size=self.img_load_size)
        if 'gt' in self.data_keys:
            data['gt'] = read_img_seq([self.imgs_gt[folder][idx]], load_size=self.img_load_size)
        data['ev'] = read_voxel_seq([self.imgs_ev[folder][i] for i in select_idx_ev], load_size=self.ev_load_size)

        data = self._transform(data)
        if 'gt' in self.data_keys:
            data['gt'] = data['gt'][0]  # (c, h, w)
        return data

    def __len__(self):
        return len(self.data_info['idx'])


@DATASET_REGISTRY.register()
class EventVideoRecurrentTestDataset(EventVideoTestDataset):
    def __getitem__(self, index):
        if self.center_frame_only:
            return super(EventVideoRecurrentTestDataset, self).__getitem__(index)
        else:
            folder = self.data_info['all_folder'][index]
            data = {'folder': folder, }
            if 'lq' in self.data_keys:
                data['lq'] = read_img_seq(self.imgs_lq[folder], mod_size=self.mod_size, load_size=self.img_load_size)
            if 'gt' in self.data_keys:
                data['gt'] = read_img_seq(self.imgs_gt[folder], mod_size=self.mod_size, load_size=self.img_load_size)
            data['ev'] = read_voxel_seq(self.imgs_ev[folder], mod_size=self.mod_size // self.scale,
                                        suffix=self.ev_file_ext, load_size=self.ev_load_size)
            data = self._transform(data)
            return data

    def __len__(self):
        if self.center_frame_only:
            return super(EventVideoRecurrentTestDataset, self).__len__()
        else:
            return len(self.data_info['all_folder'])

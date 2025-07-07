# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm

from torch.utils.data._utils.collate import default_collate
from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
from data_loaders.humanml.utils.get_opt import get_opt
from data_loaders.humanml.utils.paramUtil import *
from data_loaders.humanml.utils.data_utils import OBJECT_LIST, OBJECT_LIST_NEW, OBJECT_NEW2ORIGINAL_DICT

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)

    return default_collate(batch)


class Text2MotionDatasetV2(data.Dataset):
    """
    Args:
        std_multiplier: multiply the std by this value; maybe useful for diffusion models by keeping the range of data managable
    """
    def __init__(self,
                 opt,
                 mean,
                 std,
                 split_file,
                 w_vectorizer,
                 use_rand_proj=False,
                 proj_matrix_dir=None,
                 traject_only=False,
                 mode='train',
                 random_proj_scale=10.0,
                 augment_type='none',
                 std_scale_shift=(1., 0.),  # Test random projection
                 drop_redundant=False,
                 proj_matrix_name='',
                 hands_only=False,
                 motion_enc_frames=0):
        self.opt = opt
        self.w_vectorizer = w_vectorizer

        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 20 if not hands_only else 8

        self.use_rand_proj = use_rand_proj
        self.traject_only = traject_only
        self.hands_only = hands_only

        self.mode = mode
        self.bps = np.load(pjoin(opt.data_root, 'bps_enc.npy'),allow_pickle=True)
        self.bps_mirrored = np.load(pjoin(opt.data_root, 'bps_enc_mirrored.npy'),allow_pickle=True)

        self.augment_type = augment_type
        self.motion_enc_frames = motion_enc_frames

        self.std_scale_shift = std_scale_shift
        self.drop_redundant = drop_redundant

        data_dict = {}
        id_list = []

        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        id_dict = {}
        with open('dataset/file_names.txt', 'r') as file:
                # Read each line in the file
                for line in file:
                    # Strip any leading/trailing whitespace
                    line = line.strip()

                    # Split the line at the comma
                    key, value = line.split(',', 1)

                    # Add the key-value pair to the dictionary
                    id_dict[key] = value


        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len:
                   continue
                if (len(motion) >= 200):
                    motion = motion[:200]
                text_data = []
                flag = False
                mirrored = 'M' in name
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * 20):int(to_tag *
                                                                      20)]
                                if (len(motion)) < min_motion_len:
                                    continue
                                new_name = random.choice(
                                    'ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice(
                                        'ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {
                                    'motion': n_motion,
                                    'length': len(n_motion),
                                    'text': [text_dict]
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag,
                                      to_tag, name)

                if flag:
                    data_dict[name] = {
                        'motion': motion,
                        'length': len(motion),
                        'text': text_data,
                        'mirrored': mirrored,
                        'id': id_dict[name],
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass


        name_list = new_name_list
        self.max_motion_length = max(length_list) 
        self.mean = mean
        self.std = std

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list

        self.max_length = 20 if not hands_only else min(length_list)
        self.reset_max_len(self.max_length)


        if self.traject_only:
            if self.hands_only:
                self.traj_only_idcs = np.concatenate([np.arange(0,108)])
            else:
                self.traj_only_idcs = np.concatenate([np.arange(0,117)])

        self.non_redundant_idcs = np.concatenate([np.arange(0, 3), np.arange(63, 66), np.arange(126, 186), np.arange(motion.shape[1]-15, motion.shape[1])])

        if use_rand_proj:
            self.init_random_projection(proj_matrix_dir,
                                        scale=random_proj_scale,
                                        proj_matrix_name=proj_matrix_name)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)

        self.max_length = length

    def get_std_mean(self, traject_only=None, drop_redundant=None):

        if traject_only is None:
            traject_only = self.traject_only
        if drop_redundant is None:
            drop_redundant = self.drop_redundant

        if traject_only:
            std = self.std[self.traj_only_idcs]
            mean = self.mean[self.traj_only_idcs]
        elif drop_redundant:
            std = self.std[self.non_redundant_idcs]
            mean = self.mean[self.non_redundant_idcs]
        else:
            std = self.std
            mean = self.mean

        std = std * self.std_scale_shift[0] + self.std_scale_shift[1]
        return std, mean

    def inv_transform(self, data, traject_only=None):
        if self.use_rand_proj:
            data = self.inv_random_projection(data)
        std, mean = self.get_std_mean(traject_only)
        return data * std + mean

    def inv_transform_th(self, data, traject_only=None, use_rand_proj=None):
        use_rand_proj = self.use_rand_proj if use_rand_proj is None else use_rand_proj
        if use_rand_proj:
            data = self.inv_random_projection(data, mode="th")
        std, mean = self.get_std_mean(traject_only)
        return data * torch.from_numpy(std).to(
            data.device).to(data.dtype) + torch.from_numpy(mean).to(data.device).to(data.dtype)

    def transform_th(self, data, traject_only=None, use_rand_proj=None):
        std, mean = self.get_std_mean(traject_only)

        data = (data - torch.from_numpy(mean).to(
            data.device).to(data.dtype)) / torch.from_numpy(std).to(data.device).to(data.dtype)
        use_rand_proj = self.use_rand_proj if use_rand_proj is None else use_rand_proj
        if use_rand_proj:
            data = self.random_projection(data, mode="th")
        return data

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):

        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list, mirrored, data_id = data['motion'], data['length'], data[
            'text'], data['mirrored'], data['id']
        # Randomly select a caption

        if len(text_list) > 1:
            text_data = text_list[1]
        else:
            text_data = text_list[0]
        #text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) == 1:
            sent_len = len(tokens)
            pass
        elif len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'
                               ] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []

        if not len(tokens) == 1:
            for token in tokens:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots.append(pos_oh[None, :])
                word_embeddings.append(word_emb[None, :])
            pos_one_hots = np.concatenate(pos_one_hots, axis=0)
            word_embeddings = np.concatenate(word_embeddings, axis=0)

        # NOTE: if used for training trajectory model, discard all but the first 4 values
        motion_full = motion.copy()
        if self.traject_only:
            motion = motion[:, self.traj_only_idcs]

        if self.drop_redundant:
            motion = motion[:, self.non_redundant_idcs]

        "Z Normalization"
        std, mean = self.get_std_mean()
        motion = (motion - mean) / std

        # Projection
        # NOTE: Do not do random projection if mode is eval or gt
        if (not self.mode in ["eval", "gt"]) and self.use_rand_proj:
            # t x 263
            motion = self.random_projection(motion)
        if m_length < self.max_motion_length:
            motion = np.concatenate([
                motion,
                np.tile(motion[m_length-1],(self.max_motion_length - m_length,1))
            ],axis=0)

        obj_name = [s for s in OBJECT_LIST_NEW if s in caption]

        if mirrored:
            obj_bps = self.bps_mirrored.item()[OBJECT_NEW2ORIGINAL_DICT[obj_name[0]]]
        else:
            obj_bps = self.bps.item()[OBJECT_NEW2ORIGINAL_DICT[obj_name[0]]]


        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(
            tokens), obj_bps, self.motion_enc_frames, motion_full, data_id

    def init_random_projection(self, save_at, scale: float, proj_matrix_name=''):

        assert proj_matrix_name != '', "Please provide a name for the projection matrix"

        rand_proj_file = proj_matrix_name
        inv_rand_proj_file = "inv_" + proj_matrix_name

        if os.path.isfile(os.path.join(save_at, rand_proj_file)):
            print(f"Loading random projection matrix from {save_at}")
            self.proj_matrix = np.load(os.path.join(save_at, rand_proj_file))
            self.inv_proj_matrix = np.load(
                os.path.join(save_at, inv_rand_proj_file))

            self.proj_matrix_th = torch.from_numpy(self.proj_matrix)
            self.inv_proj_matrix_th = torch.from_numpy(self.inv_proj_matrix)

            if self.traject_only:
                self.proj_matrix = self.proj_matrix[self.traj_only_idcs][:,self.traj_only_idcs]
                self.inv_proj_matrix = self.inv_proj_matrix[self.traj_only_idcs][:,self.traj_only_idcs]
                self.proj_matrix_th = self.proj_matrix_th[self.traj_only_idcs][:,self.traj_only_idcs]
                self.inv_proj_matrix_th = self.inv_proj_matrix_th[self.traj_only_idcs][:,self.traj_only_idcs]
        else:
            print(f"Creating random projection matrix {scale}")

            self.proj_matrix = torch.normal(
            mean=0, std=1.0, size=(117, 117),
            dtype=torch.float)
            ### Change here, scale wrists global translation and rotation and object poses
            ### The following features are emphasized:
            ### Global wrist poses 0-5
            ### Global wrist 6D orientations 30-35 and 60-65
            ### Object 3D position and 6D orientation 108-116
            self.proj_matrix[[0, 1, 2, 3, 4, 5, 30, 31, 32, 33, 34, 35, 60, 61, 62, 63, 64, 65, 108, 109, 110, 111, 112, 113, 114, 115, 116], :] *= scale
            self.proj_matrix = self.proj_matrix / np.sqrt(117 - 27 + 27 * scale**2)

            self.inv_proj_matrix = torch.inverse(self.proj_matrix)

            self.proj_matrix = self.proj_matrix.detach().cpu().numpy()
            self.inv_proj_matrix = self.inv_proj_matrix.detach().cpu().numpy()

            self.proj_matrix_th = torch.from_numpy(self.proj_matrix)
            self.inv_proj_matrix_th = torch.from_numpy(self.inv_proj_matrix)

            np.save(os.path.join(save_at, rand_proj_file), self.proj_matrix)
            np.save(os.path.join(save_at, inv_rand_proj_file),self.inv_proj_matrix)

    def random_projection(self, motion, mode="np"):

        if mode == "th":
            return torch.matmul(motion, self.proj_matrix_th.to(motion.device))

        return np.matmul(motion, self.proj_matrix)

    def inv_random_projection(self, data, mode="np"):
        if mode == "th":
            return torch.matmul(data, self.inv_proj_matrix_th.to(data.device))
        return np.matmul(data, self.inv_proj_matrix)


class HumanML3D(data.Dataset):
    def __init__(self,
                 mode,
                 datapath='./dataset/humanml_opt.txt',
                 split ="train",
                 split_set = "objects_unseen",
                 use_abs3d=False,
                 traject_only=False,
                 use_random_projection=False,
                 random_projection_scale=None,
                 augment_type='none',
                 std_scale_shift=(1., 0.),
                 drop_redundant=False,
                 num_frames=None,
                 data_repr='',
                 mean_name='',
                 std_name='',
                 proj_matrix_name='',
                 hands_only = False,
                 text_detailed = False,
                 motion_enc_frames=0,
                 **kwargs):
        self.mode = mode

        self.dataset_name = 't2m'
        self.dataname = 't2m'

        abs_base_path = '.'
        dataset_opt_path = pjoin(abs_base_path, datapath)
        device = None

        opt = get_opt(dataset_opt_path, device, mode, data_repr, use_abs3d=use_abs3d, max_motion_length=num_frames,
                        hands_only=hands_only, text_detailed=text_detailed)
        opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)

        opt.text_dir = pjoin(abs_base_path, opt.text_dir)
        opt.model_dir = pjoin(abs_base_path, opt.model_dir)
        opt.checkpoints_dir = pjoin(abs_base_path, opt.checkpoints_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        opt.save_root = pjoin(abs_base_path, opt.save_root)
        opt.meta_dir = './dataset'
        self.opt = opt
        print('Loading dataset %s ...' % opt.dataset_name)

        self.absolute_3d = use_abs3d
        self.traject_only = traject_only
        self.use_rand_proj = use_random_projection
        self.random_proj_scale = random_projection_scale
        self.augment_type = augment_type
        self.std_scale_shift = std_scale_shift
        self.drop_redundant = drop_redundant

        if self.use_rand_proj:
            if self.random_proj_scale == 10:
                # NOTE: legacy code
                proj_matrix_dir = "./dataset"
            else:
                proj_matrix_dir = os.path.join(
                    f'save/random_proj_{self.random_proj_scale:.0f}')
                os.makedirs(proj_matrix_dir, exist_ok=True)
            print(f'proj_matrix_dir = {proj_matrix_dir}')
        else:
            proj_matrix_dir = None


        self.mean = np.load(pjoin(opt.data_root, mean_name))
        self.std = np.load(pjoin(opt.data_root, std_name))

        self.split_file = pjoin(opt.data_root, f'{split}_{split_set}.txt')

      
        self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, 'glove'),
                                            'our_vab')

        print(
            f't2m dataset aug: {self.augment_type} std_scale_shift: {self.std_scale_shift}'
        )
        print(f't2m dataset drop redundant information: {self.drop_redundant}')
        self.t2m_dataset = Text2MotionDatasetV2(
            self.opt,
            self.mean,
            self.std,
            self.split_file,
            self.w_vectorizer,
            use_rand_proj=self.use_rand_proj,
            proj_matrix_dir=proj_matrix_dir,
            traject_only=self.traject_only,
            mode=mode,
            random_proj_scale=self.random_proj_scale,
            augment_type=self.augment_type,
            std_scale_shift=self.std_scale_shift,
            drop_redundant=self.drop_redundant,
            proj_matrix_name=proj_matrix_name,
            hands_only=hands_only,
            motion_enc_frames=motion_enc_frames)
        # End test
        self.num_actions = 1  # dummy placeholder

        assert len(self.t2m_dataset) > 1, 'You loaded an empty dataset, ' \
                                          'it is probably because your data dir has only texts and no motions.\n' \
                                          'To train and evaluate MDM you should get the FULL data as described ' \
                                          'in the README file.'


    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return self.t2m_dataset.__len__()

# A wrapper class for t2m original dataset for MDM purposes
class GRAB(HumanML3D):
    def __init__(self,
                mode,
                datapath='./dataset/grab_opt_objects.txt',
                split="train",
                **kwargs):
        super(GRAB, self).__init__(mode, datapath, split, **kwargs)

# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
from argparse import Namespace
import re
from os.path import join as pjoin
from data_loaders.humanml.utils.word_vectorizer import POS_enumerator


def is_float(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')    # 去除正数(+)、负数(-)符号
    try:
        reg = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')
        res = reg.match(str(numStr))
        if res:
            flag = True
    except Exception as ex:
        print("is_float() - error: " + str(ex))
    return flag


def is_number(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')    # 去除正数(+)、负数(-)符号
    if str(numStr).isdigit():
        flag = True
    return flag


def get_opt(opt_path, device, mode, data_repr, max_motion_length, use_abs3d=False,
            use_contacts=False, hands_only=False, text_detailed=False):
    opt = Namespace()
    opt_dict = vars(opt)
    skip = ('-------------- End ----------------',
            '------------ Options -------------',
            '\n')
    print('Reading', opt_path)
    with open(opt_path) as f:
        for line in f:
            if line.strip() not in skip:
                # print(line.strip())
                key, value = line.strip().split(': ')
                if value in ('True', 'False'):
                    opt_dict[key] = bool(value)
                elif is_float(value):
                    opt_dict[key] = float(value)
                elif is_number(value):
                    opt_dict[key] = int(value)
                else:
                    opt_dict[key] = str(value)

    # print(opt)
    opt_dict['which_epoch'] = 'latest'
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')

    if opt.dataset_name == 't2m':
        opt.data_root = './dataset/HumanML3D'

        if use_abs3d and mode not in ['eval', 'gt']:
            # Will load the original dataset (relative) if in 'eval' or 'gt' mode
            opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs_abs_3d')
        else:
            opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')

        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        opt.dim_pose = 263
        # NOTE: UNET needs to uses multiples of 16
        opt.max_motion_length = max_motion_length
        print(f'WARNING: max_motion_length is set to {max_motion_length}')
    elif opt.dataset_name == 'kit':
        raise NotImplementedError()
    elif opt.dataset_name == 'grab':
        opt.data_root = './dataset/GRAB_HANDS'
        opt.motion_dir = pjoin(opt.data_root, data_repr)
        opt.joints_num = 42 
        if hands_only:
            opt.dim_pose = 108
            opt.text_dir = pjoin(opt.data_root, 'texts_grasp')
        else:
            opt.dim_pose = 117
            if text_detailed:
                opt.text_dir = pjoin(opt.data_root, 'texts_detailed')
            else:
                opt.text_dir = pjoin(opt.data_root, 'texts_simple')
        if use_contacts:
            opt.dim_pose += 42
        opt.max_motion_length = 196
    else:
        raise KeyError('Dataset not recognized')

    opt.dim_word = 300
    opt.num_classes = 200 // opt.unit_length
    opt.dim_pos_ohot = len(POS_enumerator)
    opt.is_train = False
    opt.is_continue = False
    opt.device = device

    return opt

# Copyright (c) Meta Platforms, Inc. and affiliates.

from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate
from typing import Tuple
from dataclasses import dataclass


def get_dataset_class(name):
    from data_loaders.humanml.data.dataset import GRAB
    return GRAB


def get_collate_fn(name, hml_mode='train'):
    if hml_mode == 'gt':
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name in ["grab"]:
        return t2m_collate
    else:
        return all_collate


@dataclass
class DatasetConfig:
    name: str
    batch_size: int
    num_frames: int
    split: str = 'train'
    split_set: str = 'objects_unseen'
    hml_mode: str = 'train'
    use_abs3d: bool = False
    traject_only: bool = False
    use_random_projection: bool = False
    random_projection_scale: float = None
    augment_type: str = 'none'
    std_scale_shift: Tuple[float] = (1.0, 0.0)
    drop_redundant: bool = False
    use_pca: bool = False
    pre_grasp: bool = False
    hands_only: bool = False
    obj_only: bool = False
    use_contacts: bool = False
    obj_enc: bool = False
    motion_enc: bool = False
    motion_enc_frames: int = 0
    text_detailed: bool = False
    data_repr: str = ''
    mean_name: str = ''
    std_name: str = ''
    proj_matrix_name: str = ''


def get_dataset(conf: DatasetConfig):
    DATA = get_dataset_class(conf.name)

    if conf.name in ["grab"]:
        dataset = DATA(split=conf.split,
                       split_set=conf.split_set,
                       num_frames=conf.num_frames,
                       mode=conf.hml_mode,
                       use_abs3d=conf.use_abs3d,
                       traject_only=conf.traject_only,
                       use_random_projection=conf.use_random_projection,
                       random_projection_scale=conf.random_projection_scale,
                       augment_type=conf.augment_type,
                       std_scale_shift=conf.std_scale_shift,
                       drop_redundant=conf.drop_redundant,
                       use_pca=conf.use_pca,
                       pre_grasp = conf.pre_grasp,
                       hands_only = conf.hands_only,
                       obj_only = conf.obj_only,
                       use_contacts=conf.use_contacts,
                       data_repr=conf.data_repr,
                       mean_name=conf.mean_name,
                       std_name=conf.std_name,
                       proj_matrix_name=conf.proj_matrix_name,
                       motion_enc_frames=conf.motion_enc_frames,
                       text_detailed=conf.text_detailed,
                       )
    else:
        raise NotImplementedError()
    return dataset


def get_dataset_loader(conf: DatasetConfig, shuffle=True):
    dataset = get_dataset(conf)
    collate = get_collate_fn(conf.name, conf.hml_mode)

    # return dataset
    loader = DataLoader(dataset,
                        batch_size=conf.batch_size,
                        num_workers=1,
                        shuffle=shuffle,
                        drop_last=True,
                        collate_fn=collate)

    return loader

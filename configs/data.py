# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from utils.parser_util import BaseOptions, DataOptions

@dataclass
class grab_motion_rel(BaseOptions, DataOptions):
    dataset: str = 'grab'
    data_dir: str = ''
    abs_3d: bool = False

@dataclass
class grab_motion_rel_pca(BaseOptions, DataOptions):
    dataset: str = 'grab'
    data_dir: str = ''
    abs_3d: bool = False
    use_pca: bool = True

## GRASPING
@dataclass
class diffh2o_grasp(grab_motion_rel_pca):
    traj_only: bool = True
    pre_grasp: bool = True
    hands_only: bool = True
    use_random_proj: bool = True
    obj_enc: bool = True
    text_detailed: bool = False
    diffusion_steps: int = 1000
    motion_enc_frames: int = 1
    random_proj_scale: float = 10
    data_repr: str = 'diffh2o_representation_grasp'
    mean_name: str = 'Mean_diffh2o_grasp.npy'
    std_name: str = 'Std_diffh2o_grasp.npy'
    proj_matrix_name: str = 'rand_proj_diffh2o_grasp.npy'

## INTERACTION
@dataclass
class diffh2o_interaction(grab_motion_rel_pca):
    traj_only: bool = False
    hands_only: bool = False
    use_random_proj: bool = True
    random_proj_scale: float = 10
    obj_enc: bool = True
    motion_enc_frames: int = 0
    data_repr: str = 'diffh2o_representation_interaction'
    mean_name: str = 'Mean_diffh2o_interaction.npy'
    std_name: str = 'Std_diffh2o_interaction.npy'
    proj_matrix_name: str = 'rand_proj_diffh2o_interaction.npy'

### Full Simple Texts (GRASPING + INTERACTION)
@dataclass
class diffh2o_full(grab_motion_rel_pca):
    traj_only: bool = False
    hands_only: bool = False
    use_random_proj: bool = True
    random_proj_scale: float = 10
    obj_enc: bool = True
    motion_enc_frames: int = 0
    text_detailed: bool = False
    data_repr: str = 'diffh2o_representation_full'
    mean_name: str = 'Mean_diffh2o_full.npy'
    std_name: str = 'Std_diffh2o_full.npy'
    proj_matrix_name: str = 'rand_proj_diffh2o.npy'

### Full Detailed Texts (GRASPING + INTERACTION)
@dataclass
class diffh2o_full_detailed(grab_motion_rel_pca):
    traj_only: bool = False
    hands_only: bool = False
    use_random_proj: bool = True
    random_proj_scale: float = 10
    obj_enc: bool = True
    motion_enc_frames: int = 0
    text_detailed: bool = True
    data_repr: str = 'diffh2o_representation_full'
    mean_name: str = 'Mean_diffh2o_full.npy'
    std_name: str = 'Std_diffh2o_full.npy'
    proj_matrix_name: str = 'rand_proj_diffh2o.npy'

### MDM Baseline
@dataclass
class mdm_full(grab_motion_rel_pca):
    traj_only: bool = False
    hands_only: bool = False
    use_random_proj: bool = False
    random_proj_scale: float = 1
    obj_enc: bool = True
    motion_enc_frames: int = 0
    data_repr: str = 'diffh2o_representation_full'
    mean_name: str = 'Mean_diffh2o_full.npy'
    std_name: str = 'Std_diffh2o_full.npy'
    proj_matrix_name: str = ''

# Copyright (c) Meta Platforms, Inc. and affiliates.

# Thirdparty
from utils.parser_util import FullModelArgs


def get_template(args: FullModelArgs, guidance=False):
    # [no, trajectory, kps, sdf]
    if guidance:
        updated_args = guidance_template(args)
    else:
        updated_args = args
    return updated_args

def guidance_template(args: FullModelArgs):
    args.do_inpaint = True
    args.gen_two_stages = True
    # NOTE: set imputation p2p mode here
    # args.p2p_impute = False
    args.p2p_impute = True

    return args

def testing_template(args: FullModelArgs):
    args.do_inpaint = False # True
    args.guidance_mode = "no"  # ["no", "trajectory", "kps", "sdf"]
    # args.classifier_scale = 1.0
    args.gen_two_stages = False
    args.p2p_impute = False
    args.use_ddim = False # True
    # args.motion_length = 4.5
    args.interpolate_cond = False # True
    return args

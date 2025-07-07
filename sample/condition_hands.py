# Copyright (c) Meta Platforms, Inc. and affiliates.

# Thirdparty
import numpy as np
import torch
import torch.nn.functional as F


def get_grasp_references(gt_motions, sampled_keyframes, dataset, dataset_post, feat_dim_traj=117, feat_dim_motion=117, max_len=200):
    '''Output from this function is already match the batch_size and moved to the target device.
    Input:
        gt_motions: [bs, 1, features, sampled_keyframes] [10, 1, 459, 33]
    Return:
        target: [bs, max_length, 22, 3]
        target_mask: [bs, max_length, 22, 3]
    '''
    batch_size = gt_motions.shape[0]

    data_device = gt_motions.device

    target = torch.zeros([batch_size, max_len, 1, feat_dim_traj], device=data_device)
    target_mask = torch.zeros_like(target, dtype=torch.bool)

    traj_only_idcs = np.arange(0,feat_dim_traj)

    for idx in range(batch_size):
        key_posi = gt_motions[idx, 0, :, :].permute(1, 0)  # [keyframe_len, feature]
        for kframe_idx, kframe in enumerate(sampled_keyframes):
            target[idx, kframe, 0] = key_posi[kframe_idx,traj_only_idcs]
            target_mask[idx, kframe, 0] = True

    ### For inpainting ###
    # inpaint_motion shape [batch, feat_dim, 1, 200], same as model output
    inpaint_traj = torch.zeros([batch_size, feat_dim_traj, 1, max_len], device=data_device)
    inpaint_traj_mask = torch.zeros_like(inpaint_traj, dtype=torch.bool)
    # For second stage inpainting, we only use key locations as target instead of the interpolated lines
    inpaint_traj_points = torch.zeros_like(inpaint_traj)
    inpaint_traj_mask_points = torch.zeros_like(inpaint_traj_mask)

    inpaint_traj_points_filled = torch.zeros_like(inpaint_traj)

    # For motion inpaint (second stage only)
    inpaint_motion = torch.zeros([batch_size, feat_dim_motion, 1, max_len], device=data_device)
    inpaint_mask = torch.zeros_like(inpaint_motion, dtype=torch.bool)
    # For second stage inpainting, we only use key locations as target instead of the interpolated lines
    inpaint_motion_points = torch.zeros_like(inpaint_motion)
    inpaint_mask_points = torch.zeros_like(inpaint_mask)


    # we draw a point-to-point line between key locations and impute
    for idx in range(batch_size):
        key_positions = gt_motions[idx, 0, :, :].permute(1, 0)  # [keyframe_len, feature]
        # Initialization
        cur_key_pos = key_positions[0]
        last_kframe = 0
        for kframe_id, kframe_t in enumerate(sampled_keyframes):

            diff = kframe_t - last_kframe
            key_pos = key_positions[kframe_id, traj_only_idcs]

            # Loop to get an evenly space trajectory
            for i in range(diff):
                inpaint_traj[idx, :, 0, last_kframe + i] = (cur_key_pos + (key_pos-cur_key_pos) * i / diff)
                inpaint_traj_mask[idx, :, 0, last_kframe + i] = True

                if i>0:
                    inpaint_traj_points_filled[idx, :, 0, last_kframe+i] = (cur_key_pos + (key_pos-cur_key_pos) * i / diff)

            inpaint_traj_points[idx, :, 0, kframe_t] = key_pos
            inpaint_traj_mask_points[idx, :, 0, kframe_t] = True
            inpaint_traj_points_filled[idx, :, 0, kframe_t] = key_pos

            cur_key_pos = key_pos
            last_kframe = kframe_t
            # Add last key point
            if kframe_id == len(sampled_keyframes) - 1:
                inpaint_traj[idx, :, 0, kframe_t] = key_pos
                inpaint_traj_mask[idx, :, 0, kframe_t] = True

                inpaint_traj_points_filled[idx, :, 0, kframe_t] = key_pos#.unsqueeze(-1)


    # Copy the traj values into inpainted motion
    # For motion we do not have to do transform, yet, because gradients are computed in original pose space
    inpaint_motion[:, traj_only_idcs, :, :] = inpaint_traj[:, :, :, :]
    inpaint_motion_points[:, traj_only_idcs, :, :] = inpaint_traj_points[:, :, :, :]
    inpaint_mask[:, traj_only_idcs, :, :] = inpaint_traj_mask[:, :, :, :]
    inpaint_mask_points[:, traj_only_idcs, :, :] = inpaint_traj_mask_points[:, :, :, :]


    # The full trajectories for inpainting we can already compute
    # [bs, 4, 1, 200]
    inpaint_traj = dataset.t2m_dataset.transform_th(inpaint_traj[:,:feat_dim_traj].permute(0, 2, 3, 1),
                                                      use_rand_proj=False).permute(0, 3, 1, 2)
    # [bs, 4, 1, 200]
    inpaint_traj_points = dataset.t2m_dataset.transform_th(inpaint_traj_points[:,:feat_dim_traj].permute(0, 2, 3, 1),
                                                      use_rand_proj=False).permute(0, 3, 1, 2)


    return (target, target_mask, inpaint_traj, inpaint_traj_mask, inpaint_traj_points, inpaint_traj_mask_points,
        inpaint_motion, inpaint_mask, inpaint_motion_points, inpaint_mask_points, inpaint_traj_points_filled)


class CondKeyLocations:
    def __init__(self,
                 target=None,
                 target_mask=None,
                 transform=None,
                 inv_transform=None,
                 abs_3d=False,
                 classifiler_scale=50.0,
                 reward_model=None,
                 reward_model_args=None,
                 use_mse_loss=False,
                 guidance_style='xstart',
                 stop_cond_from=0,
                 use_rand_projection=False,
                 cut_frame = 200,
                 print_every=None,
                 ):
        self.target = target
        self.target_mask = target_mask
        self.transform = transform
        self.inv_transform = inv_transform
        self.abs_3d = abs_3d
        self.classifiler_scale = classifiler_scale
        self.reward_model = reward_model
        self.reward_model_args = reward_model_args
        self.use_mse_loss = use_mse_loss
        self.guidance_style = guidance_style
        self.stop_cond_from = stop_cond_from
        self.use_rand_projection = use_rand_projection
        self.cut_frame = cut_frame
        self.print_every = print_every
        self.gt_style = 'target'

    def __call__(self, x, t, p_mean_var, y=None,): # *args, **kwds):
        """
        Args:
            target: [bs, 120, 22, 3]
            target_mask: [bs, 120, 22, 3]
        """
        # Stop condition

        if int(t[0]) < self.stop_cond_from:
            return torch.zeros_like(x)
        assert y is not None
        # x shape [bs, 263, 1, 120]
        with torch.enable_grad():
            if self.gt_style == 'target':
                if self.guidance_style == 'xstart':
                    if self.reward_model is not None:
                        # If the reward model is provided, we will use xstart from the
                        # reward model instead.The reward model predict M(x_start | x_t).
                        # The gradient is always computed w.r.t. x_t
                        x = x.detach().requires_grad_(True)
                        reward_model_output = self.reward_model(
                            x, t, **self.reward_model_args)  # this produces xstart
                        xstart_in = reward_model_output
                    else:
                        xstart_in = p_mean_var['pred_xstart']
                elif self.guidance_style == 'eps':
                    # using epsilon style guidance
                    assert self.reward_model is None, "there is no need for the reward model in this case"
                    raise NotImplementedError()
                else:
                    raise NotImplementedError()
                if y['grasp_model']:
                    use_rand_proj = False  # x contains only (pose,x,z,y)
                else:
                    use_rand_proj = self.use_rand_projection
                x_in_pose_space = self.inv_transform(
                    xstart_in.permute(0, 2, 3, 1),
                    traject_only=y['grasp_model'],
                    use_rand_proj=use_rand_proj
                )  # [bs, 1, 120, 263]


                trajec = x_in_pose_space.permute(0,2,1,3)
                batch_size = trajec.shape[0]
                trajec = trajec[:, :, 0]

                weights = torch.ones_like(self.target_mask[:, :self.cut_frame, 0, :], dtype=torch.int32)

                if self.use_mse_loss:
                    loss_sum = F.mse_loss(trajec , self.target[:, :self.cut_frame, 0, :],
                                        reduction='none') * self.target_mask[:, :self.cut_frame, 0, :] #* weights
                else:
                    loss_sum = F.l1_loss(trajec , self.target[:, :self.cut_frame, 0, :],
                                        reduction='none') * self.target_mask[:, :self.cut_frame, 0, :]
                loss_sum = loss_sum.sum()
                # Scale the loss up so that we get the same gradient as if each sample is computed individually
                loss_sum = loss_sum / self.target_mask.sum() * batch_size

            elif self.gt_style == 'inpainting_motion':
                batch_size = x.shape[0]
                # [bs, 4, 1, 120]
                xstart_in = p_mean_var['pred_xstart']
                inpainted_motion = y['current_inpainted_motion']
                inpainting_mask = y['current_inpainting_mask']
                # Inpainting motion
                if self.use_mse_loss:
                    loss_sum = F.mse_loss(xstart_in, inpainted_motion, reduction='none') * inpainting_mask
                else:
                    loss_sum = F.l1_loss(xstart_in, inpainted_motion, reduction='none') * inpainting_mask
                # Scale the loss up so that we get the same gradient as if each sample is computed individually
                loss_sum = loss_sum.sum() / inpainting_mask.sum() * batch_size
            else:
                raise NotImplementedError()

            if self.print_every is not None and int(t[0]) % self.print_every == 0:
                print("%03d: %f" % (int(t[0]), float(loss_sum) / batch_size))

            grad = torch.autograd.grad(-loss_sum, x)[0]

            return grad * self.classifiler_scale

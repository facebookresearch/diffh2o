# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import numpy as np
import torch

from pytorch3d.transforms.rotation_conversions import rotation_6d_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d
from scipy.spatial.transform import Rotation as R


def batched_index_select(input, index, dim=1):
    '''
    :param input: [B, N1, *]
    :param dim: the dim to be selected
    :param index: [B, N2]
    :return: [B, N2, *] selected result
    '''
    views = [input.size(0)] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim=dim, index=index)

def nearest_point(A, B, topk=1):
    # Ensure tensors are on the same device
    assert A.device == B.device, "Both tensors must be on the same device"

    # Calculate pairwise distances
    dists = torch.cdist(A, B)

    # Get the minimum distance and index
    # get top 5 nearest points

    C, D = torch.topk(dists, topk, dim=2, largest=False)

    return C, D

def sample_to_hand_motion(sample_list, args, model_kwargs, model, n_frames,
                     data_inv_transform_fn):

    if not isinstance(sample_list, list):
        sample_list = [sample_list]

    all_motions, all_motions_pose_space, all_lengths, all_text = [], [], [], []
    for sample in sample_list:
        # sample = init_image
        # Recover XYZ *positions* from HumanML3D vector representation
        #if model.data_rep == 'hml_vec':
        n_joints = 21
        # n_joints = 22 if sample.shape[1] == 263 else 21
        if args.traj_only:
            n_joints = 4

        # (1, 263, 1, 120)
        sample_pose_space = data_inv_transform_fn(sample.cpu().permute(0, 2, 3,1)).float() #use_rand_proj=False


        #sample = recover_from_ric(sample, n_joints, abs_3d=args.abs_3d)
        #sample_save = sample + 0.0
        #sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        # rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'
        #                                                 ] else model.data_rep
        # rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs[
        #     'y']['mask'].reshape(args.batch_size, n_frames).bool()
        # sample = model.rot2xyz(x=sample,
        #                         mask=rot2xyz_mask,
        #                         pose_rep=rot2xyz_pose_rep,
        #                         glob=True,
        #                         translation=True,
        #                         jointstype='smpl',
        #                         vertstrans=True,
        #                         betas=None,
        #                         beta=0,
        #                         glob_rot=None,
        #                         get_rotations_back=False)

        if args.unconstrained:
            all_text += ['unconstrained'] * args.num_samples
        else:
            text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
            # all_text += model_kwargs['y'][text_key]
            all_text += model_kwargs['y'][text_key] #* args.num_samples

        all_motions.append(sample.cpu().numpy())
        all_motions_pose_space.append(sample_pose_space.cpu().numpy())
        # all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())
        if args.text_prompt != '':
            all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())
        else:
            all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

        print(f"created {len(all_motions) * args.batch_size} samples")

    return all_motions_pose_space, all_motions, all_lengths, all_text

def recover_from_ric(data, object_rot_relative = True, add_obj_pos = True, num_joints=1):

        data = torch.tensor(data, dtype=torch.float32)

        obj_rot = rotation_6d_to_matrix(data[:,0,:,-6:])
        obj_pos = data[:,0,:,-9:-6]

        positions_left = data[..., : num_joints * 3].swapaxes(1,2)
        positions_right = data[..., num_joints * 3: num_joints * 3 * 2].swapaxes(1,2)

        '''Add rotation to local joint positions'''
        if object_rot_relative:
            positions_left = torch.matmul(obj_rot,positions_left.swapaxes(2,3)).permute(0,1,3,2)
            positions_right = torch.matmul(obj_rot,positions_right.swapaxes(2,3)).permute(0,1,3,2)
        if add_obj_pos:
            '''Add obj root to joints'''
            positions_left += obj_pos[:,:,np.newaxis]
            positions_right += obj_pos[:,:,np.newaxis]
        '''Concate root and joints'''

        global_orient_l = rotation_6d_to_matrix(data[:,0,:,num_joints * 3 * 2 + 24: num_joints * 3 * 2 + 30])
        global_orient_r = rotation_6d_to_matrix(data[:,0,:,num_joints * 3 * 2 + 54: num_joints * 3 * 2 + 60])

        if object_rot_relative:
            global_orient_l =  matrix_to_rotation_6d(torch.matmul(obj_rot,global_orient_l))
            global_orient_r = matrix_to_rotation_6d(torch.matmul(obj_rot,global_orient_r))
        else:
            global_orient_l =  matrix_to_rotation_6d(global_orient_l)
            global_orient_r = matrix_to_rotation_6d(global_orient_r)

        return positions_left.numpy(), positions_right.numpy(), global_orient_l, global_orient_r, obj_pos

def recover_from_ric_artigrasp(data, object_rot_relative = True, add_obj_pos = True, num_joints=1):

        # data = torch.tensor(data, dtype=torch.float32)

        obj_rot = R.from_euler('xyz',data[:,0,:, -3:].reshape(-1,3)).as_matrix().reshape(data.shape[0], -1, 3, 3)
        obj_pos = data[:,0,:,-6:-3]

        positions_left = data[..., : num_joints * 3].swapaxes(1,2)
        positions_right = data[..., num_joints * 3: num_joints * 3 * 2].swapaxes(1,2)

        '''Add rotation to local joint positions'''
        if object_rot_relative:
            positions_left = np.transpose(np.matmul(obj_rot,positions_left.swapaxes(2,3)),axes=(0,1,3,2))
            positions_right = np.transpose(np.matmul(obj_rot,positions_right.swapaxes(2,3)),axes=(0,1,3,2))
        if add_obj_pos:
            '''Add obj root to joints'''
            positions_left += obj_pos[:,:,np.newaxis]
            positions_right += obj_pos[:,:,np.newaxis]
        '''Concate root and joints'''

        global_orient_l = R.from_euler('xyz',data[:,0,:, 51:54].reshape(-1,3)).as_matrix().reshape(data.shape[0], -1, 3, 3)
        global_orient_r = R.from_euler('xyz',data[:,0,:,99:102].reshape(-1,3)).as_matrix().reshape(data.shape[0], -1, 3, 3)

        if object_rot_relative:
            global_orient_l =  np.matmul(obj_rot,global_orient_l)
            global_orient_r = np.matmul(obj_rot,global_orient_r)

        global_orient_l = R.from_matrix(global_orient_l.reshape(-1,3,3)).as_euler('xyz').reshape(data.shape[0], -1, 3)
        global_orient_r = R.from_matrix(global_orient_r.reshape(-1,3,3)).as_euler('xyz').reshape(data.shape[0], -1, 3)

        return positions_left, positions_right, global_orient_l, global_orient_r, obj_pos

def stich_pregrasp(data, pos_left, pos_right, global_orient_l, global_orient_r, obj_pos, add_obj_pos = False, num_joints=1):

        data = torch.tensor(data, dtype=torch.float32)

        positions_left = data[..., : num_joints * 3].swapaxes(1,2)
        positions_right = data[..., num_joints * 3: num_joints * 3 * 2].swapaxes(1,2)

        wrist_pose_l = rotation_6d_to_matrix(data[:,0,:,num_joints * 3 * 2 + 24: num_joints * 3 * 2 + 30])
        wrist_pose_r = rotation_6d_to_matrix(data[:,0,:,num_joints * 3 * 2 + 54: num_joints * 3 * 2 + 60])

        wrist_pose_l = matrix_to_rotation_6d(torch.matmul(global_orient_l[:,-1:],wrist_pose_l))
        wrist_pose_r = matrix_to_rotation_6d(torch.matmul(global_orient_r[:,-1:],wrist_pose_r))
        #R.from_matrix(torch.matmul(global_orient_r[:,-1:],wrist_pose_r).numpy().reshape(-1,3,3)).as_rotvec().reshape(data.shape[0],-1,3)

        wrist_pos_l = np.matmul(global_orient_l[:,-1:].numpy(),positions_left.swapaxes(2,3)).swapaxes(2,3) #np.matmul(positions_left,obj_rot) #
        wrist_pos_r = np.matmul(global_orient_r[:,-1:].numpy(),positions_right.swapaxes(2,3)).swapaxes(2,3) #np.matmul(positions_right,obj_rot)#

        # '''Add obj root to joints'''
        wrist_pos_l += pos_left[:,-1:]
        wrist_pos_r += pos_right[:,-1:]

        if add_obj_pos:
            obj_pos = data[:,0,:,-9:-6]
            wrist_pos_l += obj_pos[:,:, np.newaxis]
            wrist_pos_r += obj_pos[:,:, np.newaxis]

        return wrist_pos_l.numpy(), wrist_pos_r.numpy(), wrist_pose_l.numpy(), wrist_pose_r.numpy()

def sample_to_motion(sample_list, args, model_kwargs, model, n_frames,
                     data_inv_transform_fn):

    if not isinstance(sample_list, list):
        sample_list = [sample_list]

    all_motions, all_lengths, all_text = [], [], []
    for sample in sample_list:
        # sample = init_image
        # Recover XYZ *positions* from HumanML3D vector representation
        if model.data_rep == 'hml_vec':
            if sample.shape[1] == 263 or sample.shape[1] == 67:
                n_joints = 22
            else:
                n_joints = 21
            # n_joints = 22 if sample.shape[1] == 263 else 21
            if args.traj_only:
                n_joints = 4

            # (1, 263, 1, 120)
            sample = data_inv_transform_fn(sample.cpu().permute(0, 2, 3,
                                                                1)).float()
            sample = recover_from_ric(sample, n_joints, abs_3d=args.abs_3d)
            sample_save = sample + 0.0
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'
                                                       ] else model.data_rep
        rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs[
            'y']['mask'].reshape(args.batch_size, n_frames).bool()
        sample = model.rot2xyz(x=sample,
                               mask=rot2xyz_mask,
                               pose_rep=rot2xyz_pose_rep,
                               glob=True,
                               translation=True,
                               jointstype='smpl',
                               vertstrans=True,
                               betas=None,
                               beta=0,
                               glob_rot=None,
                               get_rotations_back=False)

        if args.unconstrained:
            all_text += ['unconstrained'] * args.num_samples
        else:
            text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
            # all_text += model_kwargs['y'][text_key]
            all_text += model_kwargs['y'][text_key] * args.num_samples

        all_motions.append(sample.cpu().numpy())
        # all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())
        if args.text_prompt != '':
            all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy().repeat(
                args.num_samples))
        else:
            all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy().repeat(
            args.num_repetitions))

        print(f"created {len(all_motions) * args.batch_size} samples")

    return all_motions, all_lengths, all_text


def construct_template_variables(unconstrained):
    row_file_template = 'sample{:02d}.mp4'
    all_file_template = 'samples_{:02d}_to_{:02d}.mp4'
    if unconstrained:
        sample_file_template = 'row{:02d}_col{:02d}.mp4'
        sample_print_template = '[{} row #{:02d} column #{:02d} | -> {}]'
        row_file_template = row_file_template.replace('sample', 'row')
        row_print_template = '[{} row #{:02d} | all columns | -> {}]'
        all_file_template = all_file_template.replace('samples', 'rows')
        all_print_template = '[rows {:02d} to {:02d} | -> {}]'
    else:
        sample_file_template = 'sample{:02d}_rep{:02d}.mp4'
        sample_print_template = '["{}" ({:02d}) | Rep #{:02d} | -> {}]'
        row_print_template = '[ "{}" ({:02d}) | all repetitions | -> {}]'
        all_print_template = '[samples {:02d} to {:02d} | all repetitions | -> {}]'

    return sample_print_template, row_print_template, all_print_template, \
           sample_file_template, row_file_template, all_file_template


def save_multiple_samples(args, out_path, row_print_template,
                          all_print_template, row_file_template,
                          all_file_template, caption, num_samples_in_out_file,
                          rep_files, sample_files, sample_i):
    all_rep_save_file = row_file_template.format(sample_i)
    all_rep_save_path = os.path.join(out_path, all_rep_save_file)
    ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
    # hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions}' if args.num_repetitions > 1 else ''
    hstack_args = f' -filter_complex hstack=inputs={args.num_dump_step}' if args.num_dump_step > 1 else ''
    ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(
        ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_path}'
    os.system(ffmpeg_rep_cmd)
    print(row_print_template.format(caption, sample_i, all_rep_save_file))
    sample_files.append(all_rep_save_path)

    # if (sample_i + 1
    #     ) % num_samples_in_out_file == 0 or
    if sample_i + 1 == args.num_samples:
        # if (sample_i + 1) % num_samples_in_out_file == 0 or sample_i + 1 == args.num_repetitions:
        # all_sample_save_file =  f'samples_{(sample_i - len(sample_files) + 1):02d}_to_{sample_i:02d}.mp4'
        all_sample_save_file = all_file_template.format(
            sample_i - len(sample_files) + 1, sample_i)
        all_sample_save_path = os.path.join(out_path, all_sample_save_file)
        print(
            all_print_template.format(sample_i - len(sample_files) + 1,
                                      sample_i, all_sample_save_file))
        ffmpeg_rep_files = [f' -i {f} ' for f in sample_files]
        vstack_args = f' -filter_complex vstack=inputs={len(sample_files)}' if len(
            sample_files) > 1 else ''
        ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(
            ffmpeg_rep_files) + f'{vstack_args} {all_sample_save_path}'
        os.system(ffmpeg_rep_cmd)
        sample_files = []
    return sample_files


def vis_fn(sample, filename, data, model, concat=False, abs_3d=False, traj_only=False):
    '''This function is for logging the motion during the denoising process.
    Not in use right now
    '''
    save_at = os.path.join(out_path, "log_dump", filename + ".mp4")

    sample = data.dataset.t2m_dataset.inv_transform(
        sample.detach().cpu().permute(0, 2, 3, 1)).float()
    sample = recover_from_ric(sample, 22, abs_3d=abs_3d)
    sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)
    rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'
                                                    ] else model.data_rep
    rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs[
        'y']['mask'].reshape(args.batch_size, n_frames).bool()
    sample = model.rot2xyz(x=sample,
                            mask=rot2xyz_mask,
                            pose_rep=rot2xyz_pose_rep,
                            glob=True,
                            translation=True,
                            jointstype='smpl',
                            vertstrans=True,
                            betas=None,
                            beta=0,
                            glob_rot=None,
                            get_rotations_back=False)

    sample = sample[0].detach().cpu().numpy()
    sample = sample.transpose(2, 0, 1)[:120]  # need to be [120, 22, 3]

    skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain
    if traj_only:
        skeleton = [[0, 0]]
    plot_3d_motion(save_at,
                    skeleton,
                    sample,
                    dataset=args.dataset,
                    title=model_kwargs['y']['text'][0] + " " + filename,
                    fps=fps)

    if concat:
        name_list = [
            os.path.join(out_path, "log_dump", "previous_x_start.mp4"),
            os.path.join(out_path, "log_dump", "pred_x_start.mp4")
        ]

        # all_rep_save_file = row_file_template.format(sample_i)
        all_rep_save_path = os.path.join(out_path, "log_dump",
                                            "compare.mp4")
        ffmpeg_rep_files = [f' -i {f} ' for f in name_list]
        hstack_args = f' -filter_complex hstack=inputs=2'
        ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(
            ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_path}'
        os.system(ffmpeg_rep_cmd)
        # print(row_print_template.format(caption, sample_i, all_rep_save_file))
        # sample_files.append(all_rep_save_path)


def plot_grad(out_path, rep_i):
    '''Function for check gradient of x during the denoising process.
    Not in use right now.
    '''
    # snd_last = sample_list[-2].squeeze(2).squeeze(0).detach().cpu().numpy()
    # last = sample_list[-1].squeeze(2).squeeze(0).detach().cpu().numpy()
    # plt.figure(10, 10)
    # f, axarr = plt.subplots(1,3)
    # axarr[0].imshow(snd_last)
    # axarr[1].imshow(last)
    # axarr[2].imshow(abs(last-snd_last))
    # plt.show()

    # Plot loss and autograd norm for each sample
    with open(os.path.join(out_path, "%d" % rep_i) + "_grad_norm.txt",
              "r") as ff:
        all_lines = ff.readlines()
        all_lines = [float(aa.strip()) for aa in all_lines]
        grad_norm_np = np.array(all_lines)
    with open(
            os.path.join(out_path, "%d" % rep_i) + "_grad_norm_wo_scale.txt",
            "r") as ff:
        all_lines = ff.readlines()
        all_lines = [float(aa.strip()) for aa in all_lines]
        grad_norm_wo_scale_np = np.array(all_lines)
    with open(os.path.join(out_path, "%d" % rep_i) + "_loss.txt", "r") as ff:
        all_lines = ff.readlines()
        all_lines = [float(aa.strip()) for aa in all_lines]
        loss_list_np = np.array(all_lines)
    f, axs = plt.subplots(3)
    axs[0].plot(grad_norm_np)  # np.arange(999, -1, -1),
    axs[1].plot(grad_norm_wo_scale_np)
    axs[2].plot(loss_list_np)
    axs[0].title.set_text("grad norm")
    axs[1].title.set_text("grad norm w/o scale")
    axs[2].title.set_text("sum loss")
    f.tight_layout()
    # plt.gca().invert_yaxis()
    plt.savefig(os.path.join(out_path, "loss_" + "%d.png" % (rep_i)))
    # plt.show()
    plt.close()

    ## Plot grad heat map to check denoiser chaning the final motion near the end
    # Load grad from small files
    grad_list = []
    xyz_list = []
    # from 0 to 1000. Number of denoised steps
    for ds in range(1000):
        cur_grad = np.load(
            os.path.join(out_path, "log_dump", "%d_%d_grad.npy" % (rep_i, ds)))
        cur_xyz = np.load(
            os.path.join(out_path, "log_dump", "%d_%d_xyz.npy" % (rep_i, ds)))
        grad_list.append(cur_grad)
        xyz_list.append(cur_xyz)
        os.remove(
            os.path.join(out_path, "log_dump", "%d_%d_grad.npy" % (rep_i, ds)))
        os.remove(
            os.path.join(out_path, "log_dump", "%d_%d_xyz.npy" % (rep_i, ds)))
    all_grad = np.asarray(grad_list).squeeze(3).squeeze(1)  # [1000, 263, 120]
    all_xyz = np.asarray(xyz_list).squeeze(2).squeeze(1)  # [1000, 120, 22, 3]
    np.save(os.path.join(out_path, "log_dump", "%d_all_grad.npy" % (rep_i)),
            all_grad)
    np.save(os.path.join(out_path, "log_dump", "%d_all_xyz.npy" % (rep_i)),
            all_xyz)
    # Plot grad norm
    grad_xyz_norm = np.linalg.norm(all_grad[:, :3, :], axis=1)  # [1000, 120]
    ax = sns.heatmap(grad_xyz_norm.T)
    # plt.show()
    plt.savefig(os.path.join(out_path, "grad_norm_xyz" + "%d.png" % (rep_i)))
    plt.close()

    # Plot motion step diff
    xyz_diff = all_xyz[1:, :, 0, :] - all_xyz[:-1, :, 0, :]
    xyz_diff_norm = np.linalg.norm(xyz_diff, axis=2)  # [999, 120]
    ax = sns.heatmap(xyz_diff_norm.T)
    # plt.show()
    plt.savefig(os.path.join(out_path, "loss_diff" + "%d.png" % (rep_i)))
    plt.close()

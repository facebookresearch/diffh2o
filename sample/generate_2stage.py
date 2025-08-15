# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
# Python
import copy
import json
import os
import shutil

# Thirdparty
import numpy as np
import torch
from data_loaders.get_data import DatasetConfig, get_dataset_loader
from data_loaders.humanml.utils.data_utils import OBJECT_LIST_NEW, REPRESENTATION_IDCS
from data_loaders.tensors import collate
from eval.eval_grab import EvalNode
from model.cfg_sampler import ClassifierFreeSampleModel
from sample.condition_hands import CondKeyLocations, get_grasp_references
from utils import dist_util
from utils.fixseed import fixseed
from utils.generation_template import get_template
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils.output_util import recover_from_ric, sample_to_hand_motion, stich_pregrasp
from utils.parser_util import generate_args


def load_grasp_model(data, args_traj):
    '''
    The trajectory model predicts trajectory that will be use for infilling in motion model.
    Create a trajectory model that produces trajectory to be inptained by the motion model.
    '''
    grasp_model, grasp_diffusion = create_model_and_diffusion(args_traj, data)

    print(f"Loading traj model checkpoints from [{args_traj.grasp_model_path}]...")
    load_saved_model(grasp_model, args_traj.grasp_model_path)
    
    grasp_model.to(dist_util.dev())
    grasp_model.eval()  # disable random masking
    return grasp_model, grasp_diffusion

def main():
    args = generate_args()
    print(args.__dict__)
    print(args.arch)
    print("##### Additional Guidance Mode: %s #####" % args.guidance)

    args = get_template(args, guidance=args.guidance)

    args_interaction = args
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')

    args = generate_args(model_path=args.grasp_model_path)

    fixseed(args.seed)
    out_path = args_interaction.output_dir

    num_downsampled_frames = args.num_downsampled_frames

    n_frames = args.max_frames_interaction  if not args.pre_grasp else args.max_frames_grasp
    n_frames_post = args.max_frames_interaction 

    is_using_data = not any([
        args.input_text, args.text_prompt, args.action_file, args.action_name
    ])
    dist_util.setup_dist(args.device)

    if out_path == '':
        out_path = os.path.join(os.path.dirname(args_interaction.model_path),
                                'samples_{}'.format(niter))

    # this block must be called BEFORE the dataset is loaded
    if args.text_prompt != '':
        texts = [args.text_prompt]
        args.num_samples = 3
        args.num_repetitions = 1
    elif args.input_text != '':
        assert os.path.exists(args.input_text)
        with open(args.input_text, 'r') as fr:
            texts = fr.readlines()
        texts = [s.replace('\n', '') for s in texts]
        args.num_samples = len(texts)
    elif args.action_name:
        action_text = [args.action_name]
        args.num_samples = 1
    elif args.action_file != '':
        assert os.path.exists(args.action_file)
        with open(args.action_file, 'r') as fr:
            action_text = fr.readlines()
        action_text = [s.replace('\n', '') for s in action_text]
        args.num_samples = len(action_text)

    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'

    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples
    args_interaction.batch_size = args.num_samples
    print('Loading dataset...')
    data = load_dataset(args, n_frames, n_frames)
    data_interaction = load_dataset(args_interaction, n_frames, n_frames)
    total_num_samples = args.num_samples * args.num_repetitions

    if args.eval_entire_set:
        num_reps = len(data.dataset.t2m_dataset.name_list)// args.batch_size * args.num_repetitions
    else:
        num_reps = args.num_repetitions
    print("NUM REPS", num_reps)
    all_motions = []
    all_motions_downsampled = []
    all_gt_kf = []
    all_lengths = []
    all_data_id = []
    all_text = []
    all_iv = []
    all_id = []
    all_id_max = []
    all_cr = []
    all_jp = []
    all_ja = []
    all_nc = []
    t_vels = []
    acc_glob_pos = []
    acc_loc_pos = []
    acc_glob_rot = []
    acc_loc_rot = []
    handedness_accuracy = []
    grasp_reference_error = []

    pos_left_idcs = REPRESENTATION_IDCS['pos_left']
    pos_right_idcs = REPRESENTATION_IDCS['pos_right']
    glob_orient_l_idcs = REPRESENTATION_IDCS['global_orient_l']
    glob_orient_r_idcs = REPRESENTATION_IDCS['global_orient_r']
    obj_pose_idcs = REPRESENTATION_IDCS['object_pose']

    for rep_i in range(num_reps):
        print("REP", rep_i)
        print("Creating model and diffusion...")
        model, diffusion = create_model_and_diffusion(args, data)

        ###################################
        # LOADING THE MODEL FROM CHECKPOINT
        print(f"Loading checkpoints from [{args.model_path}]...")
        load_saved_model(model, args.model_path) # , use_avg_model=args.gen_avg_model)

        model.to(dist_util.dev())
        model.eval() 
        ###################################

        evaluator = EvalNode(args.obj_model_path, args.sbj_model_path, args.mano_model_path)

        if is_using_data:
            iterator_pre = iter(data)
            iterator = iter(data_interaction)

            motion_gt_grasp, model_kwargs_grasp = next(iterator_pre)
            texts_pre = model_kwargs_grasp['y']['text']

            motion_gt, model_kwargs = next(iterator)
            texts = model_kwargs['y']['text']
            for _ in range(rep_i):
                motion_gt_grasp, model_kwargs_grasp = next(iterator_pre)
                texts_pre = model_kwargs_grasp['y']['text']

                motion_gt, model_kwargs = next(iterator)
                texts = model_kwargs['y']['text']
        else:
            iterator_pre = iter(data)
            iterator = iter(data_interaction)

            for _ in range(rep_i+1):
                motion_gt_grasp, model_kwargs_grasp = next(iterator_pre)
                motion_gt, model_kwargs = next(iterator)

            collate_args = [{
                'inp': torch.zeros(n_frames),
                'tokens': None,
                'lengths': n_frames,
            }] * args.num_samples
            is_t2m = any([args.input_text, args.text_prompt])

            if args.obj_enc:
                collate_args = [dict(arg, text=txt, object_bps=bps) for arg, txt, bps in zip(collate_args, texts, bps_list)]
            else:
                collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]


            _, model_kwargs_grasp = collate(collate_args)

        #####################################################
        # There are two sets of objects that have ambiguous names (e.g. phone and headphone)
        #####################################################
        for i in range(len(texts)):
            obj_name = [s for s in OBJECT_LIST_NEW if s in texts_pre[i]]
            obj_name_post = [s for s in OBJECT_LIST_NEW if s in texts[i]]
            assert len(obj_name) >= 1

            if len(obj_name) > 1:
                if 'phone' in obj_name:
                    obj_name = ['phone']
                    obj_name_post = ['phone']
                else:
                    obj_name = ['wristwatch']
                    obj_name_post = ['wristwatch']
      
            texts_pre[i] = texts_pre[i].replace(obj_name[0], obj_name_post[0])


        model_kwargs_grasp['y']['text'] = texts_pre
        model_kwargs['y']['text'] = texts


        key_frames = [0, 49] ### As the pre-grasp data in GRAB is always 50 frames, we take the last frame

        # Transform into unnormalized space
        motion_gt = data_interaction.dataset.t2m_dataset.inv_transform_th(motion_gt.cpu().permute(0, 2, 3, 1)).float().permute(0, 1, 3, 2)
        motion_gt_kf = motion_gt[...,key_frames] 


        #####################################################
        # Collect grasp references
        #####################################################
        (target, target_mask,
            inpaint_traj_p2p, inpaint_traj_mask_p2p,
            inpaint_traj_points, inpaint_traj_mask_points,
            inpaint_motion_p2p, inpaint_mask_p2p,
            inpaint_motion_points, inpaint_mask_points, inpaint_traj_points_filled) = get_grasp_references(motion_gt_kf,
                                                                                    key_frames, data.dataset, data_interaction.dataset, max_len=n_frames,
                                                                                    feat_dim_traj=motion_gt_grasp.shape[1], feat_dim_motion=117)
        
        # Name for logging
        model_kwargs_grasp['y']['grasp_model'] = args.traj_only
        model_kwargs['y']['grasp_model'] = args_interaction.traj_only
        #########################################
        # loading another model for trajectory conditioning
        grasp_model, grasp_diffusion = load_grasp_model(data, args)
        grasp_model_kwargs = copy.deepcopy(model_kwargs_grasp)
        grasp_model_kwargs['y']['log_name'] = out_path
        grasp_model_kwargs['y']['grasp_model'] = True
        args.do_inpaint = True
        #############################################

        model_device = next(model.parameters()).device

        # Standardized conditioning
        impute_slack = 20
        impute_until = 100
        motion_cond_until = 20
        motion_impute_until = 100

        target = target[...,:motion_gt_grasp.shape[1]].to(model_device)
        target_mask = target_mask[...,:motion_gt_grasp.shape[1]].to(model_device)
        model_kwargs['y']['target'] = target
        model_kwargs['y']['target_mask'] = target_mask

        # Output path
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
            os.makedirs(out_path)
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        args_path = os.path.join(out_path, 'args.json')
        with open(args_path, 'w') as fw:
            json.dump(vars(args), fw, indent=4, sort_keys=True)

        ############################################

        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(
                args.batch_size, device=dist_util.dev()) * args.guidance_param
            grasp_model_kwargs['y']['scale'] = torch.ones(
                args.batch_size,
                device=dist_util.dev()) * args.guidance_param

        #####################################################
        # Generate the grasp sequence
        #####################################################
        grasp_model_kwargs['y']['log_id'] = 0
        ### Standardized conditioning

        ### Inpaint with p2p
        grasp_model_kwargs['y']['inpainted_motion'] = inpaint_traj_p2p[:,:motion_gt_grasp.shape[1]].to(model_device)
        grasp_model_kwargs['y']['inpainting_mask'] = inpaint_traj_mask_p2p[:,:motion_gt_grasp.shape[1]].to(model_device)

        # Set when to stop imputing
        grasp_model_kwargs['y']['cond_until'] = impute_slack
        grasp_model_kwargs['y']['impute_until'] = impute_until
        grasp_model_kwargs['y']['impute_until_second_stage'] = impute_slack
        grasp_model_kwargs['y']['inpainted_motion_second_stage'] = inpaint_traj_points[:,:motion_gt_grasp.shape[1]].to(model_device)
        grasp_model_kwargs['y']['inpainting_mask_second_stage'] = inpaint_traj_mask_points[:,:motion_gt_grasp.shape[1]].to(model_device)

        grasp_diffusion.data_transform_fn = data.dataset.t2m_dataset.transform_th
        grasp_diffusion.data_inv_transform_fn = data.dataset.t2m_dataset.inv_transform_th
        grasp_diffusion.log_trajectory_fn = None
        grasp_diffusion.data_get_mean_fn = data.dataset.t2m_dataset.get_std_mean
        
        if args.guidance:
            cond_fn_traj = CondKeyLocations(target=target,
                                        target_mask=target_mask,
                                        transform=data.dataset.t2m_dataset.transform_th,
                                        inv_transform=data.dataset.t2m_dataset.inv_transform_th,
                                        abs_3d=args.abs_3d,
                                        classifiler_scale=args.classifier_scale,
                                        use_mse_loss=args.gen_mse_loss,
                                        use_rand_projection=False,
                                        )

        else:
            cond_fn_traj = None

        sample_fn = grasp_diffusion.p_sample_loop
        dump_steps = [999]

        grasp_sample = sample_fn(
            grasp_model,
            (args.batch_size, grasp_model.njoints, grasp_model.nfeats,
                n_frames),
            clip_denoised=True,  # False,
            model_kwargs=grasp_model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None, # None,
            progress=True,
            dump_steps= dump_steps,
            noise=None,
            const_noise=False,
            cond_fn=cond_fn_traj,
        )


        sample = grasp_sample

        gen_eff_len = min(sample[0].shape[-1], n_frames)
        print('cut the motion length to', gen_eff_len)
        for j in range(len(sample)):
            sample[j] = sample[j][ :, :, :gen_eff_len]

        num_dump_step=1

        cur_motions_grasp_pose_space, _, _, cur_texts = sample_to_hand_motion(
            grasp_sample, args, model_kwargs, model, gen_eff_len,
            data.dataset.t2m_dataset.inv_transform)

        #####################################################
        # Load Interaction Model
        #####################################################
        print("Loading interaction model and diffusion...")
        model, diffusion = create_model_and_diffusion(args_interaction, data_interaction)
        print(f"Loading checkpoints from [{args_interaction.model_path}]...")
        load_saved_model(model, args_interaction.model_path) 

        model.to(dist_util.dev())
        model.eval() 

        #####################################################
        # Subsequence Imputing
        #####################################################
        ### --- inpaint start
        model_kwargs['y']['inpainting_mask'] = torch.zeros((args.num_samples,data_interaction.dataset.t2m_dataset.mean.shape[0],1,n_frames_post),dtype=torch.bool).to(model_device)
        model_kwargs['y']['inpainting_mask'][:, :obj_pose_idcs[0], :, :n_frames] = True
        model_kwargs['y']['inpainted_motion'] = torch.zeros((args.num_samples,data_interaction.dataset.t2m_dataset.mean.shape[0],1,n_frames_post),dtype=torch.float32).to(model_device)

        # #### TRANSFORM from pre-grasp pose transformed space to post-grasp transformed space
        model_kwargs['y']['inpainted_motion'][:,:obj_pose_idcs[0], :, :n_frames] = torch.tensor(np.array(cur_motions_grasp_pose_space)[-1]).permute(0,3,1,2)[:,:obj_pose_idcs[0],:,-n_frames:].to(model_device) #
        model_kwargs['y']['inpainted_motion'][:,obj_pose_idcs[0]:, :, :n_frames] = inpaint_motion_p2p[:,obj_pose_idcs[0]:, :, :n_frames]
        model_kwargs['y']['inpainted_motion'] = data_interaction.dataset.t2m_dataset.transform_th(model_kwargs['y']['inpainted_motion'].permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        #####################################################
        # Load Interaction Model
        #####################################################
        sample_fn = diffusion.p_sample_loop
        dump_steps = [999]

        if args_interaction.pre_grasp:
            ### Inpaint with p2p
            model_kwargs['y']['inpainted_motion'] = inpaint_traj_p2p.to(model_device)
            model_kwargs['y']['inpainting_mask'] = inpaint_traj_mask_p2p.to(model_device)
            args.do_inpaint = True

        if not args.do_inpaint and "inpainted_motion" in model_kwargs['y'].keys():
            del model_kwargs['y']['inpainted_motion']
            del model_kwargs['y']['inpainting_mask']

        # Name for logging
        model_kwargs['y']['log_id'] = rep_i
        model_kwargs['y']['cond_until'] = motion_cond_until  # impute_slack
        model_kwargs['y']['impute_until'] = motion_impute_until # 20  # impute_slack
        # Pass functions to the diffusion
        diffusion.data_get_mean_fn = data_interaction.dataset.t2m_dataset.get_std_mean
        diffusion.data_transform_fn = data_interaction.dataset.t2m_dataset.transform_th
        diffusion.data_inv_transform_fn = data_interaction.dataset.t2m_dataset.inv_transform_th
        diffusion.log_trajectory_fn = None


        # TODO: move the followings to a separate function
        if args.guidance:
            target = torch.zeros((args.num_samples,n_frames_post,1,obj_pose_idcs[1]),dtype=torch.float32).to(model_device)
            model_kwargs['y']['inpainting_mask'] = torch.zeros((args.num_samples,data_interaction.dataset.t2m_dataset.mean.shape[0],1,n_frames_post),dtype=torch.bool).to(model_device)
            model_kwargs['y']['inpainting_mask'][:, :obj_pose_idcs[0], :, :n_frames] = True
            target[:,:n_frames, :, :obj_pose_idcs[0]] = torch.tensor(np.array(cur_motions_grasp_pose_space)[-1]).permute(0,2,1,3)[:,-n_frames:,:,:obj_pose_idcs[0]].to(model_device)

            target_mask = model_kwargs['y']['inpainting_mask'].permute(0,3,2,1)

            cond_fn = CondKeyLocations(target=target,
                                        target_mask=target_mask,
                                        transform=data_interaction.dataset.t2m_dataset.transform_th,
                                        inv_transform=data_interaction.dataset.t2m_dataset.inv_transform_th,
                                        abs_3d=args_interaction.abs_3d,
                                        classifiler_scale=args_interaction.classifier_scale,
                                        use_mse_loss=args_interaction.gen_mse_loss,
                                        use_rand_projection=args_interaction.use_random_proj,
                                        cut_frame=n_frames_post
                                        )

        else:
            cond_fn = None

        sample = sample_fn(
            model,
            (args.batch_size, model.njoints, model.nfeats, n_frames_post),
            clip_denoised=not args_interaction.predict_xstart,
            model_kwargs=model_kwargs,
            skip_timesteps=0, 
            init_image=None,
            progress=True,
            dump_steps=dump_steps,  
            noise=None,
            const_noise=False,
            cond_fn=cond_fn,
        )

        gen_eff_len = min(sample[0].shape[-1], n_frames_post)
        print('cut the motion length to', gen_eff_len)
        for j in range(len(sample)):
            sample[j] = sample[j][:, :, :, :gen_eff_len]
        ###################

        num_dump_step = len(dump_steps)
        args_interaction.num_dump_step = num_dump_step

        cur_motions_pose_space, cur_motions, cur_lengths, cur_texts = sample_to_hand_motion(
            sample, args_interaction, model_kwargs, model, gen_eff_len,
            data_interaction.dataset.t2m_dataset.inv_transform)

 

        cur_motions_pose_space = np.array(cur_motions_pose_space)[0]

        positions_left, positions_right, global_orient_l, global_orient_r, obj_pos = recover_from_ric(cur_motions_pose_space, object_rot_relative = False, add_obj_pos = False)
        cur_motions_pose_space[...,:pos_left_idcs[1]] = positions_left.swapaxes(1,2)
        cur_motions_pose_space[...,pos_right_idcs[0]:pos_right_idcs[1]] = positions_right.swapaxes(1,2)
        cur_motions_pose_space[...,glob_orient_l_idcs[0]:glob_orient_l_idcs[1]] = global_orient_l[:,np.newaxis]
        cur_motions_pose_space[...,glob_orient_r_idcs[0]:glob_orient_r_idcs[1]] = global_orient_r[:,np.newaxis]
        # cur_motions_pose_space[...,:n_frames, obj_pose_idcs[0]:obj_pose_idcs[1]] = cur_motions_pose_space[..., n_frames:n_frames+1, obj_pose_idcs[0]:obj_pose_idcs[1]]
        
        if args.no_subsequence_inpainting:
            motions_grasp = np.zeros((args.num_samples,1,n_frames,data_interaction.dataset.t2m_dataset.mean.shape[0]))
            motions_grasp[...,[111,115]] = 1.0 ### Set 6D rotations to zero
            motions_grasp[...,:obj_pose_idcs[0]] = np.array(cur_motions_grasp_pose_space)[-1]
            positions_left, positions_right, global_orient_l, global_orient_r  = stich_pregrasp(cur_motions_pose_space,
                                                                            positions_left,
                                                                            positions_right,
                                                                            global_orient_l,
                                                                            global_orient_r,
                                                                            obj_pos,
                                                                            add_obj_pos=True)
            cur_motions_pose_space[...,:pos_left_idcs[1]] = positions_left.swapaxes(1,2)
            cur_motions_pose_space[...,pos_right_idcs[0]:pos_right_idcs[1]] = positions_right.swapaxes(1,2)
            cur_motions_pose_space[...,glob_orient_l_idcs[0]:glob_orient_l_idcs[1]] = global_orient_l[:,np.newaxis]
            cur_motions_pose_space[...,glob_orient_r_idcs[0]:glob_orient_r_idcs[1]] = global_orient_r[:,np.newaxis]
            motions_full = np.concatenate((motions_grasp,cur_motions_pose_space),axis=2, dtype=np.float32)
        else:
            motions_full = cur_motions_pose_space
        
        all_motions.extend(motions_full[:,np.newaxis])
        all_lengths.extend(cur_lengths)
        all_text.extend(cur_texts)
        all_data_id.extend(model_kwargs['y']['data_id'])
        

        for j in range(len(cur_lengths[0])):
            frames_per_pose_input = np.array(cur_lengths[0])[j] // num_downsampled_frames
            idcs = np.arange(
                0,  np.array(cur_lengths[0])[j], frames_per_pose_input
            )

            motions_full_ds = motions_full[j, 0, idcs, :6]
            motions_full_ds = motions_full_ds[:num_downsampled_frames]

            all_motions_downsampled.append(motions_full_ds)

        #####################################################
        # Evaluation
        #####################################################
        if args.gen_only_interaction:
            res = evaluator.evaluate_seqs(motions_full[:,:,:], model_kwargs['y']['data_id'], cur_lengths, eval_grasp_reference=False, eval_physics=args.physics_metrics, num_reps=num_reps)
        elif args.gen_only_grasp:
            pass
        else:
            res = evaluator.evaluate_seqs(motions_full[:,:,n_frames-1:], model_kwargs['y']['data_id'], cur_lengths, samples_kf=motion_gt_kf, eval_grasp_reference=args.eval_grasp_reference, eval_physics=args.physics_metrics, num_reps=num_reps)
            t_vels.extend(res["t_vels"])
        
        acc_glob_pos.extend(res["acc_glob_pos"])
        acc_loc_pos.extend(res["acc_loc_pos"])
        acc_glob_rot.extend(res["acc_glob_rot"])
        acc_loc_rot.extend(res["acc_loc_rot"])
        
        if args.eval_grasp_reference:
            handedness_accuracy.extend(res["handedness"])
            grasp_reference_error.extend(res["grasp_error"])

        if args.physics_metrics:
            all_iv.extend(res["inter_volume"])
            all_id.extend(res["inter_depth"])
            all_id_max.extend(res["inter_depth_max"])
            all_cr.extend(res["contact_ratio"])
            all_jp.extend(res["jerk_pos"])
            all_ja.extend(res["jerk_ang"])
            all_nc.extend(res["num_contact_frames"])
        all_gt_kf.extend(motion_gt_kf.swapaxes(2,3))

    #####################################################
    # Store results
    #####################################################
    total_num_samples = args.num_samples * num_reps * num_dump_step * args.num_repetitions

    all_motions = np.concatenate(all_motions,axis=0)  # [bs * num_dump_step, 1, 3, 120]

    all_motions_downsampled = np.array(all_motions_downsampled)
    all_gt_kf = np.concatenate(all_gt_kf,axis=0)  # [bs * num_dump_step, 1, 3, 120]

    all_motions = all_motions[-total_num_samples:]  # #       not sure? [bs, njoints, 6, seqlen]
    all_gt_kf = all_gt_kf[-total_num_samples:]
    all_text = all_text[:total_num_samples]  # len() = args.num_samples * num_dump_step
    all_data_id = all_data_id[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

    overall_diversity = evaluator.evaluate_overall_diversity(all_motions_downsampled)


    result_file = f'results_{args.gen_split}_{args.split_set}.npy'

    npy_path = os.path.join(out_path, result_file)

    print(f"saving results file to [{npy_path}]")

    np.save(
        npy_path, {
            'motion': all_motions[:,:,:n_frames_post],
            'text': all_text,
            'lengths': all_lengths,
            'num_samples': args.num_samples,
            'num_repetitions': args.num_repetitions,
            'gt_kf': all_gt_kf[:,:n_frames_post],
            'data_id': all_data_id,
            'iv': all_iv,
            'id': all_id,
            'id_max': all_id_max,
            'cr': all_cr,
            'jp': all_jp,
            'ja': all_ja,
            'od': overall_diversity,
            't_vels': t_vels,
            'acc_glob_pos': acc_glob_pos,
            'acc_loc_pos': acc_loc_pos,
            'acc_glob_rot': acc_glob_rot,
            'acc_loc_rot': acc_loc_rot,
            'acc_glob_pos_mean' : np.mean(acc_glob_pos),
            'acc_loc_pos_mean' : np.mean(acc_loc_pos),
            'acc_glob_rot_mean' : np.mean(acc_glob_rot),
            'acc_loc_rot_mean' : np.mean(acc_loc_rot),
        })
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    with open(npy_path.replace('.npy', '_scores.txt'), 'w') as fw:
        mean_values = {
            'iv': np.mean(all_iv),
            'id': np.mean(all_id),
            'id_max': np.mean(all_id_max),
            'cr': np.mean(all_cr),
            'jp': np.mean(all_jp),
            'ja': np.mean(all_ja),
            'nc': np.mean(all_nc),
            'od': overall_diversity,
            't_vels': np.mean(t_vels),
            'acc_glob_pos_mean' : np.mean(acc_glob_pos),
            'acc_loc_pos_mean' : np.mean(acc_loc_pos),
            'acc_glob_rot_mean' : np.mean(acc_glob_rot),
            'acc_loc_rot_mean' : np.mean(acc_loc_rot),
            'handedness_accuracy': np.mean(handedness_accuracy),
            'grasp_reference_error': np.mean(grasp_reference_error)
        }
        for key, value in mean_values.items():
            fw.write(f'{key}: {value}\n')

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')



def load_dataset(args, max_frames, n_frames):
    conf = DatasetConfig(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=max_frames,
        split=args.gen_split,
        split_set=args.split_set,
        hml_mode='text_only', 
        use_abs3d=args.abs_3d,
        traject_only=args.traj_only,
        use_random_projection=args.use_random_proj,
        random_projection_scale=args.random_proj_scale,
        augment_type='none',
        std_scale_shift=args.std_scale_shift,
        drop_redundant=args.drop_redundant,
        use_contacts=args.use_contacts,
        data_repr=args.data_repr,
        mean_name=args.mean_name,
        std_name=args.std_name,
        proj_matrix_name=args.proj_matrix_name,
        pre_grasp=args.pre_grasp,
        hands_only=args.hands_only,
        obj_only=args.obj_only,
        text_detailed=args.text_detailed,
    )
    data = get_dataset_loader(conf, shuffle=args.random_order)
    # what's this for?
    data.fixed_length = n_frames
    return data

if __name__ == "__main__":
    main()

# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
# Python
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
from sample.condition_hands import get_grasp_references
from utils import dist_util
from utils.fixseed import fixseed
from utils.generation_template import get_template
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils.output_util import recover_from_ric, sample_to_hand_motion
from utils.parser_util import generate_args


def main():
    args = generate_args()
    print(args.__dict__)
    print(args.arch)
    print("##### Additional Guidance Mode: %s #####" % args.guidance)

    args = get_template(args, guidance=args.guidance)

    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    
    fixseed(args.seed)
    out_path = args.output_dir

    max_frames = 200
    max_downsampled_frames = 15

    n_frames = max_frames if not args.pre_grasp else 50
    n_frames_post = n_frames
    print('n_frames', n_frames_post)

    dist_util.setup_dist(args.device)

    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
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
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples
    args.batch_size = args.num_samples
    print('Loading dataset...')
    data = load_dataset(args, max_frames, n_frames)
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

    for i in range(num_reps):
        print("REP", i)
        print("Creating model and diffusion...")
        model, diffusion = create_model_and_diffusion(args, data)

        ###################################
        # LOADING THE MODEL FROM CHECKPOINT
        print(f"Loading checkpoints from [{args.model_path}]...")
        load_saved_model(model, args.model_path) # , use_avg_model=args.gen_avg_model)

        if args.guidance_param != 1:
            model = ClassifierFreeSampleModel(
                model)  # wrapping model with the classifier-free sampler
        model.to(dist_util.dev())
        model.eval()  # disable random masking
        ###################################

        evaluator = EvalNode(args.obj_model_path, args.sbj_model_path, args.mano_model_path)

        iterator_pre = iter(data)
        iterator = iter(data)

        motion_gt_grasp, model_kwargs_grasp = next(iterator_pre)
        texts_pre = model_kwargs_grasp['y']['text']

        motion_gt, model_kwargs = next(iterator)
        texts = model_kwargs['y']['text']

        for _ in range(i):
            motion_gt_grasp, model_kwargs_grasp = next(iterator_pre)
            texts_pre = model_kwargs_grasp['y']['text']

            motion_gt, model_kwargs = next(iterator)
            texts = model_kwargs['y']['text']

        for i in range(len(texts)):
            obj_name = [s for s in OBJECT_LIST_NEW if s in texts_pre[i]]
            obj_name_post = [s for s in OBJECT_LIST_NEW if s in texts[i]]
            assert len(obj_name) >= 1
            if len(obj_name) > 1:
                if 'phone' in obj_name:
                    obj_name = ['phone']
                else:
                    obj_name = ['wristwatch']
            assert len(obj_name) >= 1
            if len(obj_name) > 1:
                if 'phone' in obj_name:
                    obj_name = ['phone']
                else:
                    obj_name = ['wristwatch']

            texts_pre[i] = texts_pre[i].replace(obj_name[0], obj_name_post[0])


        model_kwargs_grasp['y']['text'] = texts_pre
        model_kwargs['y']['text'] = texts

        key_frames = [0, 49] ### As the pre-grasp data in GRAB is always 50 frames, we take the last frame

        # Transform into unnormalized space
        motion_gt = data.dataset.t2m_dataset.inv_transform_th(motion_gt.cpu().permute(0, 2, 3, 1)).float().permute(0, 1, 3, 2)
        motion_gt_kf = motion_gt[...,key_frames] 

        (target, target_mask,
            _, _,
            _, _,
            _, _,
            _, _, _) = get_grasp_references(motion_gt_kf,
                                            key_frames, data.dataset, data.dataset, max_len=n_frames,
                                            feat_dim_traj=motion_gt_grasp.shape[1], feat_dim_motion=117)
        
        # Name for logging
        model_kwargs_grasp['y']['grasp_model'] = args.traj_only
        model_kwargs['y']['grasp_model'] = args.traj_only

        model_device = next(model.parameters()).device

        motion_cond_until = 20
        motion_impute_until = 100

        target = target[...,:motion_gt_grasp.shape[1]].to(model_device)
        target_mask = target_mask[...,:motion_gt_grasp.shape[1]].to(model_device)
        model_kwargs['y']['target'] = target
        model_kwargs['y']['target_mask'] = target_mask

        ###########################################


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


        sample_fn = diffusion.p_sample_loop
        dump_steps = [999]

        if not args.do_inpaint and "inpainted_motion" in model_kwargs['y'].keys():
            del model_kwargs['y']['inpainted_motion']
            del model_kwargs['y']['inpainting_mask']

        for rep_i in range(args.num_repetitions):
            # Name for logging
            model_kwargs['y']['log_id'] = rep_i
            model_kwargs['y']['cond_until'] = motion_cond_until  # impute_slack
            model_kwargs['y']['impute_until'] = motion_impute_until # 20  # impute_slack
            # Pass functions to the diffusion
            diffusion.data_get_mean_fn = data.dataset.t2m_dataset.get_std_mean
            diffusion.data_transform_fn = data.dataset.t2m_dataset.transform_th
            diffusion.data_inv_transform_fn = data.dataset.t2m_dataset.inv_transform_th
            diffusion.log_trajectory_fn = None

            sample = sample_fn(
                model,
                (args.batch_size, model.njoints, model.nfeats, n_frames_post),
                clip_denoised=not args.predict_xstart,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step # NOTE: testing this
                init_image=None, # input_motions,  # init_image, # None, # NOTE: testing this
                progress=True,
                dump_steps=dump_steps,  # None,
                noise=None,
                const_noise=False,
                cond_fn=None,
            )


            gen_eff_len = min(sample[0].shape[-1], n_frames_post)
            print('cut the motion length to', gen_eff_len)
            for j in range(len(sample)):
                sample[j] = sample[j][:, :, :, :gen_eff_len]
            ###################

            num_dump_step = len(dump_steps)
            args.num_dump_step = num_dump_step

            cur_motions_pose_space, cur_motions, cur_lengths, cur_texts = sample_to_hand_motion(
                sample, args, model_kwargs, model, gen_eff_len,
                data.dataset.t2m_dataset.inv_transform)

            cur_motions_pose_space = np.array(cur_motions_pose_space)[0]
            positions_left, positions_right, global_orient_l, global_orient_r, _ = recover_from_ric(cur_motions_pose_space, object_rot_relative = False, add_obj_pos = False)

            cur_motions_pose_space[...,:pos_left_idcs[1]] = positions_left.swapaxes(1,2)
            cur_motions_pose_space[...,pos_right_idcs[0]:pos_right_idcs[1]] = positions_right.swapaxes(1,2)
            cur_motions_pose_space[...,glob_orient_l_idcs[0]:glob_orient_l_idcs[1]] = global_orient_l[:,np.newaxis]
            cur_motions_pose_space[...,glob_orient_r_idcs[0]:glob_orient_r_idcs[1]] = global_orient_r[:,np.newaxis]

            
            motions_full = cur_motions_pose_space
            all_motions.extend(motions_full[:,np.newaxis])
            all_lengths.extend(cur_lengths)
            all_text.extend(cur_texts)
            all_data_id.extend(model_kwargs['y']['data_id'])

            for j in range(len(cur_lengths[0])):
                frames_per_pose_input = np.array(cur_lengths[0])[j] // max_downsampled_frames
                idcs = np.arange(
                    0,  np.array(cur_lengths[0])[j], frames_per_pose_input
                )

                motions_full_ds = motions_full[j, 0, idcs, :6]
                motions_full_ds = motions_full_ds[:max_downsampled_frames]

                all_motions_downsampled.append(motions_full_ds)

            
            res = evaluator.evaluate_seqs(motions_full[:,:,49:], model_kwargs['y']['data_id'], cur_lengths, samples_kf=motion_gt_kf, eval_grasp_reference=args.eval_grasp_reference, eval_physics=args.physics_metrics, num_reps=num_reps)

            if args.eval_grasp_reference:
                handedness_accuracy.extend(res["handedness"])
                grasp_reference_error.extend(res["grasp_error"])
                t_vels.extend(res["t_vels"])

            if args.physics_metrics:
                all_iv.extend(res["inter_volume"])
                all_id.extend(res["inter_depth"])
                all_id_max.extend(res["inter_depth_max"])
                all_cr.extend(res["contact_ratio"])
                all_jp.extend(res["jerk_pos"])
                all_ja.extend(res["jerk_ang"])
                all_nc.extend(res["num_contact_frames"])
            all_gt_kf.extend(motion_gt_kf.swapaxes(2,3))

    ### Save videos
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

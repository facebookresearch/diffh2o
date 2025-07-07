# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import smplx
import trimesh
import os
import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation as R
from eval.metrics.metrics import ObjectContactMetrics
from pytorch3d.transforms.rotation_conversions import rotation_6d_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d
import pickle

OBJECT_LIST = ['cylinderlarge', 'mug', 'elephant', 'hand', 'cubelarge', 'stanfordbunny', 'airplane', 'alarmclock', 'banana', 'body', 'bowl', 'cubesmall', 'cup', 'doorknob', 'cubemedium',
             'eyeglasses', 'flashlight', 'flute', 'gamecontroller', 'hammer', 'headphones', 'knife', 'lightbulb', 'mouse', 'phone', 'piggybank', 'pyramidlarge', 'pyramidsmall', 'pyramidmedium',
             'duck', 'scissors', 'spherelarge', 'spheresmall', 'stamp', 'stapler', 'table', 'teapot', 'toruslarge', 'torussmall', 'train', 'watch', 'cylindersmall', 'waterbottle', 'torusmedium',
             'cylindermedium', 'spheremedium', 'wristwatch', 'wineglass', 'camera', 'binoculars', 'fryingpan', 'toothpaste', 'apple', 'toothbrush']

GENDER_MAP = {
    's1': "male",
    's2': "male",
    's3': "female",
    's4': "female",
    's5': "female",
    's6': "female",
    's7': "female",
    's8': "male",
    's9': "male",
    's10': "male"
}

special_decimation_objects = ['alarmclock', 'camera', 'gamecontroller', 'headphones', 'body', 'phone', 'wristwatch', 'watch', 'mouse', 'scissors', 'standfordbunny', 'teapot']

default_decimation_ratio = 0.01

special_decimation_ratios = {
    'alarmclock': 0.02,
    'body': 0.04,
    'camera': 0.02,
    'cup': 0.03,
    'gamecontroller': 0.04,
    'headphones': 0.04,
    'mouse': 0.02,
    'phone': 0.04,
    'scissors': 0.02,
    'standfordbunny': 0.02,
    'teapot': 0.04,
    'watch': 0.04,
    'wristwatch': 0.04,
}

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def compute_all_metrics(_vertices_sbj, _faces_sbj, _verts_obj, _faces_obj, obj_poses, start_id, end_id, fps=30, eval_contact_last_x_frames_only=False,
                        eval_contact_evenly_distributed=False, eval_contact_n_frames=5, device=None) -> dict:

    contact_collision_metric = ObjectContactMetrics(device)

    if eval_contact_last_x_frames_only:
        contact_collision_metric.compute_metrics(sbj_vertices=_vertices_sbj[-eval_contact_n_frames:],
                                                    sbj_faces=_faces_sbj,
                                                    obj_vertices=_verts_obj[-eval_contact_n_frames:], obj_faces=_faces_obj,
                                                    obj_poses=obj_poses.detach().cpu(), start_id=start_id, end_id=end_id)
    elif eval_contact_evenly_distributed:
        frames = np.linspace(start=0, stop=_verts_obj.shape[0] - 1, num=eval_contact_n_frames).round()
        contact_collision_metric.compute_metrics(sbj_vertices=_vertices_sbj[frames],
                                                    sbj_faces=_faces_sbj,
                                                    obj_vertices=_verts_obj[frames], obj_faces=_faces_obj,
                                                    obj_poses=obj_poses.detach().cpu(), start_id=start_id, end_id=end_id)
    else:
        contact_collision_metric.compute_metrics(sbj_vertices=_vertices_sbj.detach().cpu(), sbj_faces=_faces_sbj.detach().cpu(),
                                                    obj_vertices=_verts_obj.detach().cpu(), obj_faces=_faces_obj.detach().cpu(),
                                                    obj_poses=obj_poses.detach().cpu(), start_id=start_id, end_id=end_id)

    volume_pred = contact_collision_metric.volume
    depth_pred = contact_collision_metric.depth
    contact_ratio_pred = contact_collision_metric.contact_ratio
    jerk = contact_collision_metric.jerk
    return {**volume_pred, **depth_pred, **contact_ratio_pred, **jerk}

class EvalNode:

    def __init__(self, model_path):

        self.model_path = model_path

        self.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    def recover_from_ric(self, data, obj_rot, obj_r_pos, object_rot_relative = False, num_joints=15):
        #data = torch.tensor(data, dtype=torch.float32)
        obj_rot_quat = torch.tensor(R.from_matrix(obj_rot).as_quat(),dtype=torch.float32)
        positions_left = data[..., : num_joints * 3]
        positions_right = data[..., num_joints * 3: num_joints * 3 * 2]

        positions_left = positions_left.reshape(positions_left.shape[:-1] + (-1, 3))
        positions_right = positions_right.reshape(positions_right.shape[:-1] + (-1, 3))

        '''Add rotation to local joint positions'''
        if object_rot_relative:
            positions_left = np.matmul(obj_rot,positions_left.swapaxes(1,2)).swapaxes(1,2)
            positions_right = np.matmul(obj_rot,positions_right.swapaxes(1,2)).swapaxes(1,2)
        '''Add obj root to joints'''
        positions_left += obj_r_pos
        positions_right += obj_r_pos

        return torch.tensor(positions_left).to(self.device), torch.tensor(positions_right).to(self.device)

    def evaluate_seqs(self, sample_path):
        inter_volume = []
        inter_depth = []
        inter_depth_max = []
        contact_ratio = []
        jerk_pos = []
        jerk_ang = []
        num_contact_frames = []


        with open('dataset/MANO_SMPLX_vertex_ids.pkl', 'rb') as f:
            idxs_data = pickle.load(f)
            hand_idxs_left = idxs_data['left_hand']
            hand_idxs_right = idxs_data['right_hand']

        idxs_data = np.load('dataset/MANO_FACES.npy', allow_pickle=True)


        mano_faces_left = idxs_data.item()['mano_faces_left']
        mano_faces_right = idxs_data.item()['mano_faces_right']


        files = os.listdir(sample_path)
        for seq_name in files:
            print('seq name', seq_name)
            fp = os.path.join(sample_path, seq_name)
            seq_dict = torch.load(fp, map_location=torch.device('cpu'))
            seq_len = 15

            sbj_id = seq_name.split('_')[0]
            obj_name = seq_name.split('_')[1]
            intent_name = seq_name.split('_')[2]
            gender = GENDER_MAP[sbj_id]
            sbj_vtemp = trimesh.load(os.path.join(self.model_path, 'subject_templates', gender, sbj_id + '.ply'),process=False)


            self.smplx_model = smplx.create(self.model_path, model_type="smplx", use_pca=False,
                                    v_template=sbj_vtemp.vertices, flat_hand_mean=True,
                                    batch_size=seq_len).to(self.device)


            obj_mesh = trimesh.load(os.path.join(self.model_path,'object_meshes',obj_name+'.ply'), process=False)

            obj_verts_num = int(obj_mesh.vertices.shape[0] * 0.01)
            if obj_name in special_decimation_objects:
                obj_vert_num = int(obj_mesh.vertices.shape[0]*special_decimation_ratios[obj_name])

            obj_mesh = obj_mesh.simplify_quadratic_decimation(obj_verts_num)
            trimesh.repair.fix_normals(obj_mesh)


            obj_pose = seq_dict['obj_p']

            obj_trans = obj_pose['transl'].detach().numpy()  -   obj_pose['transl'].detach().numpy()[:1]
            obj_orient = R.from_rotvec(obj_pose['global_orient']).as_matrix()

            obj_verts = np.matmul(obj_mesh.vertices,obj_orient)
            obj_verts +=  obj_trans[:,np.newaxis]

            sbj_pose = seq_dict['sbj_p']
            sbj_trans = (sbj_pose['transl'] -   obj_pose['transl'][:1]).to(self.device)
            sbj_orient = torch.tensor(R.from_matrix(sbj_pose['global_orient'].detach().numpy().reshape(-1,3,3)).as_rotvec(),dtype=torch.float32).to(self.device)

            body_pose = torch.tensor(R.from_matrix(sbj_pose["body_pose"].detach().numpy().reshape(-1,3,3)).as_rotvec().reshape(-1,63),dtype=torch.float32).to(self.device)
            left_hand_pose = torch.tensor(R.from_matrix(sbj_pose["left_hand_pose"].detach().numpy().reshape(-1,3,3)).as_rotvec().reshape(-1,45),dtype=torch.float32).to(self.device)
            right_hand_pose = torch.tensor(R.from_matrix(sbj_pose["right_hand_pose"].detach().numpy().reshape(-1,3,3)).as_rotvec().reshape(-1,45),dtype=torch.float32).to(self.device)


            out_seq = self.smplx_model(
                    transl=sbj_trans,
                    global_orient=sbj_orient,
                    body_pose=body_pose,
                    left_hand_pose=left_hand_pose,
                    right_hand_pose=right_hand_pose,
                )


            verts_hand_l = out_seq.vertices[:,hand_idxs_left]
            verts_hand_r = out_seq.vertices[:,hand_idxs_right]
            faces_hand_l = torch.tensor(mano_faces_left.astype(np.int32)).to(self.device)
            faces_hand_r = torch.tensor(mano_faces_right.astype(np.int32)).to(self.device)
            obj_verts =  torch.tensor(obj_verts).to(self.device)

            verts_dict = {}
            verts_dict["verts_hand_l"] = verts_hand_l.detach().cpu().numpy()
            verts_dict["verts_hand_r"] = verts_hand_r.detach().cpu().numpy()
            verts_dict["verts_obj"] = obj_verts.cpu().numpy()
            verts_dict["obj_faces"] = obj_mesh.faces.astype(np.int32)
            verts_dict["faces_hand_l"] = faces_hand_l.detach().cpu().numpy().astype(np.int32)
            verts_dict["faces_hand_r"] = faces_hand_r.detach().cpu().numpy().astype(np.int32)

            np.save("verts_dict.npy",verts_dict)

            obj_poses = torch.cat((obj_pose['transl'],obj_pose['global_orient']),axis=-1).to(self.device)

            res_l = compute_all_metrics(verts_hand_l, faces_hand_l,obj_verts,
            torch.tensor(obj_mesh.faces.astype(np.int32)).to(self.device), obj_poses, 0, 15, device=self.device)
            res_r = compute_all_metrics(verts_hand_r, faces_hand_r, obj_verts,
            torch.tensor(obj_mesh.faces.astype(np.int32)).to(self.device), obj_poses,
            0, 15, device=self.device)

            inter_volume.append(np.mean([res_l['inter_volume_mean']+res_r['inter_volume_mean']])*1e6)
            inter_depth.append(np.mean([res_l['inter_depth_mean']+res_r['inter_depth_mean']])*1e3)
            inter_depth_max.append(np.max([res_l['inter_depth_max'],res_r['inter_depth_mean']])*1e3)
            contact_ratio.append(np.mean([res_l['contact_ratio_contact'],res_r['contact_ratio_contact']]))
            jerk_pos.append(res_l['jerk_pos'])
            jerk_ang.append(res_l['jerk_ang'])

            print('inter volume', np.mean([res_l['inter_volume_mean']+res_r['inter_volume_mean']])*1e6)
            print('inter depth', np.mean([res_l['inter_depth_mean']+res_r['inter_depth_mean']])*1e3)
            print('inter depth max', np.max([res_l['inter_depth_max'],res_r['inter_depth_max']])*1e3)
            print('contact ratio', np.mean([res_l['contact_ratio_mean'],res_r['contact_ratio_mean']]))

            contact_frames = np.unique(np.concatenate([res_l['contact_frames'],res_r['contact_frames']]))
            num_contact_frames.append(contact_frames.shape[0]/seq_len)
            print('num contact frames', contact_frames.shape[0]/seq_len)


        res_dict = {}
        res_dict['inter_volume'] = np.array(inter_volume)
        res_dict['inter_depth'] = np.array(inter_depth)
        res_dict['inter_depth_max'] = np.array(inter_depth)
        res_dict['contact_ratio'] = np.array(contact_ratio)
        res_dict['jerk_pos'] = np.array(jerk_pos)
        res_dict['jerk_ang'] = np.array(jerk_ang)
        res_dict['num_contact_frames'] = np.array(num_contact_frames)

        res_dict['inter_volume_mean'] = np.mean(inter_volume)
        res_dict['inter_depth_mean'] = np.mean(inter_depth)
        res_dict['contact_ratio_mean'] = np.mean(contact_ratio)
        res_dict['jerk_pos_mean'] = np.mean(jerk_pos)
        res_dict['jerk_ang_mean'] = np.mean(jerk_ang)
        res_dict['num_contact_frames_mean'] = np.mean(num_contact_frames)

        return res_dict


if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description="Script with command line arguments")
    parser.add_argument(
        "--mano_model_path",
        type=str,
        required=False,
        default='./assets',
    )
    parser.add_argument(
        "--file_path",
        type=str,
        required=True,
        help="The file path argument"
    )
    args = parser.parse_args()

    evaluator = EvalNode(args.mano_model_path)
    res = evaluator.evaluate_seqs(args.file_path)

    with open(os.path.join(args.file_path,'imos_scores.txt'), 'w') as fw:
        mean_values = {
            'iv': np.mean(res["inter_volume"]),
            'id': np.mean(res["inter_depth"]),
            'id_max': np.mean(res["inter_depth_max"]),
            'cr': np.mean(res["contact_ratio"]),
            'jp': np.mean(res["jerk_pos"]),
            'ja': np.mean(res["jerk_ang"]),
            'nc': np.mean(res["num_contact_frames"]),
        }

        for key, value in mean_values.items():
            fw.write(f'{key}: {value}\n')

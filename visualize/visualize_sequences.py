# Copyright (c) Meta Platforms, Inc. and affiliates.

# Python
import argparse
import os
import shutil

# Thirdparty
import aitviewer
import numpy as np
import scipy
import torch
import torch.nn.functional as F
import trimesh
import trimesh.registration
from aitviewer.headless import HeadlessRenderer
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer
from pytorch3d.transforms.rotation_conversions import rotation_6d_to_matrix
from scipy.spatial.transform import Rotation as R
from vis_utils import GENDER_MAP, make_mano_sequence_wt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



hand_r_mesh_color = (121 / 255.0, 119 / 255.0, 158 / 255.0, 1.0)
hand_l_mesh_color = (158 / 255.0, 121 / 255.0, 119 / 255.0, 1.0)
obj_mesh_color = (121 / 255.0, 140 / 255.0, 119 / 255.0, 1.0)
keyframe_color = (50 / 255, 131 / 255, 131 / 255, 206 / 255)
pc_hand_color = (200 / 255, 85 / 255, 85 / 255, 0.6)



def render_sequence(
    pose_data_path,
    keyframe_vis=False,
    pre_grasp=False,
    save_video=False,
    vis_gt=False,
    num_reps=1,
    resolution='high',
    range_min=0,
    range_max=20,
):
    assert range_max > range_min, "range_max must be greater than range_min"

    if resolution == 'high':
        resolution = (4000, 3000)
    elif resolution == 'medium':
        resolution = (2000, 1500)
    else:
        resolution = (500, 375)

    # Initialize Viewer
    v = (
            HeadlessRenderer(size=resolution) if save_video else Viewer(size=resolution)
        )

    ### Camera settings
    v.scene.camera.fov = 40
    v.scene.camera.target = [0.0, 0.218, -0.093]
    v.scene.camera.position = [0.0, 0.65, 2.152]
    v.scene.camera.up = [0, 1.0, 0]
    v.scene.floor.position = [0.0, -0.4, 0.0]
    ### Further adjustments
    v.auto_set_camera_target = False
    v.auto_set_floor = False
    v.scene.remove(v.scene.lights[0])
    v.scene.remove(v.scene.origin)


    if not vis_gt:
        text_file = pose_data_path.replace('.npy', '.txt')
        len_file = pose_data_path.replace('.npy', '_len.txt')
        feature_vec_in = (
            np.load(pose_data_path, allow_pickle=True)
        )

        # Read in text prompts
        with open(text_file, "r") as f:
            texts = np.asarray([f.read().splitlines()])[0]

        # Read in sequence lengths
        with open(len_file, "r") as f:
            lengths = np.asarray([f.read().splitlines()])[0]
 
        range_max = min(len(texts), range_max)

    id_dict = {}
    with open("dataset/file_names.txt", "r") as file:
        for line in file:
            line = line.strip()
            value, key = line.split(",", 1)
            if vis_gt:
                id_dict[value] = key
            else:
                id_dict[key] = value


    count = 0

    for rep in range(num_reps):
        for i in range(range_min // num_reps, range_max // num_reps):
            if vis_gt:
                seq_id = str(i).zfill(6)
                seq_name = id_dict[seq_id]
                feature_vec_in = np.load(os.path.join(pose_data_path, seq_id + ".npy"), allow_pickle=True)
            else:
                seq_name = feature_vec_in.item()['data_id'][i]
                seq_id = id_dict[seq_name]

            sbj_id = seq_name.split('_')[0]
            obj_name = seq_name.split('_')[1]
            intent_name = seq_name.split('_')[2]
            data = [[seq_id,sbj_id, obj_name, intent_name]]

            sbj_vtemp_lhand = trimesh.load("assets/" + GENDER_MAP[sbj_id] + "/" + sbj_id + "_lhand.ply")
            sbj_vtemp_rhand = trimesh.load("assets/" + GENDER_MAP[sbj_id] + "/" + sbj_id + "_rhand.ply")

            # Load MANO hand models
            smpl_layer_lhand = SMPLLayer(
                model_type="mano",
                use_pca=args.is_pca,
                v_template=sbj_vtemp_lhand.vertices,
                flat_hand_mean=True,
                is_rhand=False,
                num_pca_comps=24,
            )

            smpl_layer_rhand = SMPLLayer(
                model_type="mano",
                use_pca=args.is_pca,
                v_template=sbj_vtemp_rhand.vertices,
                flat_hand_mean=True,
                is_rhand=True,
                num_pca_comps=24,
            )

            if vis_gt:
                feature_vec = feature_vec_in
            else:
                feature_vec = feature_vec_in.item()["motion"][
                    rep * (range_max-range_min) // num_reps + i
                ][
                    0
                ]

            obj_mesh = trimesh.load(
                "assets/contact_meshes/"
                + obj_name
                + ".ply"
            )

            # Get object pose
            obj_verts = obj_mesh.vertices
            obj_rot = rotation_6d_to_matrix(torch.tensor(feature_vec[:, -6:])).reshape(
                -1, 3, 3
            ).numpy()

            # Retrieve hand positions
            pos_left = torch.tensor(feature_vec[:, :3], dtype=torch.float32).to(device)
            pos_right = torch.tensor(feature_vec[:, 3:6], dtype=torch.float32).to(device)


            # Retrieve global hand rotations
            global_orient_l = R.from_matrix(
                rotation_6d_to_matrix(torch.tensor(feature_vec[:, 30:36])).numpy()
            ).as_rotvec()
            global_orient_r =  R.from_matrix(
                rotation_6d_to_matrix(torch.tensor(feature_vec[:, 60:66])).numpy()
            ).as_rotvec()

            # Retrieve local hand rotations
            joint_rotations_l = feature_vec[:, 6:30]
            joint_rotations_r = feature_vec[:, 36:60]

            # Translate sequence to avoid overlap
            if not save_video:
                pos_left[..., 0] += count * 2.0
                pos_right[..., 0] += count * 2.0

            # Shift by the MANO zero offset for visualization

            trans_l = (
                pos_left
                - smpl_layer_lhand.bm(hand_pose=torch.zeros((1, 24)).to(device)).joints[0, 0]
            ).detach().cpu().numpy()
            trans_r = (
                pos_right
                - smpl_layer_rhand.bm(hand_pose=torch.zeros((1, 24)).to(device))
                .joints[0, 0]
            ).detach().cpu().numpy()

            # Create MANO sequences
            seq_l = SMPLSequence(
                joint_rotations_l,
                smpl_layer_lhand,
                trans=trans_l,
                poses_root=global_orient_l,
                z_up=True,
                is_rigged=False,
            )
            seq_r = SMPLSequence(
                joint_rotations_r,
                smpl_layer_rhand,
                trans=trans_r,
                poses_root=global_orient_r,
                z_up=True,
                is_rigged=False,
            )

            # Create object sequences
            if pre_grasp:
                obj_verts = torch.matmul(
                    torch.tensor(obj_verts, dtype=torch.float64), obj_rot[-1:]
                ).numpy()
            else:
                obj_verts = np.matmul(obj_verts, obj_rot)
                obj_verts += feature_vec[:, np.newaxis, -9:-6]

            if not save_video:
                obj_verts[..., 0] += count * 2.0
                obj_verts[..., 1] += rep * 2.0

            # Make hands watertight for nicer visualization
            (
                verts_hand_l,
                verts_hand_r,
                faces_hand_l,
                faces_hand_r,
            ) = make_mano_sequence_wt(
                seq_l.vertices, seq_r.vertices, seq_l.faces, seq_r.faces
            )

            ### Collect Meshes to render
            mesh_frame = Meshes(
                obj_verts, obj_mesh.faces, z_up=True, color=obj_mesh_color
            )

            seq_l_mesh = Meshes(
                verts_hand_l, faces_hand_l, z_up=True, color=hand_l_mesh_color
            )
            seq_r_mesh = Meshes(
                verts_hand_r, faces_hand_r, z_up=True, color=hand_r_mesh_color
            )

            ### Add object and hand meshes to the scene
            v.scene.add(mesh_frame, seq_l_mesh, seq_r_mesh)

            ### Visualization of grasp keyframes
            if keyframe_vis:
                feature_vec_kf = feature_vec_in.item()["gt_kf"][
                    rep * (range_max-range_min) // num_reps + i
                ]
                obj_rot_kf = rotation_6d_to_matrix(
                    torch.tensor(feature_vec_kf[:, -6:])
                ).reshape(-1, 3, 3)
                obj_verts_kf = obj_mesh.vertices
                obj_verts_kf = np.tile(
                    obj_mesh.vertices, (feature_vec_kf.shape[0], 1, 1)
                )

                pos_left_kf = feature_vec_kf[:, np.newaxis, :3]
                pos_right_kf = feature_vec_kf[:, np.newaxis, 3:6]

                if not save_video:
                    pos_left_kf[..., 0] += count * 2.0
                    pos_right_kf[..., 0] += count * 2.0
                    pos_left_kf[..., 1] += rep * 2.0
                    pos_right_kf[..., 1] += rep * 2.0

                global_orient_l_kf = R.from_matrix(
                    rotation_6d_to_matrix(
                        torch.tensor(feature_vec_kf[:, 30:36])
                    ).numpy()
                ).as_rotvec()
                global_orient_r_kf = R.from_matrix(
                    rotation_6d_to_matrix(
                        torch.tensor(feature_vec_kf[:, 60:66])
                    ).numpy()
                ).as_rotvec()

                joint_rotations_l_kf = feature_vec_kf[:, 6:30]
                joint_rotations_r_kf = feature_vec_kf[:, 36:60]

                trans_l_kf = (
                    pos_left_kf[:, 0]
                    - smpl_layer_lhand.bm(hand_pose=torch.zeros((1, 24)).to(device))
                    .joints[0, 0]
                    .detach()
                    .cpu()
                    .numpy()
                )
                trans_r_kf = (
                    pos_right_kf[:, 0]
                    - smpl_layer_rhand.bm(hand_pose=torch.zeros((1, 24)).to(device))
                    .joints[0, 0]
                    .detach()
                    .cpu()
                    .numpy()
                )


                seq_l_kf = SMPLSequence(
                    joint_rotations_l_kf[-1:],
                    smpl_layer_lhand,
                    trans=trans_l_kf[-1:],
                    poses_root=global_orient_l_kf[-1:],
                    z_up=True,
                    is_rigged=False,
                    color=keyframe_color,
                )
                seq_r_kf = SMPLSequence(
                    joint_rotations_r_kf[-1:],
                    smpl_layer_rhand,
                    trans=trans_r_kf[-1:],
                    poses_root=global_orient_r_kf[-1:],
                    z_up=True,
                    is_rigged=False,
                    color=keyframe_color,
                )


                obj_verts_kf = torch.matmul(
                    torch.tensor(obj_verts_kf, dtype=torch.float32), obj_rot_kf
                ).numpy()

                if not save_video:
                    obj_verts_kf[..., 0] += count * 2.0
                    obj_verts_kf[..., 1] += rep * 2.0

                obj_verts_kf += feature_vec_kf[:, np.newaxis, -9:-6]
                (
                    verts_hand_l_kf,
                    verts_hand_r_kf,
                    faces_hand_l_kf,
                    faces_hand_r_kf,
                ) = make_mano_sequence_wt(
                    seq_l_kf.vertices, seq_r_kf.vertices, seq_l_kf.faces, seq_r_kf.faces
                )

                seq_l_kf_mesh = Meshes(
                    verts_hand_l_kf, faces_hand_l, z_up=True, color=keyframe_color
                )
                seq_r_kf_mesh = Meshes(
                    verts_hand_r_kf, faces_hand_r, z_up=True, color=keyframe_color
                )

                mesh_frame_kf = Meshes(
                    obj_verts_kf[-1], obj_mesh.faces, z_up=True, color=keyframe_color
                )

                ### Add keyframe meshes to scene
                v.scene.add(seq_r_kf_mesh, seq_l_kf_mesh, mesh_frame_kf)


            if save_video:
                v.run()
                v.save_video(
                    video_dir="/".join(pose_data_path.split("/")[:-1])
                    + "/ours_videos/"
                    + str(count).zfill(4)
                    + ".mp4",
                    frame_dir="/".join(pose_data_path.split("/")[:-1])
                    + "/ours_videos/"
                    + str(count).zfill(4),
                    quality="high",
                    output_fps=30,
                )


                nodes = v.scene.nodes.copy()
                for node in nodes:
                    if node.name == "Meshes":
                        v.scene.remove(node)
            ### Count the actual number of sequences (if some are skipped)
            count += 1

    if not save_video:
        v.run()

        for k in range(num_reps):
            shutil.rmtree(os.path.join(pose_data_path.split(".")[0], str(k).zfill(4)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script with command line arguments")

    parser.add_argument(
        "--file_path", type=str, required=True, help="The file path argument"
    )
    parser.add_argument("--is_pca", action="store_true", default=True, help="If PCA representation is used for the pose representation")
    parser.add_argument("--pre_grasp", action="store_true", default=False, help="If only the pre-grasp pose is used for the visualization")
    parser.add_argument("--kf_vis", action="store_true", default=False, help="If the keyframes should be visualized")
    parser.add_argument("--vis_gt", action="store_true", default=False, help="If the GT poses should be visualized")
    parser.add_argument("--save_video", action="store_true", default=False, help="If a video should be saved instead of rendering in interactive mode")
    parser.add_argument("--num_reps", default=1, type=int, help="If multiple samples were generated for the same text prompt.")
    parser.add_argument("--range_min", default=0, type=int, help="The index of the first sequence to be visualized")
    parser.add_argument("--range_max", default=20, type=int, help="The index of the last sequence to be visualized")
    parser.add_argument("--resolution", default="high", type=str, help="The resolution of the visualization. Either 'high', 'medium', or 'low'")

    args = parser.parse_args()

    render_sequence(
        args.file_path,
        args.kf_vis,
        args.pre_grasp,
        args.save_video,
        args.vis_gt,
        args.num_reps,
        args.resolution,
        args.range_min,
        args.range_max,
    )

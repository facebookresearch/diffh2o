import os
import numpy as np
import torch
import scipy


from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

def process_files(folder_path, indices, seq_len=50):

    for idx, filename in enumerate(sorted(os.listdir(folder_path))):
        if filename.endswith(".npy"):
            file_path = os.path.join(folder_path, filename)
            data = np.load(file_path)
            # Check if the shape of the data is as expected

            if data.shape[1] != 117:
                print(f"Skipping file {filename} due to unexpected shape.")
                continue

            new_data = np.zeros((data.shape[0], 117))

            new_data[:,6:30] = data[:,6:30]
            new_data[:,36:60] = data[:,36:60]
            new_data[:,66:108] = data[:,66:108]

            ### Object normalization
            obj_pos = data[:, 108:111]
            obj_rot = rotation_6d_to_matrix(torch.tensor(data[:, 111:],dtype=torch.float32)).numpy()

            obj_pos_new = obj_pos - data[-1:, 108:111]

            obj_rot_aa = R.from_matrix(obj_rot).as_rotvec()

            ### Finger position normalization
            positions_left = np.matmul(obj_rot,data[:, :3, np.newaxis])[...,0]
            positions_right = np.matmul(obj_rot,data[:, 3:6, np.newaxis])[...,0]

            finger_pos_l = positions_left+obj_pos
            finger_pos_r = positions_right+obj_pos

            finger_pos_l -= data[-1:, 108:111]
            finger_pos_r -= data[-1:, 108:111]

            global_orient_l = rotation_6d_to_matrix(torch.tensor(data[:,30:36].copy()))
            global_orient_r = rotation_6d_to_matrix(torch.tensor(data[:,60:66].copy()))

            global_orient_l_mat = torch.matmul(torch.tensor(obj_rot,dtype=torch.float64),global_orient_l)
            global_orient_r_mat = torch.matmul(torch.tensor(obj_rot,dtype=torch.float64),global_orient_r)

            global_orient_l = matrix_to_rotation_6d(global_orient_l_mat).numpy()
            global_orient_r = matrix_to_rotation_6d(global_orient_r_mat).numpy()

            global_orient_l_aa = R.from_matrix(global_orient_l_mat.numpy()).as_rotvec()
            global_orient_r_aa = R.from_matrix(global_orient_r_mat.numpy()).as_rotvec()

            new_data[:,:3] = finger_pos_l
            new_data[:,3:6] = finger_pos_r
            new_data[:,30:36] = global_orient_l
            new_data[:,60:66] = global_orient_r
            new_data[:,108:111] = obj_pos_new
                
            if new_data.shape[0] > seq_len:
                pos_left_init_dist = np.linalg.norm(finger_pos_l[:1,:] - finger_pos_l, axis=-1)>0.1
                pos_right_init_dist = np.linalg.norm(finger_pos_r[:1,:] - finger_pos_r, axis=-1)>0.1
                start_idx = max(np.where(pos_left_init_dist * pos_right_init_dist)[0][0]-5, 0)

                new_data = new_data[start_idx:,:]
                global_orient_l_aa = global_orient_l_aa[start_idx:,:]
                global_orient_r_aa = global_orient_r_aa[start_idx:,:]
                obj_rot_aa = obj_rot_aa[start_idx:,:]

            original_length = new_data.shape[0]
            new_length = seq_len

            original_indices = np.linspace(0, original_length - 1, num=original_length)
            new_indices = np.linspace(0, original_length - 1, num=new_length)

            # Interpolate each dimension
            interpolated_data = np.zeros((new_length, new_data.shape[1]))
            interpolated_orient_l = np.zeros((new_length, 3))
            interpolated_orient_r = np.zeros((new_length, 3))
            interpolated_obj = np.zeros((new_length, 3))

            for i in range(new_data.shape[1]):
                interp_func = interp1d(original_indices, new_data[:, i], kind='linear')
                interpolated_data[:, i] = interp_func(new_indices)


            for i in range(interpolated_orient_l.shape[1]):
                interp_func_l = interp1d(original_indices, global_orient_l_aa[:, i], kind='linear')
                interp_func_r = interp1d(original_indices, global_orient_r_aa[:, i], kind='linear')
                interp_func_obj = interp1d(original_indices, obj_rot_aa[:, i], kind='linear')
                interpolated_orient_l[:, i] = interp_func_l(new_indices)
                interpolated_orient_r[:, i] = interp_func_r(new_indices)
                interpolated_obj[:,i] = interp_func_obj(new_indices)

            new_data = interpolated_data

            new_data[:,30:36] = matrix_to_rotation_6d(torch.tensor(R.from_rotvec(interpolated_orient_l).as_matrix())).numpy()
            new_data[:,60:66] = matrix_to_rotation_6d(torch.tensor(R.from_rotvec(interpolated_orient_r).as_matrix())).numpy()
            new_data[:,-6:] = matrix_to_rotation_6d(torch.tensor(R.from_rotvec(interpolated_obj).as_matrix())).numpy()
          
            out_path = os.path.join(folder_path+"_interp", filename)
            np.save(out_path, new_data)
            print(f"Processed and saved {filename}")

folder_path = 'dataset/GRAB_HANDS/joint_vecs_interpolated'  # Replace with your folder path
indices = np.load('separating_idcs.npy', allow_pickle=True)
process_files(folder_path, indices)

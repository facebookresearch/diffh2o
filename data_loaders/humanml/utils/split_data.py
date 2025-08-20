import os
import numpy as np

def process_files(folder_path, folder_path_contacts):
    separating_idcs = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".npy"):
            print(f"Processing {filename}")
            file_path = os.path.join(folder_path, filename)
            if filename.startswith("M"):
                file_path_contacts = os.path.join(folder_path_contacts, filename.split("M")[-1])
            else:
                file_path_contacts = os.path.join(folder_path_contacts, filename)
            data = np.load(file_path)
            data_contacts = np.load(file_path_contacts)
            contact_array = data_contacts[:,-99:-57]
            transition_idx = find_transition(data, contact_array, 0.01)
            trim_start_idx = trim_start(data, 0.3)
            trim_end_idx = trim_end(data, contact_array, 0.02)
            data_pre = data[:transition_idx]
            data_post = data[transition_idx:trim_end_idx]

            separating_idcs.append([transition_idx, trim_end_idx])
            file_out_path_pre = os.path.join(folder_path+"_pre_full", filename)
            #file_out_path_post = os.path.join(folder_path+"_post", filename)
            np.save(file_out_path_pre, data_pre)

            print(f"Processed and saved {filename}")
            print(f"Len {transition_idx}")

    np.save('separating_idcs.npy', np.array(separating_idcs))

def find_transition(sequence, contact_array, dist_threshold, min_num_contacts=6):
    start_cond_vert = sequence[:,-7] > dist_threshold ## vertical distance
    start_cond_hor = np.linalg.norm(sequence[:,-9:-7],axis=-1) > dist_threshold ## lateral distance
    start_cond_contact = np.sum(contact_array, axis=1)>min_num_contacts
    try:
        threshold_idx = np.where(start_cond_vert+start_cond_hor+start_cond_contact) ## if there are no detected contacts, use positions only
        return threshold_idx[0][0]
    except:
        start_cond_contact = np.sum(contact_array, axis=1)>2
        threshold_idx = np.where(start_cond_vert+start_cond_hor+start_cond_contact)
        return threshold_idx[0][0]

def trim_end(sequence, contact_array, stop_movement_threshold):
    end_cond_contact = np.sum(contact_array, axis=1) > 0
    threshold_idx = np.where(end_cond_contact)

    if threshold_idx[0].size > 0:
        return threshold_idx[0][-1]
    else:
        return sequence.shape[0]-1


folder_path = 'dataset/GRAB_HANDS/joint_vecs_grab'  # Replace with your folder path
folder_path_contacts = 'dataset/GRAB_HANDS/joint_vecs_grab_contacts'  # Replace with your folder path
process_files(folder_path, folder_path_contacts)

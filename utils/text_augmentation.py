# Copyright (c) Meta Platforms, Inc. and affiliates.

import csv
import os

def read_csv_to_dict(csv_file):
    """Reads a CSV file and converts it to a hierarchical dictionary."""
    hierarchy_dict = {}
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            key = row.pop(reader.fieldnames[0])
            hierarchy_dict[key] = row
            hierarchy_dict['M' + key] = row.copy()
    return hierarchy_dict

def swap_left_right(sentence):
    """Swaps 'left' with 'right' and vice versa in a sentence."""
    sentence = sentence.replace('left', 'tempLeft').replace('right', 'left').replace('tempLeft', 'right')
    return sentence

def ensure_period(sentence):
    """Ensures the sentence ends with a period."""
    return sentence if sentence.endswith('.') else sentence + '.'

def append_sentence_to_files(folder_path, hierarchy_dict):
    """Appends sentences to files based on the dictionary."""
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            print('file', file_name)
            key = os.path.splitext(file_name)[0]
            if key in hierarchy_dict and 'Hand' in hierarchy_dict[key]:
                sentence = hierarchy_dict[key]['Hand']
                # if sentence:  # Only proceed if the sentence is not empty
                    # sentence = ensure_period(sentence)
                    # if file_name.startswith('M'):
                    #     sentence = swap_left_right(sentence)
                file_path = os.path.join('dataset/GRAB_HANDS/hand_labels', file_name)
                    # with open(file_path, 'a', encoding='utf-8') as file:
                    #     file.write('\n' + sentence +"##0.0#0.0")
                if sentence == 'both':
                    label = 2
                elif sentence == 'left':
                    if file_name.startswith('M'):
                        label = 1
                    else:
                        label = 0
                else:
                    if file_name.startswith('M'):
                        label = 0
                    else:
                        label = 1

                with open(file_path, 'a', encoding='utf-8') as file:
                    file.write(str(label))

# Usage
csv_file = 'dataset/text_augmentations_label.csv' # Replace with your CSV file path
folder_path = 'dataset/GRAB_HANDS/texts' # Replace with your folder path

# Read CSV file and convert to dictionary
hierarchy_dict = read_csv_to_dict(csv_file)

# Append sentences to text files
append_sentence_to_files(folder_path, hierarchy_dict)

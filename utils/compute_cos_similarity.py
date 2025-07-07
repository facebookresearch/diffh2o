# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
import argparse
# Parse command-line arguments
parser = argparse.ArgumentParser(description='Compute cosine similarity between texts in two folders')
parser.add_argument('train_folder', type=str, help='Path to folder containing training text files')
parser.add_argument('test_folder', type=str, help='Path to folder containing test text files')
parser.add_argument('file_list_train', type=str, help='Path to file containing list of files to use for training and testing')
parser.add_argument('file_list_test', type=str, help='Path to file containing list of files to use for training and testing')
args = parser.parse_args()
# Read the list of files to use for training and testing from the file
with open(args.file_list_train, 'r') as f:
    file_list_train = [line.strip() for line in f]
with open(args.file_list_test, 'r') as f:
    file_list_test = [line.strip() for line in f]
# Read the training and test texts from the files in the folders
train_texts = []
test_texts = []

for file_name in file_list_train:
    lines = []
    file_path = f"{args.train_folder}/{file_name}.txt"
    with open(file_path, 'r') as f:
        lines = f.readlines()[-1].split('#')[0]

    train_texts += [lines]
for file_name in file_list_test:
    lines = []
    file_path = f"{args.test_folder}/{file_name}.txt"
    with open(file_path, 'r') as f:
        lines = f.readlines()[-1].split('#')[0]

    test_texts += [lines]



# Convert the texts into vectors using a CountVectorizer
vectorizer = CountVectorizer(lowercase=True).fit(train_texts + test_texts)
X_train = vectorizer.transform(train_texts).toarray()
X_test = vectorizer.transform(test_texts).toarray()
# Compute the cosine similarity between the training and test vectors
similarity = cosine_similarity(X_train, X_test)
#jac_similarity = jaccard_score(X_train, X_test)
# Compute the mean cosine similarity

mean_similarity = np.mean(similarity)
#mean_similarity_jac = np.mean(jac_similarity)
print("Mean cosine similarity:", mean_similarity)
#print("Mean Jaccard similarity:", mean_similarity_jac)

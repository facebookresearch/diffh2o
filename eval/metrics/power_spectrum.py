"""
Original source: https://github.com/eth-ait/motion-transformer/blob/master/metrics/distribution_metrics.py
from the paper https://arxiv.org/pdf/2004.08692.pdf.
"""

import numpy as np


def split_into_equal_len_chunks_list(seq_list: list, chunk_len=10):
    """
    https://github.com/eth-ait/motion-transformer/blob/c134b266d2cdcc247a169c5369e3b3a747e65ee3/spl/evaluation_dist_metrics_amass.py#L379
    :param seq_list: [batch_size, (n_steps, n_joints, n_features)]
    :param chunk_len:
    :return:
    """
    chunks = []
    for seq in seq_list:
        for i in range(0, seq.shape[0] - chunk_len + 1, chunk_len):
            chunks.append(seq[i:i + chunk_len])

    chunks = np.stack(chunks, axis=0)

    return chunks


def extract_chunk_stack_list(seq_list: list, frame, chunk_len):
    """
    :param seq_list: [batch_size, (n_steps, n_joints, n_features)]
    :param frame: start frame id
    :param chunk_len: chunk length to extract
    :return:
    """
    chunks = []
    for seq in seq_list:

        extracted_chunk = seq[frame:frame + chunk_len]
        if extracted_chunk.shape[0] < chunk_len:
            continue
        chunks.append(extracted_chunk)

    chunks = np.stack(chunks, axis=0)

    return chunks


def split_into_equal_len_chunks(a, chunk_len=10):
    """
    https://github.com/eth-ait/motion-transformer/blob/c134b266d2cdcc247a169c5369e3b3a747e65ee3/spl/evaluation_dist_metrics_amass.py#L379
    :param a: (batch_size, n_steps, n_joints, n_features)
    :param chunk_len:
    :return:
    """
    chunks = []
    for i in range(0, a.shape[1] - chunk_len + 1, chunk_len):
        chunks.append(a[:, i:i + chunk_len])

    chunks = np.vstack(chunks)

    return chunks


def power_spectrum(seq, filter_dims=False):
    """
    # seq = seq[:, :, 0:-1:12, :]  # 5 fps for amass (in 60 fps)

    Args:
      seq: (batch_size, n_joints, seq_len, feature_size)

    Returns:
        (1, seq_len/frequency bins, n_joints * feature_size)
    """
    feature_size = seq.shape[-1]
    n_joints = seq.shape[1]

    seq_t = np.transpose(seq, [0, 2, 1, 3])  # (batch_size, seq_len, n_joints, feature_size)
    if filter_dims:
        dims_to_use = np.where((np.reshape(seq_t, [-1, n_joints, feature_size]).std(0) >= 1e-4).all(axis=-1))[0]
        seq_t = seq_t[:, :, dims_to_use]

    seq_t = np.reshape(seq_t, [seq_t.shape[0], seq_t.shape[1], 1, -1])  # (batch_size, seq_len, 1, n_joints * feature_size)
    seq = np.transpose(seq_t, [0, 2, 1, 3])  # (batch_size, 1, seq_len, n_joints * feature_size)

    seq_fft = np.fft.fft(seq, axis=2)
    seq_ps = np.abs(seq_fft) ** 2

    seq_ps_global = seq_ps.sum(axis=0) + 1e-8  # (1, seq_len, n_joints * feature_size)
    seq_ps_global /= seq_ps_global.sum(axis=1, keepdims=True)

    return seq_ps_global


def ps_entropy(seq_ps):
    """

    Args:
        seq_ps: (1, seq_len, n_joints * feature_size)
    Returns:
    """
    return -np.sum(seq_ps * np.log(seq_ps), axis=1)


def ps_kld(seq_ps_from, seq_ps_to):
    """ Calculates KL(seq_ps_from, seq_ps_to).
    Args:
        seq_ps_from:
        seq_ps_to:
    Returns:
    """
    return np.sum(seq_ps_from * np.log(seq_ps_from / seq_ps_to), axis=1)


def compute_npss(euler_gt_sequences, euler_pred_sequences):
    """
    Computing normalized Normalized Power Spectrum Similarity (NPSS)
    Taken from @github.com neural_temporal_models/blob/master/metrics.py#L51

    1) fourier coeffs
    2) power of fft
    3) normalizing power of fft dim-wise
    4) cumsum over freq.
    5) EMD

    Args:
        euler_gt_sequences:
        euler_pred_sequences:
    Returns:
    """
    gt_fourier_coeffs = np.zeros(euler_gt_sequences.shape)
    pred_fourier_coeffs = np.zeros(euler_pred_sequences.shape)

    # power vars
    gt_power = np.zeros((gt_fourier_coeffs.shape))
    pred_power = np.zeros((gt_fourier_coeffs.shape))

    # normalizing power vars
    gt_norm_power = np.zeros(gt_fourier_coeffs.shape)
    pred_norm_power = np.zeros(gt_fourier_coeffs.shape)

    cdf_gt_power = np.zeros(gt_norm_power.shape)
    cdf_pred_power = np.zeros(pred_norm_power.shape)

    emd = np.zeros(cdf_pred_power.shape[0:3:2])

    # used to store powers of feature_dims and sequences used for avg later
    seq_feature_power = np.zeros(euler_gt_sequences.shape[0:3:2])
    power_weighted_emd = 0

    for s in range(euler_gt_sequences.shape[0]):

        for d in range(euler_gt_sequences.shape[2]):
            gt_fourier_coeffs[s, :, d] = np.fft.fft(
                euler_gt_sequences[s, :, d])  # slice is 1D array
            pred_fourier_coeffs[s, :, d] = np.fft.fft(
                euler_pred_sequences[s, :, d])

            # computing power of fft per sequence per dim
            gt_power[s, :, d] = np.square(
                np.absolute(gt_fourier_coeffs[s, :, d]))
            pred_power[s, :, d] = np.square(
                np.absolute(pred_fourier_coeffs[s, :, d]))

            # matching power of gt and pred sequences
            gt_total_power = np.sum(gt_power[s, :, d])
            pred_total_power = np.sum(pred_power[s, :, d])
            # power_diff = gt_total_power - pred_total_power

            # adding power diff to zero freq of pred seq
            # pred_power[s,0,d] = pred_power[s,0,d] + power_diff

            # computing seq_power and feature_dims power
            seq_feature_power[s, d] = gt_total_power

            # normalizing power per sequence per dim
            if gt_total_power != 0:
                gt_norm_power[s, :, d] = gt_power[s, :, d] / gt_total_power

            if pred_total_power != 0:
                pred_norm_power[s, :, d] = pred_power[s, :, d] / pred_total_power

            # computing cumsum over freq
            cdf_gt_power[s, :, d] = np.cumsum(gt_norm_power[s, :, d])  # slice is 1D
            cdf_pred_power[s, :, d] = np.cumsum(pred_norm_power[s, :, d])

            # computing EMD
            emd[s, d] = np.linalg.norm((cdf_pred_power[s, :, d] - cdf_gt_power[s, :, d]), ord=1)

    # computing weighted emd (by sequence and feature powers)
    power_weighted_emd = np.average(emd, weights=seq_feature_power)

    return power_weighted_emd

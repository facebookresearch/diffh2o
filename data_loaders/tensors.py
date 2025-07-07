# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch

def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]

    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]


    databatchTensor = collate_tensors(databatch)

    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({'text': textbatch})

    if 'object_bps' in notnone_batches[0]:
        object_bps_batch = [b['object_bps'] for b in notnone_batches]
        cond['y'].update({'object_bps': object_bps_batch})

    if 'motion_enc_frames' in notnone_batches[0] and notnone_batches[0]['motion_enc_frames']>0:
        databatch_full = [b['motion_full'] for b in notnone_batches]
        motion_full = collate_tensors(databatch_full)
        motion_enc = motion_full[...,:notnone_batches[0]['motion_enc_frames']]

        cond['y'].update({'motion_enc_gt': motion_enc})

    if 'tokens' in notnone_batches[0]:
        textbatch = [b['tokens'] for b in notnone_batches]
        cond['y'].update({'tokens': textbatch})

    if 'action' in notnone_batches[0]:
        actionbatch = [b['action'] for b in notnone_batches]
        cond['y'].update({'action': torch.as_tensor(actionbatch).unsqueeze(1)})

    # collate action textual names
    if 'action_text' in notnone_batches[0]:
        action_text = [b['action_text']for b in notnone_batches]
        cond['y'].update({'action_text': action_text})

    if 'data_id' in notnone_batches[0]:
        data_ids = [b['data_id'] for b in notnone_batches]
        cond['y'].update({'data_id': data_ids})

    return motion, cond

# an adapter to our collate func
def t2m_collate(batch):

    adapted_batch = [{
        'inp': torch.tensor(b[4].T).float().unsqueeze(1),
        'text': b[2],
        'tokens': b[6],
        'lengths': b[5],
        'object_bps': b[7],
        'motion_enc_frames': b[8],
        'motion_full': torch.tensor(b[9].T).float().unsqueeze(1),
        'data_id': b[10],
    } for b in batch]

    return collate(adapted_batch)

# Copyright (c) Meta Platforms, Inc. and affiliates.

import copy
import functools
import os
import time
from types import SimpleNamespace
import numpy as np

import blobfile as bf
import torch
from torch.optim import AdamW

from diffusion import logger
from utils import dist_util
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.resample import LossAwareSampler, UniformSampler
from tqdm import tqdm
from diffusion.resample import create_named_schedule_sampler
from torch.cuda import amp
from utils.parser_util import TrainingOptions
from utils.model_util import load_model_wo_clip
from torch import nn


class TrainLoop:
    def __init__(self, args: TrainingOptions, train_platform, model: nn.Module,
                 diffusion, data):
        self.args = args
        self.dataset = args.dataset
        self.train_platform = train_platform
        self.model = model
        self.model_avg = None

        if args.avg_model_beta > 0:
            self.model_avg = copy.deepcopy(self.model)

        self.diffusion = diffusion
        self.cond_mode = model.cond_mode
        self.data = data
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint

        self.use_fp16 = args.use_fp16  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size  # * dist.get_world_size()
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.data) + 1

        self.sync_cuda = torch.cuda.is_available()

        self._load_and_sync_parameters()

        if self.use_fp16:
            # using the new mixed precision trainer
            self.scaler = amp.GradScaler()
            # don't use in fp16
            self.mp_trainer = None
        else:
            self.mp_trainer = MixedPrecisionTrainer(
                model=self.model,
                use_fp16=self.use_fp16,
                fp16_scale_growth=self.fp16_scale_growth,
            )

        self.opt = AdamW(
            # with amp, we don't need to use the mp_trainer's master_params
            (self.model.parameters()
             if self.use_fp16 else self.mp_trainer.master_params),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, args.adam_beta2),
        )

        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(
            self.schedule_sampler_type, diffusion)
        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None

        self.use_ddp = False
        self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(
                resume_checkpoint)
            logger.log(
                f"loading model from checkpoint: {resume_checkpoint}...")

            state_dict = dist_util.load_state_dict(
                resume_checkpoint, map_location=dist_util.dev())

            if 'model_avg' in state_dict:
                print('loading both model and model_avg')
                state_dict, state_dict_avg = state_dict['model'], state_dict[
                    'model_avg']
                load_model_wo_clip(self.model, state_dict)
                load_model_wo_clip(self.model_avg, state_dict_avg)
            else:
                load_model_wo_clip(self.model, state_dict)
                if self.args.avg_model_beta > 0:
                    # in case we load from a legacy checkpoint, just copy the model
                    print('loading model_avg from model')
                    self.model_avg.load_state_dict(self.model.state_dict())

            # self.model.load_state_dict(
            #     dist_util.load_state_dict(
            #         resume_checkpoint, map_location=dist_util.dev()
            #     )
            # )

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(bf.dirname(main_checkpoint),
                                 f"opt{self.resume_step:09}.pt")
        if bf.exists(opt_checkpoint):
            logger.log(
                f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev())

            if self.use_fp16:
                if 'scaler' not in state_dict:
                    print("scaler state not found ... not loading it.")
                else:
                    # load grad scaler state
                    self.scaler.load_state_dict(state_dict['scaler'])
                    # for the rest
                    state_dict = state_dict['opt']

            tgt_wd = self.opt.param_groups[0]['weight_decay']
            print('target weight decay:', tgt_wd)
            self.opt.load_state_dict(state_dict)
            print('loaded weight decay (will be replaced):',
                  self.opt.param_groups[0]['weight_decay'])
            # preserve the weight decay parameter
            for group in self.opt.param_groups:
                group['weight_decay'] = tgt_wd

    def run_loop(self):
        print('train steps:', self.num_steps)
        for epoch in range(self.num_epochs):
            print(f'Starting epoch {epoch}')
            for motion, cond in tqdm(self.data):
                if not (not self.lr_anneal_steps or
                        self.step + self.resume_step < self.lr_anneal_steps):
                    break

                if self.args.drop_redundant:
                    non_redundant_idcs = np.concatenate([np.arange(0, 3), np.arange(63, 66), np.arange(126, 186), np.arange(motion.shape[1]-15, motion.shape[1])])
                    motion_target = motion.clone()
                    motion = motion[:,non_redundant_idcs]
                motion = motion.to(self.device)
                cond['y'] = {
                    key: val.to(self.device) if torch.is_tensor(val) else val
                    for key, val in cond['y'].items()
                }

                if self.args.drop_redundant:
                    cond['y']['motion_target'] =  motion_target.to(self.device)

                self.run_step(motion, cond)
                if self.step % self.log_interval == 0:
                    for k, v in logger.get_current().name2val.items():
                        if k == 'loss':
                            print('step[{}]: loss[{:0.5f}]'.format(
                                self.step + self.resume_step, v))

                        if k in ['step', 'samples'] or '_q' in k:
                            continue
                        else:
                            self.train_platform.report_scalar(
                                name=k,
                                value=v,
                                iteration=self.step + self.resume_step,
                                group_name='Loss')

                if self.step % self.save_interval == 0:
                    self.save()
                    self.model.eval()
                    self.model.train()

                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST",
                                      "") and self.step > 0:
                        return
                self.step += 1
            if not (not self.lr_anneal_steps
                    or self.step + self.resume_step < self.lr_anneal_steps):
                break
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()


    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)

        # optimizer step
        if self.use_fp16:
            self.scaler.unscale_(self.opt)

            # gradient clipping
            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.args.grad_clip)

            grad_norm, param_norm = compute_norms(self.model.parameters())
            logger.logkv('param_norm', param_norm)
            logger.logkv('grad_norm', grad_norm)

            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            # FP32, gradient clipping is done in the optimize call
            self.mp_trainer.optimize(self.opt)

        self.update_average_model()
        self._anneal_lr()
        self.log_step()

    def update_average_model(self):
        # update the average model using exponential moving average
        if self.args.avg_model_beta > 0:
            # master params are FP32
            params = self.model.parameters(
            ) if self.use_fp16 else self.mp_trainer.master_params
            for param, avg_param in zip(params, self.model_avg.parameters()):
                # avg = avg + (param - avg) * (1 - alpha)
                # avg = avg + param * (1 - alpha) - (avg - alpha * avg)
                # avg = alpha * avg + param * (1 - alpha)
                avg_param.data.mul_(self.args.avg_model_beta).add_(
                    param.data, alpha=1 - self.args.avg_model_beta)

    def forward_backward(self, batch, cond):
        if self.use_fp16:
            self.opt.zero_grad()
        else:
            self.mp_trainer.zero_grad()

        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            micro_cond = cond
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0],
                                                      dist_util.dev())

            # forward pass
            with amp.autocast(enabled=self.use_fp16, dtype=torch.float16):
                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.ddp_model,
                    micro,  # [bs, ch, image_size, image_size]
                    t,  # [bs](int) sampled timesteps
                    model_kwargs=micro_cond,
                    dataset=self.data.dataset)

                if last_batch or not self.use_ddp:
                    losses = compute_losses()
                else:
                    with self.ddp_model.no_sync():
                        losses = compute_losses()

                if isinstance(self.schedule_sampler, LossAwareSampler):
                    self.schedule_sampler.update_with_local_losses(
                        t, losses["loss"].detach())

                loss = (losses["loss"] * weights).mean()

            log_loss_dict(self.diffusion, t,
                          {k: v * weights
                           for k, v in losses.items()})

            # backward
            if self.use_fp16:
                self.scaler.scale(loss).backward()
            else:
                self.mp_trainer.backward(loss)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples",
                     (self.step + self.resume_step + 1) * self.global_batch)

    def ckpt_file_name(self):
        return f"model{(self.step+self.resume_step):09d}.pt"

    def save(self):
        def save_checkpoint():
            def del_clip(state_dict):
                # Do not save CLIP weights
                clip_weights = [
                    e for e in state_dict.keys() if e.startswith('clip_model.')
                ]
                for e in clip_weights:
                    del state_dict[e]

            if self.use_fp16:
                state_dict = self.model.state_dict()
            else:
                state_dict = self.mp_trainer.master_params_to_state_dict(
                    self.mp_trainer.master_params)
            del_clip(state_dict)

            if self.args.avg_model_beta > 0:
                # save both the model and the average model
                state_dict_avg = self.model_avg.state_dict()
                del_clip(state_dict_avg)
                state_dict = {'model': state_dict, 'model_avg': state_dict_avg}

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint()

        with bf.BlobFile(
                bf.join(self.save_dir,
                        f"opt{(self.step+self.resume_step):09d}.pt"),
                "wb",
        ) as f:
            opt_state = self.opt.state_dict()
            if self.use_fp16:
                # with fp16 we also save the state dict
                opt_state = {
                    'opt': opt_state,
                    'scaler': self.scaler.state_dict(),
                }

            torch.save(opt_state, f)


def compute_norms(params):
    grad_norm = 0.0
    param_norm = 0.0
    for p in params:
        with torch.no_grad():
            param_norm += torch.norm(p, p=2, dtype=torch.float32).item()**2
            if p.grad is not None:
                grad_norm += torch.norm(p.grad, p=2,
                                        dtype=torch.float32).item()**2
    return np.sqrt(grad_norm), np.sqrt(param_norm)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(),
                                   values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

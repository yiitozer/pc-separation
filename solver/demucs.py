#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Main training script entry point"""
import asteroid.losses.multi_scale_spectral
import logging
import os
from pathlib import Path

import json

import torch
from torch.nn import CosineSimilarity
import torch.nn.functional as F
from tqdm import tqdm

from . import Solver
from model.demucs.demucs import Demucs
from model.demucs.hdemucs import HDemucs, hpss
from model.demucs.htdemucs import HTDemucs
from model.demucs.states import capture_init
from model.demucs import distrib, states
from model.demucs.apply import apply_model, apply_model_hpss
from model.demucs.ema import ModelEMA
from model.demucs.evaluate import new_sdr
from model.demucs.svd import svd_penalty
from model.demucs.utils import EMA
from solver.utils import AverageMeter, Config

class DemucsSolver(Solver):
    def __init__(self,
                 device: torch.device = 'cuda',
                 optimizer_func: torch.optim = torch.optim.Adam,
                 model_cfg: Config = None,
                 dataset_dir: str = '',
                 output_dir: str = '',
                 train_folder_name: str = 'train_aug',
                 val_folder_name: str = 'val_aug',
                 train: bool = True,
                 dataloader_kwargs: dict = {}):

        super(DemucsSolver, self).__init__(model_cfg.model, device, optimizer_func, model_cfg, dataset_dir, output_dir,
                                           train_folder_name, val_folder_name, train, dataloader_kwargs)

        self._targets = model_cfg.targets
        self._model_type = model_cfg.model
        self._init_model()

        if model_cfg.model == 'hdemucs':
            self._model._hpss_latent = model_cfg.train.hpss_latent
            self._model._hpss_output = model_cfg.optim.loss in ['hpss_l1',
                                                                'hpss_rev_l1',
                                                                'cosine_l1',
                                                                'decorrelation_l1',
                                                                'pearson_l1',
                                                                'energy_l1',
                                                                'sparsity_l1',
                                                                'sparsity_decorrelation_l1']
        else:
            self._model._hpss_latent = False
            self._model._hpss_output = None

        self._optimizer = self._optimizer_func(self._model.parameters(),
                                               lr=model_cfg.optim.lr)

        self._quantizer = states.get_quantizer(self._model, model_cfg.quant, self._optimizer)
        self._dmodel = distrib.wrap(self._model)
        self._device = next(iter(self._model.parameters())).device

        self._test_metric = model_cfg.test.metric

        # Exponential moving average of the model, either updated every batch or epoch.
        # The best model from all the EMAs and the original one is kept based on the valid
        # loss for the final best model.
        self._emas = {'batch': [], 'epoch': []}
        for kind in self._emas.keys():
            decays = getattr(model_cfg.ema, kind)
            device = self._device if kind == 'batch' else 'cpu'
            if decays:
                for decay in decays:
                    self._emas[kind].append(ModelEMA(self._model, decay, device=device))

        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer,
            factor=self._model_cfg.train.lr_decay_gamma,
            patience=self._model_cfg.train.lr_decay_patience,
            cooldown=10,
        )

        self._mssstft_loss = asteroid.losses.multi_scale_spectral.SingleSrcMultiScaleSpectral(
            windows_size=[2048, 1024, 512, 256, 128, 64, 32],
            hops_size=[1024, 512, 256, 128, 64, 32, 16]).to(self._device)

        if not train:
            self._model.eval()

    def _init_model(self):
        klass = {'demucs': Demucs,
                 'hdemucs': HDemucs,
                 'htdemucs': HTDemucs}[self._model_cfg.model]

        if self._model_cfg.model == 'hdemucs':
            self._model = klass(sources=self._targets,
                                audio_channels=self._num_channels,
                                samplerate=self._model_cfg.audio.sample_rate,
                                segment=self._model_cfg.model_segment,
                                time_dec=self._model_cfg.time_dec,
                                freq_dec=self._model_cfg.freq_dec,
                                sep_conv_enc=self._model_cfg.sep_conv_enc,
                                sep_conv_dec=self._model_cfg.sep_conv_dec).to(self._device)
        else:
            self._model = klass(sources=self._targets,
                                audio_channels=self._num_channels,
                                samplerate=self._model_cfg.audio.sample_rate,
                                segment=self._model_cfg.model_segment).to(self._device)


    def _format_train(self, metrics: dict) -> dict:
        """Formatting for train/valid metrics."""
        losses = {
            'loss': format(metrics['loss'], ".4f"),
            'reco': format(metrics['reco'], ".4f"),
        }
        if 'nsdr' in metrics:
            losses['nsdr'] = format(metrics['nsdr'], ".3f")
        if self._quantizer is not None:
            losses['ms'] = format(metrics['ms'], ".2f")
        if 'grad' in metrics:
            losses['grad'] = format(metrics['grad'], ".4f")
        if 'best' in metrics:
            losses['best'] = format(metrics['best'], '.4f')
        if 'bname' in metrics:
            losses['bname'] = metrics['bname']
        if 'penalty' in metrics:
            losses['penalty'] = format(metrics['penalty'], ".4f")
        if 'hloss' in metrics:
            losses['hloss'] = format(metrics['hloss'], ".4f")
        return losses

    def _format_test(self, metrics: dict) -> dict:
        """Formatting for test metrics."""
        losses = {}
        if 'sdr' in metrics:
            losses['sdr'] = format(metrics['sdr'], '.3f')
        if 'nsdr' in metrics:
            losses['nsdr'] = format(metrics['nsdr'], '.3f')
        for source in self._model.sources:
            key = f'sdr_{source}'
            if key in metrics:
                losses[key] = format(metrics[key], '.3f')
            key = f'nsdr_{source}'
            if key in metrics:
                losses[key] = format(metrics[key], '.3f')
        return losses

    def _compute_loss(self, estimates, targets_gt):
        dims = tuple(range(2, targets_gt.dim()))

        if self._model_cfg.optim.loss == 'l1':
            loss = F.l1_loss(estimates, targets_gt, reduction='none')
            loss = loss.mean(dims).mean(0)
            reco = loss

        elif self._model_cfg.optim.loss == 'mse':
            loss = F.mse_loss(estimates, targets_gt, reduction='none')
            loss = loss.mean(dims)
            reco = loss ** 0.5
            reco = reco.mean(0)

        elif self._model_cfg.optim.loss == 'l1+msstft':
            l1_loss = F.l1_loss(estimates, targets_gt, reduction='none')
            l1_loss = l1_loss.mean(dims).mean(0)

            msstft_loss = self._mssstft_loss(estimates, targets_gt)
            msstft_loss = msstft_loss.mean(0) / targets_gt.numel()
            loss = l1_loss + msstft_loss
            reco = loss

        elif self._model_cfg.optim.loss == 'hpss_l1':
            # supi dupi mega messy
            estimates_h = estimates[1]
            estimates_t = estimates[2]
            estimates = estimates[0]

            l1_loss_waveform = F.l1_loss(estimates, targets_gt, reduction='none')
            length = estimates.shape[-1]

            targets_gt_h = torch.zeros_like(targets_gt)
            targets_gt_p = torch.zeros_like(targets_gt)

            for target_idx in range(len(self._targets)):
                z = self._model._spec(targets_gt[:, target_idx, ...])
                x_mag = z.abs()
                mask_h, mask_p = hpss(x_mag)
                x_mag_h = z * mask_h
                x_mag_p = z * mask_p
                targets_gt_h[:, target_idx, ...] = self._model._ispec(x_mag_h, length)
                targets_gt_p[:, target_idx, ...] = self._model._ispec(x_mag_p, length)

            l1_loss_h = F.l1_loss(estimates_h, targets_gt_h, reduction='none')
            l1_loss_p = F.l1_loss(estimates_t, targets_gt_p, reduction='none')

            # TEST!
            # import soundfile as sf
            # sf.write('test_h.wav', targets_gt_h.squeeze(0)[0, ...].T, 44100)
            # sf.write('test_p.wav', targets_gt_p.squeeze(0)[0, ...].T, 44100)

            loss = l1_loss_waveform + l1_loss_h + l1_loss_p
            loss = loss.mean(dims).mean(0)
            reco = loss
            
        elif self._model_cfg.optim.loss == 'hpss_rev_l1':
            # supi dupi mega messy
            estimates_z = estimates[1]
            estimates_t = estimates[2]
            estimates = estimates[0]

            l1_loss_waveform = F.l1_loss(estimates, targets_gt, reduction='none')
            length = estimates.shape[-1]

            targets_gt_h = torch.zeros_like(targets_gt)
            targets_gt_p = torch.zeros_like(targets_gt)

            for target_idx in range(len(self._targets)):
                z = self._model._spec(targets_gt[:, target_idx, ...])
                x_mag = z.abs()
                mask_h, mask_p = hpss(x_mag)
                x_mag_h = z * mask_h
                x_mag_p = z * mask_p
                targets_gt_h[:, target_idx, ...] = self._model._ispec(x_mag_h, length)
                targets_gt_p[:, target_idx, ...] = self._model._ispec(x_mag_p, length)

            l1_loss_h = F.l1_loss(estimates_t, targets_gt_h, reduction='none')
            l1_loss_p = F.l1_loss(estimates_z, targets_gt_p, reduction='none')

            loss = l1_loss_waveform + l1_loss_h + l1_loss_p
            loss = loss.mean(dims).mean(0)
            reco = loss

            
        # Enforce decorrelation between h and t
        elif self._model_cfg.optim.loss == 'decorrelation_l1':
            estimates_h = estimates[1]
            estimates_t = estimates[2]
            estimates = estimates[0]

            conv_loss = torch.zeros([len(self._targets)]).to(self._device)
            for target_idx in range(len(self._targets)):
                z_h = self._decor_spect(x=estimates_h[:, target_idx, ...])
                z_t = self._decor_spect(x=estimates_t[:, target_idx, ...])
                z_t_conj = torch.conj(z_t)
                mult = (z_h * z_t_conj)
                corr_x = torch.fft.ifft(mult).abs()
                conv_loss += corr_x.max(dim=2).values.mean(-1).mean(0)

            l1_loss_waveform = F.l1_loss(estimates, targets_gt, reduction='none')
            loss = l1_loss_waveform.mean(dims).mean(0) + conv_loss
            reco = loss

        # Enforce cosine DISSIMLARITY between h and t
        elif self._model_cfg.optim.loss == 'cosine_l1':
            estimates_h = estimates[1]
            estimates_t = estimates[2]
            estimates = estimates[0]

            cs = torch.zeros([len(self._targets)]).to(self._device)

            for target_idx in range(len(self._targets)):
                z_h = self._model._spec(estimates_h[:, target_idx, ...])
                z_t = self._model._spec(estimates_t[:, target_idx, ...])
                sim = torch.nn.functional.cosine_similarity(z_h.abs(), z_t.abs(), dim=0)
                cs[target_idx] = sim.mean() * 0.01 # TODO: LAMBDA, HARD-CODED!!
                   
            l1_loss_waveform = F.l1_loss(estimates, targets_gt, reduction='none')

            loss = l1_loss_waveform.mean(dims).mean(0) + cs
            reco = loss

        elif self._model_cfg.optim.loss == 'pearson_l1':
            estimates_h = estimates[1]
            estimates_t = estimates[2]
            estimates = estimates[0]

            pearson_cost = torch.zeros([len(self._targets)]).to(self._device)
            for target_idx in range(len(self._targets)):
                h = estimates_h[:, target_idx, ...]
                t = estimates_t[:, target_idx, ...]
                h = h - h.mean()
                t = t - t.mean()
                pearson_cost[target_idx] = torch.sum(h * t) / (torch.sqrt(torch.sum(h ** 2)) * torch.sqrt(torch.sum(t ** 2)))
            
            # -1: negative correlation, +1: positive correlation
            pearson_cost = pearson_cost.abs()
                
            l1_loss_waveform = F.l1_loss(estimates, targets_gt, reduction='none')
            loss = l1_loss_waveform.mean(dims).mean(0) + pearson_cost
            reco = loss

        elif self._model_cfg.optim.loss == 'energy_l1':
            estimates_h = estimates[1]
            estimates_t = estimates[2]
            estimates = estimates[0]
            e = (estimates_h ** 2 + estimates_t ** 2).sum(dims).mean(0) / 500 # TODO: WATCH OUT! HARD-CODED!

            l1_loss_waveform = F.l1_loss(estimates, targets_gt, reduction='none')
            loss = l1_loss_waveform.mean(dims).mean(0) + e 
          
            reco = loss
            
        elif self._model_cfg.optim.loss == 'sparsity_l1':
            estimates_h = estimates[1]
            estimates_t = estimates[2]
            estimates = estimates[0]

            sparsity_loss = torch.zeros([len(self._targets)]).to(self._device)
            for target_idx in range(len(self._targets)):
                z_h = self._decor_spect(x=estimates_h[:, target_idx, ...])
                z_t = self._decor_spect(x=estimates_t[:, target_idx, ...])
                y_h = z_h.abs()
                y_t = z_t.abs()
                sl = y_h.mean() + y_t.mean()
                sparsity_loss[target_idx] = sl
                
            l1_loss_waveform = F.l1_loss(estimates, targets_gt, reduction='none')
            loss = l1_loss_waveform.mean(dims).mean(0)            
            reco = loss
            
        elif self._model_cfg.optim.loss == 'sparsity_decorrelation_l1':
            estimates_h = estimates[1]
            estimates_t = estimates[2]
            estimates = estimates[0]

            conv_loss = torch.zeros([len(self._targets)]).to(self._device)
            sparsity_loss = torch.zeros([len(self._targets)]).to(self._device)

            for target_idx in range(len(self._targets)):
                z_h = self._decor_spect(x=estimates_h[:, target_idx, ...])
                z_t = self._decor_spect(x=estimates_t[:, target_idx, ...])
                z_t_conj = torch.conj(z_t)
                mult = (z_h * z_t_conj)
                corr_x = torch.fft.ifft(mult).abs()
                conv_loss += corr_x.max(dim=2).values.mean(-1).mean(0)
                y_h = z_h.abs()
                y_t = z_t.abs()
                sl = y_h.mean() + y_t.mean()
                sparsity_loss[target_idx] = sl
                

            l1_loss_waveform = F.l1_loss(estimates, targets_gt, reduction='none')
            loss = l1_loss_waveform.mean(dims).mean(0) + conv_loss + sparsity_loss
            reco = loss

        else:
            raise ValueError(f"Invalid loss {self._model_cfg.optim.loss}")

        return loss, reco

    def train(self,
              epoch:int):
        if self._model_cfg.audio.random_start_frame:
            self._train_dataset.generate_start_frames(seed=epoch)
        if self._model_cfg.audio.random_gain:
            self._train_dataset.generate_gains(seed=epoch+42)
        if self._model_cfg.audio.random_silence:
            self._train_dataset.generate_random_silence(seed=epoch+12)

        averager = EMA()
        self._model.train()
        pbar = tqdm(self._train_sampler, disable=False, miniters=1)
        pbar.set_description('Training batch')
        for idx, train_dict in enumerate(pbar):
            self._optimizer.zero_grad()
            mix = train_dict['mix'].to(self._device)

            targets_gt = torch.stack([train_dict[target] for target in self._targets], dim=1).to(self._device)

            estimates = self._model(mix)
            loss, reco = self._compute_loss(estimates, targets_gt)
            weights = torch.tensor(self._model_cfg.weights).to(targets_gt)
            loss = (loss * weights).sum() / weights.sum()

            ms = 0
            if self._quantizer is not None:
                ms = self._quantizer.model_size()

            if self._model_cfg.quant.diffq:
                loss += self._model_cfg.quant.diffq.quant.diffq * ms

            losses = {}
            losses['reco'] = (reco * weights).sum() / weights.sum()
            losses['ms'] = ms

            if self._model_cfg.svd.penalty > 0:
                penalty = svd_penalty(model=self._model,
                                      min_size=self._model_cfg.svd.min_size,
                                      dim=self._model_cfg.svd.dim,
                                      niters=self._model_cfg.svd.niters,
                                      powm=self._model_cfg.svd.powm,
                                      convtr=self._model_cfg.svd.convtr,
                                      proba=self._model_cfg.svd.proba,
                                      conv_only=self._model_cfg.svd.conv_only,
                                      bs=self._model_cfg.svd.bs)

                losses['penalty'] = penalty
                loss += self._model_cfg.svd.penalty * penalty

            losses['loss'] = loss

            for k, source in enumerate(self._targets):
                losses[f'reco_{source}'] = reco[k]

            # optimize model in training mode
            loss.backward()

            grad_norm = 0
            grads = []
            for p in self._model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm() ** 2
                    grads.append(p.grad.data)
            losses['grad'] = grad_norm ** 0.5

            if self._model_cfg.optim.clip_grad:
                torch.nn.utils.clip_grad_norm_(
                    self._model.parameters(),
                    self._model_cfg.optim.clip_grad)

            if self._model_cfg.flag == 'uns':
                for n, p in self.model.named_parameters():
                    if p.grad is None:
                        print('no grad', n)

            self._optimizer.step()

            for ema in self._emas['batch']:
                ema.update()

            losses = averager(losses)

            pbar.set_postfix(loss=self._format_train(losses))

            del loss, estimates, reco, ms

            if self._model_cfg.train.max_batches == idx:
                break

            for ema in self._emas['epoch']:
                ema.update()

        self._train_losses.append(losses)

        return losses

    def val(self):
        self._model.eval()
        averager = EMA()
        with torch.no_grad():
            for idx, val_dict in enumerate(self._val_sampler):
                self._optimizer.zero_grad()
                mix = val_dict['mix'].to(self._device)
                if self._model._hpss_output:
                    estimates = apply_model_hpss(self._model,
                                        mix,
                                        split=self._model_cfg.test.split,
                                        overlap=0)
                else:
                    estimates = apply_model(self._model,
                                            mix,
                                            split=self._model_cfg.test.split,
                                            overlap=0)
                losses = {}
                targets_gt = torch.stack([val_dict[target] for target in self._targets], dim=1).to(self._device)
                loss, reco = self._compute_loss(estimates, targets_gt)
                weights = torch.tensor(self._model_cfg.weights).to(targets_gt)
                loss = (loss * weights).sum() / weights.sum()

                ms = 0
                if self._quantizer is not None:
                    ms = self._quantizer.model_size()

                if self._model_cfg.quant.diffq:
                    loss += self._model_cfg.quant.diffq.quant.diffq * ms

                losses['reco'] = (reco * weights).sum() / weights.sum()
                losses['ms'] = ms

                if self._model._hpss_output:
                    nsdrs = new_sdr(targets_gt, estimates[0].detach()).mean(0)
                    
                else:
                    nsdrs = new_sdr(targets_gt, estimates.detach()).mean(0)

                total = 0
                for source, nsdr, w in zip(self._targets, nsdrs, weights):
                    losses[f'nsdr_{source}'] = nsdr
                    total += w * nsdr
                losses['nsdr'] = total / weights.sum()
                losses['loss'] = loss

                for k, target in enumerate(self._targets):
                    losses[f'reco_{target}'] = reco[k]

                losses = averager(losses)

                del loss, estimates, reco, ms

                if self._model_cfg.train.max_batches == idx:
                    break

        self._scheduler.step(losses['loss'])
        self._val_losses.append(losses)
        return losses

    def save_params(self,
                    params: dict):
        with open(Path(self._output_dir, f'{self._model_type}.json'), 'w') as outfile:
            outfile.write(json.dumps(params, indent=4, sort_keys=True))

    @property
    def test_metric(self):
        return self._test_metric

    def separate(self, mix):
        with torch.no_grad():
            estimates = self._model(mix)
        estimates_dict = {target: estimates[:, target_idx, ...].squeeze(0) for target_idx, target in enumerate(self._targets)}

        return estimates_dict

    def separate_corr(self, mix):
        estimates, time_estimates, freq_estimates = self._model.forward_corr(mix)
        estimates_dict = {target: estimates[:, target_idx, ...] for target_idx, target in enumerate(self._targets)}
        time_dict = {target: time_estimates[:, target_idx, ...] for target_idx, target in enumerate(self._targets)}
        freq_dict = {target: freq_estimates[:, target_idx, ...] for target_idx, target in enumerate(self._targets)}

        return estimates_dict, time_dict, freq_dict
    
    @staticmethod
    def _decor_spect(x, n_fft=4096, hop_length=None, pad=0):
        *other, length = x.shape
        x = x.reshape(-1, length)
        z = torch.stft(x,
                n_fft * 2 - 1,
                hop_length or n_fft // 4,
                window=torch.hann_window(n_fft).to(x),
                win_length=n_fft,
                normalized=True,
                center=True,
                return_complex=True,
                pad_mode='reflect')
        _, freqs, frame = z.shape
        return z.view(*other, freqs, frame)

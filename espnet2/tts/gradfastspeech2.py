import logging

from typing import Dict
from typing import Sequence
from typing import Tuple

import yaml
import argparse
import torch
import torch.nn.functional as F
from espnet2.tts.abs_tts import AbsTTS
from espnet.nets.pytorch_backend.gradtts.diffusion import Diffusion
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.tts.fastspeech2 import FastSpeech2
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask


class GradFastSpeech2(AbsTTS):
    def __init__(
        self,
        idim,
        odim,
        config_file=None,
        model_file=None,
        ddim: int = 64,
        beta_min: float = 0.05,
        beta_max: float = 20.0,
        pe_scale: int = 1000,
    ) -> None:
        super().__init__()
        with open(config_file, "r", encoding="utf-8") as f:
            args = yaml.safe_load(f)
        args = argparse.Namespace(**args)
        self.idim = idim
        self.odim = odim
        self.padding_idx = 0
        self.eos = idim - 1

        self.fastspeech2 = FastSpeech2(idim=len(args.token_list), odim=odim, **args.tts_conf)
        tmp = torch.load(model_file)
        d = {}
        for key, value in tmp.items():
            if key.startswith("tts"):
                d[key[4:]] = value
        self.fastspeech2.load_state_dict(d)
        for p in self.fastspeech2.parameters():
            p.requires_grad = False

        self.diffusion = Diffusion(ddim, beta_min, beta_max, pe_scale)

        self.criterion = GradFastSpeech2Loss(odim)

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        durations: torch.Tensor,
        durations_lengths: torch.Tensor,
        pitch: torch.Tensor,
        pitch_lengths: torch.Tensor,
        energy: torch.Tensor,
        energy_lengths: torch.Tensor,
        spembs: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        text = text[:, : text_lengths.max()]  # for data-parallel
        speech = speech[:, : speech_lengths.max()]  # for data-parallel
        durations = durations[:, : durations_lengths.max()]  # for data-parallel
        pitch = pitch[:, : pitch_lengths.max()]  # for data-parallel
        energy = energy[:, : energy_lengths.max()]  # for data-parallel

        batch_size = text.size(0)

        # Add eos at the last of sequence
        xs = F.pad(text, [0, 1], "constant", self.padding_idx)
        for i, l in enumerate(text_lengths):
            xs[i, l] = self.eos
        ilens = text_lengths + 1

        ys, ds, ps, es = speech, durations, pitch, energy
        olens = speech_lengths
        
        before_outs, after_outs, d_outs, p_outs, e_outs = self.fastspeech2._forward(
            xs, ilens, ys, olens, ds, ps, es, spembs=spembs, is_inference=False
        )

        ys = speech.transpose(1, 2)
        y_masks = self._source_mask(olens)
        mu = after_outs.transpose(1, 2)
        
        if ys.size(2) % 4 != 0:
            ys = torch.cat([ys, torch.zeros([batch_size, self.odim, 4 - ys.size(2) % 4], dtype=ys.dtype, device=ys.device)], dim=2)
            mu = torch.cat([mu, torch.zeros([mu.size(0), self.odim, 4 - mu.size(2) % 4], dtype=mu.dtype, device=mu.device)], dim=2)
            y_masks = torch.cat([y_masks, torch.zeros([y_masks.size(0), 1, 4 - y_masks.size(2) % 4], dtype=y_masks.dtype, device=y_masks.device)], dim=2)

        noise_estimation, z = self.diffusion(ys, y_masks, mu)
        
        diff_loss = self.criterion(noise_estimation, z, y_masks)
        loss = diff_loss
        stats = dict(
            diff_loss=diff_loss.item(),
            loss=loss.item(),
        )
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def inference(
        self,
        text: torch.Tensor,
        timesteps: int = 50,
        spembs: torch.Tensor = None,
        temperature: float = 1.0,
        alpha: float = 1.03,
        use_teacher_forcing: bool = False,
    ):
        mu, _, _ = self.fastspeech2.inference(text, alpha=alpha, use_teacher_forcing=use_teacher_forcing)
        # import numpy as np
        # np.save("/nolan/inference/gradtts_pre.mel.npy", mu.data.cpu().numpy())
        length = mu.shape[0]
        olens = torch.tensor([length], dtype=torch.long, device=mu.device)
        y_masks = self._source_mask(olens)
        mu = mu.unsqueeze(0).transpose(1, 2)
        if mu.size(2) % 4 != 0:
            mu = torch.cat([mu, torch.zeros([1, self.odim, 4 - mu.size(2) % 4], dtype=mu.dtype, device=mu.device)], dim=2)
            y_masks = torch.cat([y_masks, torch.zeros([1, 1, 4 - y_masks.size(2) % 4], dtype=y_masks.dtype, device=y_masks.device)], dim=2)
        z = mu + torch.randn_like(mu, device=mu.device) / temperature
        out = self.diffusion.inference(z, y_masks, mu, timesteps).transpose(1, 2)
        return out[0, :length, :], None, None

    def _source_mask(self, ilens: torch.Tensor) -> torch.Tensor:
        x_masks = make_non_pad_mask(ilens).to(next(self.parameters()).device)
        return x_masks.unsqueeze(-2)

class GradFastSpeech2Loss(torch.nn.Module):
    def __init__(self, odim):
        super().__init__()
        self.odim = odim

    def forward(
        self,
        noise_estimation: torch.Tensor,
        z: torch.Tensor,
        y_masks: torch.Tensor,
    ):
        diff_loss = torch.sum((noise_estimation + z) ** 2) / (torch.sum(y_masks) * self.odim)

        return diff_loss

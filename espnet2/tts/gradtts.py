import torch
from torch._C import device
import torch.nn.functional as F
import logging
import math
from typing import Dict
from typing import Sequence
from typing import Tuple
from espnet2.tts.abs_tts import AbsTTS

from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.fastspeech.length_regulator import LengthRegulator
from espnet.nets.pytorch_backend.transformer.embedding import ScaledPositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder import (
    Encoder as TransformerEncoder,  # noqa: H301
)
from espnet.nets.pytorch_backend.fastspeech.duration_predictor import DurationPredictor
from espnet.nets.pytorch_backend.fastspeech.duration_predictor import (
    DurationPredictorLoss,  # noqa: H301
)
from espnet.nets.pytorch_backend.gradtts.diffusion import Diffusion
from espnet2.torch_utils.initialize import initialize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.tts.variance_predictor import VariancePredictor


class ConvReluNorm(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, n_layers, p_dropout):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.conv_layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()
        self.conv_layers.append(torch.nn.Conv1d(in_channels, hidden_channels,
                                                kernel_size, padding=kernel_size // 2))
        self.norm_layers.append(torch.nn.LayerNorm(hidden_channels, eps=1e-12))
        self.relu_drop = torch.nn.Sequential(
            torch.nn.ReLU(), torch.nn.Dropout(p_dropout))
        for _ in range(n_layers - 1):
            self.conv_layers.append(torch.nn.Conv1d(hidden_channels, hidden_channels,
                                                    kernel_size, padding=kernel_size // 2))
            self.norm_layers.append(torch.nn.LayerNorm(hidden_channels, eps=1e-12))
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, masks):
        x_org = x
        for i in range(self.n_layers):
            x = self.conv_layers[i](x * masks)
            x = x.transpose(1, 2)
            x = self.norm_layers[i](x)
            x = x.transpose(1, 2)
            x = self.relu_drop(x)
        x = x_org + self.proj(x)
        return x * masks


class GradTTSEncoder(TransformerEncoder):
    def __init__(
        self,
        idim,
        **kwargs,
    ):
        super().__init__(
            idim=idim,
            **kwargs,
        )
        self.idim = idim
        self.prenet = ConvReluNorm(
            idim, idim, idim, kernel_size=5, n_layers=3, p_dropout=0.5)

    def forward(self, xs, masks):
        xs = self.embed(xs) * math.sqrt(self.idim)
        xs = xs.transpose(1, 2) # (B, idim, Tmax)
        xs = self.prenet(xs, masks)

        xs = xs.transpose(1, 2) # (B, Tmax, idim)
        xs, _ = self.encoders(xs, masks)  # (B, Tmax, adim)

        return xs


class GradTTS(AbsTTS):
    def __init__(
        self,
        idim,
        odim,
        adim: int = 192,
        aheads: int = 2,
        elayers: int = 4,
        eunits: int = 768,
        dlayers: int = 2,
        dunits: int = 768,
        use_scaled_pos_enc: bool = True,
        positionwise_layer_type: str = "conv1d",
        positionwise_conv_kernel_size: int = 1,
        encoder_normalize_before: bool = True,
        decoder_normalize_before: bool = True,
        encoder_concat_after: bool = False,
        decoder_concat_after: bool = False,
        duration_predictor_layers: int = 2,
        duration_predictor_chans: int = 192,
        duration_predictor_kernel_size: int = 3,
        duration_predictor_dropout_rate: float = 0.1,
        transformer_enc_dropout_rate: float = 0.1,
        transformer_enc_positional_dropout_rate: float = 0.1,
        transformer_enc_attn_dropout_rate: float = 0.1,
        transformer_dec_dropout_rate: float = 0.1,
        transformer_dec_positional_dropout_rate: float = 0.1,
        transformer_dec_attn_dropout_rate: float = 0.1,
        init_type: str = "xavier_uniform",
        init_enc_alpha: float = 1.0,
        init_dec_alpha: float = 1.0,
        use_masking: bool = False,
        use_weighted_masking: bool = False,
        ddim: int = 64,
        beta_min: float = 0.05,
        beta_max: float = 20.0,
        pe_scale: int = 1000,
        # energy predictor
        energy_predictor_layers: int = 2,
        energy_predictor_chans: int = 384,
        energy_predictor_kernel_size: int = 3,
        energy_predictor_dropout: float = 0.5,
        energy_embed_kernel_size: int = 9,
        energy_embed_dropout: float = 0.5,
        stop_gradient_from_energy_predictor: bool = False,
        # pitch predictor
        pitch_predictor_layers: int = 2,
        pitch_predictor_chans: int = 384,
        pitch_predictor_kernel_size: int = 3,
        pitch_predictor_dropout: float = 0.5,
        pitch_embed_kernel_size: int = 9,
        pitch_embed_dropout: float = 0.5,
        stop_gradient_from_pitch_predictor: bool = False,
    ):
        super(GradTTS, self).__init__()
        self.idim = idim
        self.odim = odim
        self.eos = idim - 1
        self.stop_gradient_from_pitch_predictor = stop_gradient_from_pitch_predictor
        self.stop_gradient_from_energy_predictor = stop_gradient_from_energy_predictor

        self.padding_idx = 0
        self.use_scaled_pos_enc = use_scaled_pos_enc
        encoder_input_layer = torch.nn.Embedding(
            num_embeddings=idim, embedding_dim=adim, padding_idx=self.padding_idx
        )
        pos_enc_class = (
            ScaledPositionalEncoding if self.use_scaled_pos_enc else PositionalEncoding
        )
        self.encoder = TransformerEncoder(
            idim=adim,
            attention_dim=adim,
            attention_heads=aheads,
            linear_units=eunits,
            num_blocks=elayers,
            input_layer=encoder_input_layer,
            dropout_rate=transformer_enc_dropout_rate,
            positional_dropout_rate=transformer_enc_positional_dropout_rate,
            attention_dropout_rate=transformer_enc_attn_dropout_rate,
            pos_enc_class=pos_enc_class,
            normalize_before=encoder_normalize_before,
            concat_after=encoder_concat_after,
            positionwise_layer_type=positionwise_layer_type,
            positionwise_conv_kernel_size=positionwise_conv_kernel_size,
        )

        self.duration_predictor = DurationPredictor(
            idim=adim,
            n_layers=duration_predictor_layers,
            n_chans=duration_predictor_chans,
            kernel_size=duration_predictor_kernel_size,
            dropout_rate=duration_predictor_dropout_rate,
        )
        self.pitch_predictor = VariancePredictor(
            idim=adim,
            n_layers=pitch_predictor_layers,
            n_chans=pitch_predictor_chans,
            kernel_size=pitch_predictor_kernel_size,
            dropout_rate=pitch_predictor_dropout,
        )
        # NOTE(kan-bayashi): We use continuous pitch + FastPitch style avg
        self.pitch_embed = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=1,
                out_channels=adim,
                kernel_size=pitch_embed_kernel_size,
                padding=(pitch_embed_kernel_size - 1) // 2,
            ),
            torch.nn.Dropout(pitch_embed_dropout),
        )

        # define energy predictor
        self.energy_predictor = VariancePredictor(
            idim=adim,
            n_layers=energy_predictor_layers,
            n_chans=energy_predictor_chans,
            kernel_size=energy_predictor_kernel_size,
            dropout_rate=energy_predictor_dropout,
        )
        # NOTE(kan-bayashi): We use continuous enegy + FastPitch style avg
        self.energy_embed = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=1,
                out_channels=adim,
                kernel_size=energy_embed_kernel_size,
                padding=(energy_embed_kernel_size - 1) // 2,
            ),
            torch.nn.Dropout(energy_embed_dropout),
        )

        self.length_regulator = LengthRegulator()

        self.pre_decoder = TransformerEncoder(
            idim=0,
            attention_dim=adim,
            attention_heads=aheads,
            linear_units=dunits,
            num_blocks=dlayers,
            input_layer=None,
            dropout_rate=transformer_dec_dropout_rate,
            positional_dropout_rate=transformer_dec_positional_dropout_rate,
            attention_dropout_rate=transformer_dec_attn_dropout_rate,
            pos_enc_class=pos_enc_class,
            normalize_before=decoder_normalize_before,
            concat_after=decoder_concat_after,
            positionwise_layer_type=positionwise_layer_type,
            positionwise_conv_kernel_size=positionwise_conv_kernel_size,
        )
        
        self.feat_out = torch.nn.Linear(adim, odim)

        self.decoder = Diffusion(ddim, beta_min, beta_max, pe_scale)

        self._reset_parameters(
            init_type=init_type,
            init_enc_alpha=init_enc_alpha,
            init_dec_alpha=init_dec_alpha,
        )

        self.criterion = GradTTSLoss(
            odim,
            use_masking=use_masking,
            use_weighted_masking=use_weighted_masking,
        )

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        durations: torch.Tensor,
        durations_lengths: torch.Tensor,
        pitch: torch.Tensor = None,
        pitch_lengths: torch.Tensor = None,
        energy: torch.Tensor = None,
        energy_lengths: torch.Tensor = None,
        spembs: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        text = text[:, : text_lengths.max()]  # for data-parallel
        speech = speech[:, : speech_lengths.max()]  # for data-parallel
        durations = durations[:, : durations_lengths.max()]  # for data-parallel
        pitch = pitch[:, : pitch_lengths.max()]  # for data-parallel
        energy = energy[:, : energy_lengths.max()]  # for data-parallel

        batch_size = text.size(0)

        xs = F.pad(text, [0, 1], "constant", self.padding_idx)
        for i, l in enumerate(text_lengths):
            xs[i, l] = self.eos # (B, Tmax, idim)
        ilens = text_lengths + 1
        ys, ds, ps, es = speech, durations, pitch, energy
        olens = speech_lengths
        ys = ys.transpose(1, 2)  # (B, odim, Lmax)
        if ys.size(2) % 4 != 0:
            ys = torch.cat([ys, torch.zeros([batch_size, self.odim, 4 - ys.size(2) % 4], dtype=ys.dtype, device=ys.device)], dim=2)

        noise_estimation, z, d_outs, p_outs, e_outs, mu, y_masks = self._forward(xs, ilens, ys, olens, ds, ps, es)

        prior_loss, duration_loss, diff_loss, pitch_loss, energy_loss = self.criterion(mu, noise_estimation, z, d_outs, p_outs, e_outs, ys, ds, ps, es, y_masks, ilens)

        loss = prior_loss + duration_loss + diff_loss + pitch_loss + energy_loss
        stats = dict(
            prior_loss=prior_loss.item(),
            duration_loss=duration_loss.item(),
            diff_loss=diff_loss.item(),
            pitch_loss=pitch_loss.item(),
            energy_loss=energy_loss.item(),
            loss=loss.item(),
        )
        if self.use_scaled_pos_enc:
            stats.update(
                encoder_alpha=self.encoder.embed[-1].alpha.data.item(),
            )
            stats.update(
                decoder_alpha=self.pre_decoder.embed[-1].alpha.data.item(),
            )
        
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def _forward(
        self,
        xs: torch.Tensor,
        ilens: torch.Tensor,
        ys: torch.Tensor = None,
        olens: torch.Tensor = None,
        ds: torch.Tensor = None,
        ps: torch.Tensor = None,
        es: torch.Tensor = None,
    ):
        x_masks = self._source_mask(ilens)  # (B, 1, Tmax)
        y_masks = self._source_mask(olens)  # (B, 1, Lmax)
        hs, _ = self.encoder(xs, x_masks)  # (B, Tmax, adim)

        d_masks = make_pad_mask(ilens).to(xs.device)
        if self.stop_gradient_from_pitch_predictor:
            p_outs = self.pitch_predictor(hs.detach(), d_masks.unsqueeze(-1))
        else:
            p_outs = self.pitch_predictor(hs, d_masks.unsqueeze(-1))
        if self.stop_gradient_from_energy_predictor:
            e_outs = self.energy_predictor(hs.detach(), d_masks.unsqueeze(-1))
        else:
            e_outs = self.energy_predictor(hs, d_masks.unsqueeze(-1))

        d_outs = self.duration_predictor(hs, d_masks)
        # use groundtruth in training
        p_embs = self.pitch_embed(ps.transpose(1, 2)).transpose(1, 2)
        e_embs = self.energy_embed(es.transpose(1, 2)).transpose(1, 2)

        hs = hs + e_embs + p_embs
        mu = self.length_regulator(hs, ds)  # (B, Lmax, adim)

        mu, _ = self.pre_decoder(mu, y_masks) # (B, Lmax, adim)

        mu = self.feat_out(mu) # (B, Lmax, odim)

        mu = mu.transpose(1, 2)  # (B, odim, Lmax)
        if mu.size(2) % 4 != 0:
            mu = torch.cat([mu, torch.zeros([mu.size(0), self.odim, 4 - mu.size(2) % 4], dtype=mu.dtype, device=mu.device)], dim=2)
            y_masks = torch.cat([y_masks, torch.zeros([y_masks.size(0), 1, 4 - y_masks.size(2) % 4], dtype=y_masks.dtype, device=y_masks.device)], dim=2)
        noise_estimation, z = self.decoder(ys, y_masks, mu)

        return noise_estimation, z, d_outs, p_outs, e_outs, mu, y_masks

    def inference(
        self,
        text: torch.Tensor,
        timesteps: int = 20,
        spembs: torch.Tensor = None,
        temperature: float = 1.0,
        alpha: float = 1.03,
        use_teacher_forcing: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = text
        x = F.pad(x, [0, 1], "constant", self.eos)
        ilens = torch.tensor([x.shape[0]], dtype=torch.long, device=x.device)
        xs = x.unsqueeze(0)
        x_masks = self._source_mask(ilens)
        hs, _ = self.encoder(xs, x_masks)

        d_masks = make_pad_mask(ilens).to(xs.device)
        d_outs = self.duration_predictor.inference(hs, d_masks)

        if self.stop_gradient_from_pitch_predictor:
            p_outs = self.pitch_predictor(hs.detach(), d_masks.unsqueeze(-1))
        else:
            p_outs = self.pitch_predictor(hs, d_masks.unsqueeze(-1))
        if self.stop_gradient_from_energy_predictor:
            e_outs = self.energy_predictor(hs.detach(), d_masks.unsqueeze(-1))
        else:
            e_outs = self.energy_predictor(hs, d_masks.unsqueeze(-1))
        p_embs = self.pitch_embed(p_outs.transpose(1, 2)).transpose(1, 2)
        e_embs = self.energy_embed(e_outs.transpose(1, 2)).transpose(1, 2)
        hs = hs + e_embs + p_embs

        mu = self.length_regulator(hs, d_outs, alpha)  # (B, Lmax, odim)
        length = mu.size(1)
        y_masks = torch.ones([1, 1, mu.size(1)], dtype=torch.int64, device=mu.device)
        mu, _ = self.pre_decoder(mu, y_masks) # (B, Lmax, adim)
        mu = self.feat_out(mu) # (B, Lmax, odim)

        mu = mu.transpose(1, 2)  # (B, odim, Lmax)

        # import numpy as np
        # np.save("/nolan/inference/gradtts_pre.mel.npy", mu[0].transpose(0, 1).data.cpu().numpy())
        if mu.size(2) % 4 != 0:
            mu = torch.cat([mu, torch.zeros([1, self.odim, 4 - mu.size(2) % 4], dtype=mu.dtype, device=mu.device)], dim=2)
            y_masks = torch.cat([y_masks, torch.zeros([1, 1, 4 - y_masks.size(2) % 4], dtype=y_masks.dtype, device=y_masks.device)], dim=2)
        
        z = mu + torch.randn_like(mu, device=mu.device) / temperature

        out = self.decoder.inference(z, y_masks, mu, timesteps, length).transpose(1, 2)

        return out[0, :length, :], None, None

    def decode_inference(self, mu, temperature=1.0, timesteps=10):
        length = mu.shape[0]
        olens = torch.tensor([length], dtype=torch.long, device=mu.device)
        y_masks = self._source_mask(olens)
        mu = mu.unsqueeze(0).transpose(1, 2)
        if mu.size(2) % 4 != 0:
            mu = torch.cat([mu, torch.zeros([1, self.odim, 4 - mu.size(2) % 4], dtype=mu.dtype, device=mu.device)], dim=2)
            y_masks = torch.cat([y_masks, torch.zeros([1, 1, 4 - y_masks.size(2) % 4], dtype=y_masks.dtype, device=y_masks.device)], dim=2)
        z = mu + torch.randn_like(mu, device=mu.device) / temperature
        out = self.decoder.inference(z, y_masks, mu, timesteps).transpose(1, 2)
        return out[0, :length, :]

    def _source_mask(self, ilens: torch.Tensor) -> torch.Tensor:
        x_masks = make_non_pad_mask(ilens).to(next(self.parameters()).device)
        return x_masks.unsqueeze(-2)

    def _reset_parameters(
        self, init_type: str, init_enc_alpha: float, init_dec_alpha: float,
    ):
        # initialize parameters
        if init_type != "pytorch":
            initialize(self, init_type)
        # initialize alpha in scaled positional encoding
        if self.use_scaled_pos_enc:
            self.encoder.embed[-1].alpha.data = torch.tensor(init_enc_alpha)
        if self.use_scaled_pos_enc:
            self.pre_decoder.embed[-1].alpha.data = torch.tensor(init_dec_alpha)


class GradTTSLoss(torch.nn.Module):
    def __init__(self, odim, use_masking: bool = True, use_weighted_masking: bool = False):
        super().__init__()

        assert (use_masking != use_weighted_masking) or not use_masking
        self.odim = odim
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking

        # define criterions
        reduction = "none" if self.use_weighted_masking else "mean"
        self.duration_criterion = DurationPredictorLoss(reduction=reduction)
        self.mse_criterion = torch.nn.MSELoss(reduction=reduction)

    def forward(
        self,
        mu: torch.Tensor,
        noise_estimation: torch.Tensor,
        z: torch.Tensor,
        d_outs: torch.Tensor,
        p_outs: torch.Tensor, 
        e_outs: torch.Tensor,
        ys: torch.Tensor,
        ds: torch.Tensor,
        ps: torch.Tensor,
        es: torch.Tensor,
        y_masks: torch.Tensor,
        ilens: torch.Tensor,
    ):
        if self.use_masking:
            duration_masks = make_non_pad_mask(ilens).to(ys.device)
            d_outs = d_outs.masked_select(duration_masks)
            ds = ds.masked_select(duration_masks)
        prior_loss = torch.sum(0.5 * ((ys - mu) ** 2 + math.log(2 * math.pi)) * y_masks)
        prior_loss = prior_loss / (torch.sum(y_masks) * self.odim)
        duration_loss = self.duration_criterion(d_outs, ds)
        pitch_loss = self.mse_criterion(p_outs, ps)
        energy_loss = self.mse_criterion(e_outs, es)
        diff_loss = torch.sum((noise_estimation + z) ** 2) / (torch.sum(y_masks) * self.odim)

        return prior_loss, duration_loss, diff_loss, pitch_loss, energy_loss

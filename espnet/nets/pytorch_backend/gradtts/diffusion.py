import torch
import math
from einops import rearrange
from espnet.nets.pytorch_backend.transformer.encoder import (
    Encoder as TransformerDecoder,  # noqa: H301
)

class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Mish(torch.nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))

class Block(torch.nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block = torch.nn.Sequential(torch.nn.Conv2d(dim, dim_out, 3, 
                                         padding=1), torch.nn.GroupNorm(
                                         groups, dim_out), Mish())

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask

class Rezero(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g

class Upsample(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class LinearAttention(torch.nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = torch.nn.Conv2d(hidden_dim, dim, 1)            

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', 
                            heads = self.heads, qkv=3)            
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', 
                        heads=self.heads, h=h, w=w)
        return self.to_out(out)

class Residual(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        output = self.fn(x, *args, **kwargs) + x
        return output

class Downsample(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = torch.nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class ResnetBlock(torch.nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super(ResnetBlock, self).__init__()
        self.mlp = torch.nn.Sequential(Mish(), torch.nn.Linear(time_emb_dim, dim_out))

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        if dim != dim_out:
            self.res_conv = torch.nn.Conv2d(dim, dim_out, 1)
        else:
            self.res_conv = torch.nn.Identity()

    def forward(self, x, mask, time_emb):
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output

class Estimator(torch.nn.Module):
    def __init__(self, dim, dim_mults=(1, 2, 4), groups=8, pe_scale=1000):
        super().__init__()
        self.pe_scale = pe_scale
        
        dims = [2, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim, dim * 4), Mish(),
                                       torch.nn.Linear(dim * 4, dim))

        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(torch.nn.ModuleList([
                       ResnetBlock(dim_in, dim_out, time_emb_dim=dim),
                       ResnetBlock(dim_out, dim_out, time_emb_dim=dim),
                       Residual(Rezero(LinearAttention(dim_out))),
                       Downsample(dim_out) if not is_last else torch.nn.Identity()]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(torch.nn.ModuleList([
                     ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim),
                     ResnetBlock(dim_in, dim_in, time_emb_dim=dim),
                     Residual(Rezero(LinearAttention(dim_in))),
                     Upsample(dim_in)]))
        self.final_block = Block(dim, dim)
        self.final_conv = torch.nn.Conv2d(dim, 1, 1)

    def forward(self, yt, y_masks, mu, t):
        t = self.time_pos_emb(t, scale=self.pe_scale) # (B, dim)
        t = self.mlp(t) # (B, dim)

        yt = torch.stack([mu, yt], 1) # (B, 2, 80, T)
        y_masks = y_masks.unsqueeze(1) # (B, 1, 1, T)

        hiddens = []
        masks = [y_masks]
        for resnet1, resnet2, attn, downsample in self.downs:
            mask_down = masks[-1]
            yt = resnet1(yt, mask_down, t)
            yt = resnet2(yt, mask_down, t)
            yt = attn(yt)
            hiddens.append(yt)
            yt = downsample(yt * mask_down)
            masks.append(mask_down[:, :, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]
        yt = self.mid_block1(yt, mask_mid, t)
        yt = self.mid_attn(yt)
        yt = self.mid_block2(yt, mask_mid, t) # (B, 256, 80/4, T/4)

        for resnet1, resnet2, attn, upsample in self.ups:
            mask_up = masks.pop()
            yt = torch.cat((yt, hiddens.pop()), dim=1)
            yt = resnet1(yt, mask_up, t)
            yt = resnet2(yt, mask_up, t)
            yt = attn(yt)
            yt = upsample(yt * mask_up)

        yt = self.final_block(yt, y_masks) # (B, 64, 80, T)
        output = self.final_conv(yt * y_masks) # (B, 1, 80, T)

        return (output * y_masks).squeeze(1)


class Diffusion(torch.nn.Module):
    def __init__(self, ddim, beta_min, beta_max, pe_scale):
        super().__init__()
        self.estimator = Estimator(ddim, pe_scale=pe_scale)
        self.ddim = ddim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale

    def forward_diffusion(self, ys, y_masks, mu, cum_noise):
        tmp = torch.exp(-0.5 * cum_noise)
        mean = ys * tmp + mu * (1.0 - tmp)
        variance = 1.0 - torch.exp(-cum_noise)
        z = torch.randn(ys.shape, dtype=ys.dtype, device=ys.device, requires_grad=False)
        yt = mean + z * torch.sqrt(variance)

        return yt * y_masks, z * y_masks

    def reverse_diffusion(self, z, mask, mu, timesteps, length):
        h = 1.0 / timesteps
        xt = z * mask  # (B, 80, T)
        import numpy as np
        np.save("/nolan/inference/gradtts_step_00.mel.npy", mu[0, :, :length].transpose(0, 1).data.cpu().numpy())
        for i in range(timesteps):
            t = (1.0 - (i + 0.5) * h) * torch.ones(z.shape[0], dtype=z.dtype, device=z.device)  # (B,)
            time = t.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
            noise_t = self.get_noise(time, self.beta_min, self.beta_max, cumulative=False)  # (B, 1, 1)
            dxt = 0.5 * (mu - xt - self.estimator(xt, mask, mu, t))
            dxt = dxt * noise_t * h
            xt = (xt - dxt) * mask
            np.save("/nolan/inference/gradtts_step_{:02d}.mel.npy".format(i+1), xt[0, :, :length].transpose(0, 1).data.cpu().numpy())

        return xt

    def forward(self, ys, y_masks, mu, offset=1e-5):
        t = torch.rand(ys.shape[0], dtype=ys.dtype,
                       device=ys.device, requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)

        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = self.get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        yt, z = self.forward_diffusion(ys, y_masks, mu, cum_noise)
        noise_estimation = self.estimator(yt, y_masks, mu, t)
        noise_estimation *= torch.sqrt(1.0 - torch.exp(-cum_noise))

        return noise_estimation, z

    def inference(self, z, mask, mu, timesteps, length):
        return self.reverse_diffusion(z, mask, mu, timesteps, length)

    def get_noise(self, time, beta_init, beta_term, cumulative=False):
        if cumulative:
            noise = beta_init * time + 0.5 * (beta_term - beta_init) * (time ** 2)
        else:
            noise = beta_init + (beta_term - beta_init) * time
        return noise

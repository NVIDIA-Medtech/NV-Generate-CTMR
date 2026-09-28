from __future__ import annotations

import types

import torch
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import DiffusionModelUNetMaisi
from monai.networks.nets.diffusion_model_unet import DiffusionUNetResnetBlock
from monai.utils.type_conversion import convert_to_tensor
from torch import nn


def _adagn_resnet_forward(self: DiffusionUNetResnetBlock, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
    h = x
    h = self.norm1(h)
    h = self.nonlinearity(h)

    if self.upsample is not None:
        x = self.upsample(x)
        h = self.upsample(h)
    elif self.downsample is not None:
        x = self.downsample(x)
        h = self.downsample(h)

    h = self.conv1(h)

    if self.spatial_dims == 2:
        scale_shift = self.time_emb_proj(self.nonlinearity(emb))[:, :, None, None]
    else:
        scale_shift = self.time_emb_proj(self.nonlinearity(emb))[:, :, None, None, None]
    scale, shift = scale_shift.chunk(2, dim=1)

    h = self.norm2(h)
    h = h * (1.0 + scale) + shift
    h = self.nonlinearity(h)
    h = self.conv2(h)
    output: torch.Tensor = self.skip_connection(x) + h
    return output


class DiffusionModelUNetMaisiAdaGN(DiffusionModelUNetMaisi):
    """MAISI UNet with condition-aware adaptive GroupNorm in every ResBlock.

    The base MAISI UNet already passes timestep/spacing embeddings into each ResBlock and sends `context` through
    cross-attention only at configured attention levels. This variant keeps that cross-attention path and appends a
    learned embedding of the same 19-D conditioning vector to the ResBlock embedding. Each ResBlock then predicts
    `(scale, shift)` and applies `GroupNorm(h) * (1 + scale) + shift`, making the size condition available at all
    resolutions instead of only inside the deeper attention blocks.
    """

    def __init__(
        self,
        *args,
        adagn_condition_dim: int | None = None,
        adagn_context_embed_dim: int | None = None,
        **kwargs,
    ) -> None:
        cross_attention_dim = kwargs.get("cross_attention_dim")
        super().__init__(*args, **kwargs)
        if adagn_condition_dim is None:
            if cross_attention_dim is None:
                raise ValueError("adagn_condition_dim is required when cross_attention_dim is not set")
            adagn_condition_dim = int(cross_attention_dim)
        if adagn_context_embed_dim is None:
            adagn_context_embed_dim = int(self.block_out_channels[0]) * 4
        self.adagn_condition_dim = int(adagn_condition_dim)
        self.adagn_context_embed_dim = int(adagn_context_embed_dim)
        self.adagn_context_embed = nn.Sequential(
            nn.Linear(self.adagn_condition_dim, self.adagn_context_embed_dim),
            nn.SiLU(),
            nn.Linear(self.adagn_context_embed_dim, self.adagn_context_embed_dim),
        )
        self._upgrade_resblocks_to_adagn(self.adagn_context_embed_dim)

    def _upgrade_resblocks_to_adagn(self, extra_emb_dim: int) -> None:
        for module in self.modules():
            if not isinstance(module, DiffusionUNetResnetBlock):
                continue
            old_proj = module.time_emb_proj
            old_in = old_proj.in_features
            new_proj = nn.Linear(old_in + extra_emb_dim, module.out_channels * 2)
            with torch.no_grad():
                scale_weight = new_proj.weight[: module.out_channels]
                scale_bias = new_proj.bias[: module.out_channels]
                shift_weight = new_proj.weight[module.out_channels :]
                shift_bias = new_proj.bias[module.out_channels :]
                scale_weight[:, :old_in].zero_()
                scale_bias.zero_()
                shift_weight[:, :old_in].copy_(old_proj.weight)
                shift_bias.copy_(old_proj.bias)
            module.time_emb_proj = new_proj
            module.emb_channels = old_in + extra_emb_dim
            module.forward = types.MethodType(_adagn_resnet_forward, module)

    def _context_embedding(self, x: torch.Tensor, context: torch.Tensor | None) -> torch.Tensor:
        if context is None:
            cond = x.new_full((x.shape[0], self.adagn_condition_dim), -1.0)
        elif context.ndim == 3:
            cond = context.mean(dim=1)
        elif context.ndim == 2:
            cond = context
        else:
            raise ValueError(f"context must have shape (B,C) or (B,T,C), got {tuple(context.shape)}")
        if cond.shape[-1] != self.adagn_condition_dim:
            raise ValueError(f"context dim {cond.shape[-1]} != adagn_condition_dim {self.adagn_condition_dim}")
        return self.adagn_context_embed(cond.to(device=x.device, dtype=x.dtype))

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
        down_block_additional_residuals: tuple[torch.Tensor] | None = None,
        mid_block_additional_residual: torch.Tensor | None = None,
        top_region_index_tensor: torch.Tensor | None = None,
        bottom_region_index_tensor: torch.Tensor | None = None,
        spacing_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = self._get_time_and_class_embedding(x, timesteps, class_labels)
        emb = self._get_input_embeddings(emb, top_region_index_tensor, bottom_region_index_tensor, spacing_tensor)
        emb = torch.cat((emb, self._context_embedding(x, context)), dim=1)

        h = self.conv_in(x)
        h, down_block_res_samples = self._apply_down_blocks(h, emb, context, down_block_additional_residuals)
        h = self.middle_block(h, emb, context)
        if mid_block_additional_residual is not None:
            h += mid_block_additional_residual
        h = self._apply_up_blocks(h, emb, context, down_block_res_samples)
        h = self.out(h)
        h_tensor: torch.Tensor = convert_to_tensor(h)
        return h_tensor

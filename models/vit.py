# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from typing import Optional
import torch
import torch.nn as nn

import timm
from transformers import AutoModel
from transformers.models.dinov3_vit import DINOv3ViTConfig, DINOv3ViTModel


class ViT(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        patch_size=16,
        backbone_name="vit_large_patch14_reg4_dinov2",
        ckpt_path: Optional[str] = None,
        local_backbone_init: bool = False,
        backbone_ckpt_path: Optional[str] = None,
    ):
        super().__init__()

        if local_backbone_init and backbone_name.startswith("facebook/dinov3-"):
            backbone = self.local_dinov3_model(backbone_name, img_size)
            if backbone_ckpt_path is not None:
                converted_state_dict = self.convert_local_dinov3_checkpoint(
                    backbone_ckpt_path
                )
                incompatible = backbone.load_state_dict(
                    converted_state_dict, strict=False
                )
                if incompatible.missing_keys:
                    raise ValueError(
                        f"Missing keys when loading local DINOv3 checkpoint: {incompatible.missing_keys}"
                    )
                unexpected_keys = [
                    key
                    for key in incompatible.unexpected_keys
                    if key != "mask_token"
                ]
                if unexpected_keys:
                    raise ValueError(
                        f"Unexpected keys when loading local DINOv3 checkpoint: {unexpected_keys}"
                    )
            self.backbone = self.transformers_to_timm(backbone, img_size)
        elif "/" in backbone_name:
            self.backbone = self.transformers_to_timm(
                AutoModel.from_pretrained(
                    backbone_name,
                ),
                img_size,
            )
        else:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=ckpt_path is None,
                img_size=img_size,
                patch_size=patch_size,
                num_classes=0,
            )

        pixel_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

    def local_dinov3_model(self, backbone_name: str, img_size: tuple[int, int]):
        image_size = img_size[0] if img_size[0] == img_size[1] else max(img_size)

        configs = {
            "facebook/dinov3-vits16-pretrain-lvd1689m": dict(
                hidden_size=384,
                intermediate_size=1536,
                num_hidden_layers=12,
                num_attention_heads=6,
                num_register_tokens=4,
                image_size=image_size,
            ),
            "facebook/dinov3-vitb16-pretrain-lvd1689m": dict(
                hidden_size=768,
                intermediate_size=3072,
                num_hidden_layers=12,
                num_attention_heads=12,
                num_register_tokens=4,
                image_size=image_size,
            ),
            "facebook/dinov3-vitl16-pretrain-lvd1689m": dict(
                hidden_size=1024,
                intermediate_size=4096,
                num_hidden_layers=24,
                num_attention_heads=16,
                num_register_tokens=4,
                image_size=image_size,
            ),
        }

        if backbone_name not in configs:
            raise ValueError(f"Unsupported local DINOv3 backbone: {backbone_name}")

        return DINOv3ViTModel(DINOv3ViTConfig(**configs[backbone_name]))

    def convert_local_dinov3_checkpoint(self, backbone_ckpt_path: str):
        state_dict = torch.load(backbone_ckpt_path, map_location="cpu", weights_only=True)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        converted = {}
        for key, value in state_dict.items():
            layer_key = key.replace("blocks.", "layer.")
            if key == "cls_token":
                converted["embeddings.cls_token"] = value
            elif key == "mask_token":
                converted["embeddings.mask_token"] = value[:, None, :]
            elif key == "storage_tokens":
                converted["embeddings.register_tokens"] = value
            elif key == "patch_embed.proj.weight":
                converted["embeddings.patch_embeddings.weight"] = value
            elif key == "patch_embed.proj.bias":
                converted["embeddings.patch_embeddings.bias"] = value
            elif key == "norm.weight":
                converted["norm.weight"] = value
            elif key == "norm.bias":
                converted["norm.bias"] = value
            elif key == "rope_embed.periods":
                continue
            elif ".attn.qkv.weight" in key:
                prefix = layer_key.replace(".attn.qkv.weight", ".attention")
                q_weight, k_weight, v_weight = value.chunk(3, dim=0)
                converted[f"{prefix}.q_proj.weight"] = q_weight
                converted[f"{prefix}.k_proj.weight"] = k_weight
                converted[f"{prefix}.v_proj.weight"] = v_weight
            elif ".attn.qkv.bias_mask" in key:
                continue
            elif ".attn.qkv.bias" in key:
                prefix = layer_key.replace(".attn.qkv.bias", ".attention")
                q_bias, k_bias, v_bias = value.chunk(3, dim=0)
                converted[f"{prefix}.q_proj.bias"] = q_bias
                converted[f"{prefix}.v_proj.bias"] = v_bias
            elif ".attn.proj.weight" in key:
                converted[layer_key.replace(".attn.proj", ".attention.o_proj")] = value
            elif ".attn.proj.bias" in key:
                converted[layer_key.replace(".attn.proj", ".attention.o_proj")] = value
            elif ".ls1.gamma" in key:
                converted[layer_key.replace(".ls1.gamma", ".layer_scale1.lambda1")] = value
            elif ".ls2.gamma" in key:
                converted[layer_key.replace(".ls2.gamma", ".layer_scale2.lambda1")] = value
            elif ".mlp.fc1.weight" in key:
                converted[layer_key.replace(".mlp.fc1", ".mlp.up_proj")] = value
            elif ".mlp.fc1.bias" in key:
                converted[layer_key.replace(".mlp.fc1", ".mlp.up_proj")] = value
            elif ".mlp.fc2.weight" in key:
                converted[layer_key.replace(".mlp.fc2", ".mlp.down_proj")] = value
            elif ".mlp.fc2.bias" in key:
                converted[layer_key.replace(".mlp.fc2", ".mlp.down_proj")] = value
            elif key.startswith("blocks."):
                converted[layer_key] = value

        return converted

    def transformers_to_timm(self, backbone, img_size: tuple[int, int]):
        backbone.patch_embed = backbone.embeddings
        backbone.patch_embed.patch_size = (
            backbone.embeddings.config.patch_size,
            backbone.embeddings.config.patch_size,
        )
        backbone.patch_embed.grid_size = (
            img_size[0] // backbone.embeddings.config.patch_size,
            img_size[1] // backbone.embeddings.config.patch_size,
        )

        backbone.embed_dim = backbone.embeddings.config.hidden_size
        backbone.num_prefix_tokens = backbone.patch_embed.config.num_register_tokens + 1
        backbone.blocks = backbone.layer

        del (
            backbone.patch_embed.mask_token,
            backbone.embeddings,
            backbone.layer,
        )

        return backbone

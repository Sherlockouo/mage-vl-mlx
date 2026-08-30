from typing import Optional

import mlx.core as mx
import mlx.nn as nn

try:
    from ..base import InputEmbeddingsFeatures
    from ..qwen3.language import LanguageModel
except ImportError:  # 顶级导入
    from mlx_vlm.models.base import InputEmbeddingsFeatures
    from mlx_vlm.models.qwen3.language import LanguageModel
from .config import ModelConfig
from .vision import VisionModel


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)

    @property
    def layers(self):
        return self.language_model.model.layers

    def _vision_features(self, pixel_values, kwargs):
        grid_thw = kwargs.get("image_grid_thw", kwargs.get("video_grid_thw"))
        patch_positions = kwargs.get("patch_positions")
        if grid_thw is None or patch_positions is None:
            raise ValueError(
                "Mage-VL vision path requires 'image_grid_thw' (or 'video_grid_thw') "
                "and 'patch_positions' in the processor output."
            )
        grid_thw = mx.array(grid_thw) if not isinstance(grid_thw, mx.array) else grid_thw
        patch_positions = (
            mx.array(patch_positions)
            if not isinstance(patch_positions, mx.array)
            else patch_positions
        )
        if patch_positions.ndim == 3:
            patch_positions = patch_positions[0]
        dtype = self.vision_tower.patch_embedding.weight.dtype
        pixel_values = pixel_values.astype(dtype)
        return self.vision_tower(pixel_values, grid_thw, patch_positions)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        if pixel_values is None:
            pixel_values = kwargs.get("pixel_values_videos", None)

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        if pixel_values is None:
            return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

        image_features = self._vision_features(pixel_values, kwargs)
        merged = self.merge_input_ids_with_image_features(
            self.config.image_token_id,
            self.config.video_token_id,
            image_features,
            inputs_embeds,
            input_ids,
        )
        return InputEmbeddingsFeatures(inputs_embeds=merged)

    @staticmethod
    def merge_input_ids_with_image_features(
        image_token_id, video_token_id, image_features, inputs_embeds, input_ids
    ):
        positions = input_ids == image_token_id
        if mx.sum(positions).item() == 0:
            positions = input_ids == video_token_id

        image_features = image_features.astype(inputs_embeds.dtype)
        batch_outputs = []
        start = 0
        for b in range(input_ids.shape[0]):
            mask = positions[b]
            n = int(mx.sum(mask).item())
            if n == 0:
                batch_outputs.append(inputs_embeds[b])
                continue
            feats = image_features[start : start + n]
            cumsum = mx.cumsum(mask.astype(mx.int32))
            idx = mx.where(mask, cumsum - 1, 0)
            gathered = feats[idx]
            batch_outputs.append(
                mx.where(mx.expand_dims(mask, -1), gathered, inputs_embeds[b])
            )
            start += n
        return mx.stack(batch_outputs, axis=0)

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        feats = self.get_input_embeddings(input_ids, pixel_values, **kwargs)
        return self.language_model(
            inputs=input_ids,
            cache=cache,
            inputs_embeds=feats.inputs_embeds,
            mask=mask,
        )

    def sanitize(self, weights):
        out = {}
        for k, v in weights.items():
            nk = k
            if k.startswith("model.visual."):
                inner = k[len("model.visual.") :]
                if inner.startswith("embeddings.patch_embedding"):
                    nk = "vision_tower.patch_embedding.weight"
                    v = v.reshape(v.shape[0], -1)  # conv [O,C,ph,pw] -> linear [O,C*ph*pw]
                elif inner.startswith("encoder.layers."):
                    nk = "vision_tower.layers." + inner[len("encoder.layers.") :]
                else:
                    nk = "vision_tower." + inner
            elif k.startswith("model.language_model."):
                nk = "language_model.model." + k[len("model.language_model.") :]
            elif k == "lm_head.weight":
                nk = "language_model.lm_head.weight"
            out[nk] = v
        return out

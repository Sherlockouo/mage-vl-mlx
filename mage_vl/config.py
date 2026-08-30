import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

try:
    from ..base import BaseModelConfig
except ImportError:  # 顶级导入(插件未装入 mlx_vlm 命名空间时)
    from mlx_vlm.models.base import BaseModelConfig


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "mage_vl_vision"
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int = 448
    patch_size: int = 16
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    layer_norm_type: str = "layer_norm"
    rope_theta: float = 10000.0
    out_hidden_size: int = 2560
    spatial_merge_size: int = 2
    temporal_patch_size: int = 1
    frame_windows_size: int = 4
    use_head: bool = False
    use_patch_position_encoding: bool = False
    patch_position_encoding_type: str = "absolute"
    max_position_embeddings: int = 8192


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "qwen3"
    hidden_size: int = 2560
    num_hidden_layers: int = 36
    intermediate_size: int = 9728
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    head_dim: int = 128
    max_position_embeddings: int = 262144
    rope_theta: float = 5000000.0
    tie_word_embeddings: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    @classmethod
    def from_dict(cls, params):
        params = dict(params or {})
        # config.json nests rope_theta under text_config.rope_parameters
        rope_params = params.get("rope_parameters")
        if rope_params and "rope_theta" not in params and "rope_theta" in rope_params:
            params["rope_theta"] = rope_params["rope_theta"]
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str = "mage_vl"
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    vocab_size: int = 151936
    eos_token_id: Optional[List[int]] = None

    @classmethod
    def from_dict(cls, params):
        params = dict(params)
        params["text_config"] = TextConfig.from_dict(params.get("text_config", {}))
        params["vision_config"] = VisionConfig.from_dict(params.get("vision_config", {}))
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )

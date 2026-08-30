"""Ensure the plugin is discoverable by ``mlx_vlm.utils.load_model``.

mlx-vlm resolves models dynamically as ``mlx_vlm.models.<model_type>``. This
helper symlinks the ``mage_vl`` package into the installed mlx-vlm's models
directory (idempotent, best-effort)."""
from __future__ import annotations

import sys
from pathlib import Path


def ensure_plugin_installed() -> Path:
    import mlx_vlm

    target = Path(mlx_vlm.__file__).resolve().parent / "models" / "mage_vl"
    if target.exists():
        return target
    src = Path(__file__).resolve().parent
    try:
        target.symlink_to(src)
        print(f"[mage_vl] plugin linked: {target} -> {src}", file=sys.stderr)
    except OSError as exc:
        raise RuntimeError(
            "无法把 mage_vl 插件链接进 mlx_vlm(权限不足?)。"
            f"请手动执行:\n  ln -sfn {src} {target}"
        ) from exc
    return target

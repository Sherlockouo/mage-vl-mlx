#!/usr/bin/env python3
"""Link the mage_vl plugin into the installed mlx-vlm (idempotent).

mlx-vlm discovers models as ``mlx_vlm.models.<model_type>`` — until the plugin
is linked there, ``load_model()`` cannot find Mage-VL. ``python -m mage_vl.server``
and ``python -m mage_vl.cli`` run this automatically; use this script to do it
standalone (e.g. right after ``pip install -e .``).

    python scripts/install_plugin.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mage_vl._link import ensure_plugin_installed  # noqa: E402

if __name__ == "__main__":
    target = ensure_plugin_installed()
    print(f"plugin OK: {target}")

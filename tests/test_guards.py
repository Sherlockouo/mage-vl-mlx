"""Unit tests for engine guards + severity parsing (no model/weights needed;
mlx must be importable — any Apple Silicon env with `pip install mlx`)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

mlx = pytest.importorskip("mlx")  # noqa: E402

from mage_vl.engine import MageVLEngine  # noqa: E402
from mage_vl.severity import parse_severity  # noqa: E402


def test_loop_guard_catches_long_phrase():
    toks = [1, 2, 3] + [10, 11, 12, 13, 14, 15, 16] * 50
    assert MageVLEngine._in_loop(toks)


def test_loop_guard_catches_short_cycle():
    assert MageVLEngine._in_loop([5] * 12 + [5, 6] * 6)


def test_loop_guard_ignores_normal_text():
    assert not MageVLEngine._in_loop(list(range(1, 40)))


def test_loop_guard_needs_three_cycles():
    base = list(range(100, 112))
    cyc = [7, 8, 9, 10, 11, 12, 13, 14]
    assert not MageVLEngine._in_loop(base + cyc + cyc)  # 2 cycles: keep
    assert MageVLEngine._in_loop(base + cyc * 3)  # 3 cycles: loop


def test_rep_penalty_shapes():
    import mlx.core as mx

    logits = mx.zeros((1, 16))
    out = MageVLEngine._apply_rep_penalty(logits, [3, 3, 7])
    assert out.shape == (1, 16)
    assert float(out[0, 3]) == pytest.approx(0.0)  # zeros stay zero; shape path works


def test_severity():
    assert parse_severity("描述\n告警:紧急") == "紧急"
    assert parse_severity("告警：注意") == "注意"
    assert parse_severity("现场有明火") == "紧急"
    assert parse_severity("正常车流") == "无"

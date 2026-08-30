"""Incremental streaming detokenizer for the native (transformers-free) processor.

Adapts ``mlx_lm.tokenizer_utils.NaiveStreamingDetokenizer`` semantics to the
plain ``tokenizers.Tokenizer`` used by :class:`MageVLProcessor`, so generation
code can stream readable text without re-decoding the whole history per token,
and without importing HuggingFace ``transformers``.

Contract: ``text`` grows monotonically (byte-level BPE prefixes are stable);
consumers read ``last_segment`` repeatedly — each call returns the exact new
slice appended since the previous read, so concatenating segments reproduces
the full text. Pending buffers whose decode ends in U+FFFD (an incomplete
multi-byte character split across tokens) are held back instead of emitting a
broken char that would later change under it.
"""
from __future__ import annotations

_WINDOW_LIMIT = 48


class StreamingDetokenizer:
    """Decode generated tokens one by one; read ``last_segment`` for new text."""

    __slots__ = ("tokens", "_tok", "_final", "_pending", "_offset")

    def __init__(self, tokenizer):
        self._tok = tokenizer
        self.reset()

    def reset(self) -> None:
        self.tokens: list[int] = []
        self._final = ""            # bytes known-final: committed text
        self._pending: list[int] = []  # tokens since last commit point
        self._offset = 0            # index into ``text`` consumed by the reader

    def add_token(self, token: int) -> None:
        self._pending.append(token)
        self.tokens.append(token)

    def finalize(self) -> None:
        if self._pending:
            self._final += self._visible()
            self._pending = []

    def _visible(self) -> str:
        piece = self._tok.decode(self._pending, skip_special_tokens=True)
        if piece.endswith("\ufffd"):
            piece = piece[:-1]
        return piece

    @property
    def text(self) -> str:
        """Full decoded text so far (monotonic)."""
        return self._final + self._visible()

    @property
    def last_segment(self) -> str:
        """New text since this property was last accessed."""
        t = self.text
        seg = t[self._offset:]
        self._offset = len(t)
        # progress guarantee: clear pending when the visible chunk looks
        # complete so the buffer does not grow without bound within a line
        if self._pending and len(self._pending) >= _WINDOW_LIMIT:
            self.finalize()
            t = self.text
            self._offset = min(self._offset, len(t))
        return seg

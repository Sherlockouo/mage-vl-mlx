"""Alert-severity parsing for monitoring-style prompts.

The model is asked to end answers with an explicit ``告警:无|注意|紧急`` line;
parsing falls back to a keyword scan so a malformed answer can never silently
swallow a real alert.
"""
from __future__ import annotations

import re

SEVERITY_RE = re.compile(r"告警\s*[:：]\s*(无|注意|紧急)")

EMERGENCY_KW = (
    "明火", "火灾", "起火", "火焰", "爆炸", "事故", "碰撞", "相撞", "追尾",
    "翻车", "车祸", "倒塌", "坍塌", "洪水", "内涝", "滑坡", "泥石流", "被困",
    "溺水", "翻覆", "坠", "横穿", "闯入",
)
WARN_KW = (
    "烟雾", "浓烟", "烟", "拥堵", "缓行", "逆行", "违停", "占道", "遗撒",
    "积水", "裂缝", "裂开", "倾斜", "变形", "拥挤", "聚集", "奔跑", "跌倒",
    "摔倒", "徘徊", "异常",
)


def parse_severity(text: str) -> str:
    """Extract ``无|注意|紧急`` from the model answer (explicit line first,
    keyword fallback second)."""
    m = SEVERITY_RE.search(text or "")
    if m:
        return m.group(1)
    low = text or ""
    if any(k in low for k in EMERGENCY_KW):
        return "紧急"
    if any(k in low for k in WARN_KW):
        return "注意"
    return "无"

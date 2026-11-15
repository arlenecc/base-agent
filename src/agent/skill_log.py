"""Structured skill activity log (JSONL, rotating).

技能检索 / 加载 / 保存 / 索引同步 / 学习门槛判定都写到这里，便于事后排查
「为什么没召回」「这个技能用过吗」「为什么没有固化」。

落盘位置：``<workspace>/.agent/logs/skills.log``，单文件 2MB 轮转、保留 3 份。
日志绝不能影响技能主流程：所有写入异常都在本模块内吞掉。
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional

_LOGGER_NAME = "agent.skills.activity"
_LOG_ENV = "BASE_AGENT_SKILL_LOG"
_MAX_FIELD_CHARS = 500
_MAX_BYTES = 2_000_000
_BACKUP_COUNT = 3

# 只脱敏明显的凭据形态；普通查询文本保留，便于排查召回问题。
_SECRET_RE = re.compile(
    r"(?i)((?:api[_-]?key|token|secret|password|authorization)\s*[:=]\s*)\S+"
    r"|\bsk-[A-Za-z0-9]{8,}\b"
)

_lock = threading.Lock()
_active_path: Optional[str] = None


def skill_log_path(workspace: str) -> str:
    return os.path.join(os.path.abspath(workspace), ".agent", "logs", "skills.log")


def _ensure_handler(workspace: str) -> None:
    """把 logger 指向当前工作区的日志文件；切换工作区时换 handler。"""
    global _active_path
    path = skill_log_path(workspace)
    with _lock:
        logger = logging.getLogger(_LOGGER_NAME)
        if _active_path == path and logger.handlers:
            return
        # 同一 logger 不能同时挂多个工作区的 handler，否则 A 的事件会写进 B。
        for h in list(logger.handlers):
            try:
                logger.removeHandler(h)
                h.close()
            except Exception:
                pass
        os.makedirs(os.path.dirname(path), exist_ok=True)
        handler = RotatingFileHandler(
            path, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False   # 不进入 root/UI 日志，避免刷屏
        _active_path = path


def _clip(value: Any) -> Any:
    if isinstance(value, str):
        value = _SECRET_RE.sub(
            lambda m: (m.group(1) + "[redacted]") if m.group(1) else "[redacted]",
            value,
        )
        if len(value) > _MAX_FIELD_CHARS:
            value = value[:_MAX_FIELD_CHARS] + "…"
    return value


def _enabled() -> bool:
    """环境变量开关：BASE_AGENT_SKILL_LOG=0/false/no/off 时关闭技能日志。"""
    raw = os.environ.get(_LOG_ENV, "1").strip().lower()
    return raw not in ("0", "false", "no", "off", "disabled")


def log_skill_event(workspace: str, event: str, **fields: Any) -> None:
    """写一条结构化技能日志；任何失败都静默忽略。"""
    if not workspace or not _enabled():
        return
    try:
        _ensure_handler(workspace)
        record: Dict[str, Any] = {
            "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
            "event": event,
        }
        for k, v in fields.items():
            record[k] = _clip(v)
        logging.getLogger(_LOGGER_NAME).info(
            json.dumps(record, ensure_ascii=False, default=str)
        )
    except Exception:
        # 日志是旁路能力：磁盘满 / 权限不足都不该让检索或保存失败。
        pass

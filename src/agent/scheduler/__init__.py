"""定时任务（APScheduler）。

对外只暴露 :class:`SchedulerService` 与模块级单例存取函数：工具层通过
``get_scheduler()`` 拿到 UI 启动的服务；UI 在窗口初始化时创建并注入执行器。
"""
from __future__ import annotations

from .service import SchedulerError, SchedulerService, get_scheduler, set_scheduler

__all__ = [
    "SchedulerService", "SchedulerError", "get_scheduler", "set_scheduler",
]

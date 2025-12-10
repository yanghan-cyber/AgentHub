import atexit
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from loguru import logger

# 导入你的配置和上下文
from config.settings import settings

from .context import (
    get_current_api_path,
    get_current_env,
    get_current_trace_id,
    get_current_user_name,
    get_current_user_type,
)


# ==========================================
# 1. 核心修复：Context Patcher
# ==========================================
def context_patcher(record):
    """
    Loguru Patcher: 
    1. 注入 ContextVars (TraceID, UserID等)
    2. 处理 Logger Name (支持 get_logger("custom_name"))
    """
    # 注入 ContextVars
    record["extra"]["trace_id"] = get_current_trace_id()
    record["extra"]["user_name"] = get_current_user_name()
    record["extra"]["user_type"] = get_current_user_type()
    record["extra"]["api_path"] = get_current_api_path()
    record["extra"]["env"] = get_current_env()
# 2. 处理 Logger Name
    # 逻辑：如果没有手动 bind 名字，我们自己算一个漂亮的名字
    if "custom_name" not in record["extra"]:
        
        # 获取日志来源文件的绝对路径
        file_path = Path(record["file"].path)
        
        # 尝试计算相对于项目根目录的路径
        # 例如: /home/user/project/app/main.py -> app.main
        try:
            # relative_to 需要 settings.MEMOS_DIR 设置正确 (指向项目根目录)
            relative_path = file_path.relative_to(settings.MEMOS_DIR)
            
            # 将路径转换为点分模块名 (例如 app/utils/tools.py -> app.utils.tools)
            # with_suffix('') 去掉 .py
            # parts 拆分路径，'.'.join 重新组合
            module_name = "/".join(relative_path.with_suffix("").parts)
            record["extra"]["custom_name"] = module_name
            
        except ValueError:
            # 如果文件不在 MEMOS_DIR 目录下 (比如是第三方库)，或者计算失败
            # 退回到默认的 record["name"]
            # 如果是 __main__，我们可以强制显示文件名
            if record["name"] == "__main__":
                record["extra"]["custom_name"] = file_path.stem # 只显示文件名(不含后缀)
            else:
                record["extra"]["custom_name"] = record["name"]

# ==========================================
# 2. 过滤器逻辑 (复刻原配置的 Log Filter)
# ==========================================
def console_filter(record):
    """
    控制台过滤器：复刻 package_tree_filter
    只允许:
    1. 属于 settings.LOG_FILTER_TREE_PREFIX (如 "memos") 的日志
    2. 或者 级别 >= WARNING 的日志 (即使是第三方库的报错，控制台也该显示)
    """
    # 如果配置了前缀过滤 (例如 "memos")
    prefix = getattr(settings, "LOG_FILTER_TREE_PREFIX", None)
    
    if prefix:
        # 如果日志名字以 prefix 开头，允许通过
        if record["name"].startswith(prefix):
            return True
        # 如果是第三方库的 ERROR/CRITICAL，也放行 (防止漏掉关键错误)
        if record["level"].no >= logging.WARNING:
            return True
        # 否则屏蔽 (比如 urllib3 的 INFO)
        return False
        
    return True

# ==========================================
# 3. HTTP Sink (复刻 CustomLoggerRequestHandler)
# ==========================================
class AsyncHTTPSink:
    def __init__(self):
        self.url = os.getenv("CUSTOM_LOGGER_URL")
        self.token = os.getenv("CUSTOM_LOGGER_TOKEN")
        self.enabled = bool(self.url)
        
        if self.enabled:
            workers = int(os.getenv("CUSTOM_LOGGER_WORKERS", "2"))
            self.executor = ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="log_sender"
            )
            self.session = requests.Session()
            atexit.register(self.cleanup)

    def write(self, message):
        if not self.enabled: return
        record = message.record
        if record["level"].no < 20: return # INFO以上

        self.executor.submit(self._send_log_sync, record)

    def _send_log_sync(self, record):
        try:
            extra = record["extra"]
            payload = {
                "message": record["message"],
                "trace_id": extra.get("trace_id"),
                "action": extra.get("api_path"),
                "current_time": round(time.time(), 3),
                "env": extra.get("env"),
                "user_type": extra.get("user_type"),
                "user_name": extra.get("user_name"),
                "level": record["level"].name,
                "module": extra.get("custom_name", record["name"]), 
                "line": record["line"]
            }

            for key, value in os.environ.items():
                if key.startswith("CUSTOM_LOGGER_ATTRIBUTE_"):
                    attr_key = key[len("CUSTOM_LOGGER_ATTRIBUTE_"):].lower()
                    payload[attr_key] = value

            headers = {
                "Content-Type": "application/json",
                "traceId": str(extra.get("trace_id"))
            }
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"

            self.session.post(self.url, headers=headers, json=payload, timeout=5)
        except Exception:
            pass

    def cleanup(self):
        if self.enabled:
            try:
                self.executor.shutdown(wait=False)
                self.session.close()
            except Exception:
                pass

# ==========================================
# 4. 拦截标准日志 Handler
# ==========================================
class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        logging_file = os.path.normcase(logging.__file__)
        this_file = os.path.normcase(__file__)

        while frame:
            frame_file = os.path.normcase(frame.f_code.co_filename)
            is_logging = (frame_file == logging_file) or (frame_file == this_file) or ("logging" in frame_file and "lib" in frame_file)
            if is_logging:
                frame = frame.f_back
                depth += 1
                continue
            else:
                break

        logger.opt(depth=depth-2, exception=record.exc_info).bind(custom_name=record.name).log(level, record.getMessage())

def intercept_standard_logging():
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

# ==========================================
# 5. 初始化配置 (复刻 dictConfig 逻辑)
# ==========================================
def setup_logger():
    logger.remove()
    
    # --- 格式定义 (复刻 formatters) ---
    
    # 1. Console 格式 (对应 "no_datetime" + "simplified")
    # 通常 Console 不需要日期(Systemd会加)，且需要颜色
    console_fmt = (
        "<cyan>{extra[trace_id]}</cyan> | "
        "path=<magenta>{extra[api_path]}</magenta> | "
        "<magenta>{extra[custom_name]}:{line}</magenta> | " 
        "<level>{message}</level>"
    )

    # 2. File 格式 (对应 "standard")
    # 文件需要全量信息：时间、环境、用户类型等
    file_fmt = (
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{extra[trace_id]} | "
        "path={extra[api_path]} | "
        "env={extra[env]} | "
        "user_type={extra[user_type]} | "
        "user_name={extra[user_name]} | "
        "{extra[custom_name]} - {level} - {file}:{line} - {function} - {message}"
    )

    # --- Handlers 配置 ---

    # [Handler 1] Console
    # 特性：启用 console_filter (只看本项目日志)，简化格式
    logger.add(
        sys.stdout, 
        format=console_fmt, 
        level="DEBUG" if settings.DEBUG else "INFO",
        filter=console_filter, # <--- 关键：应用过滤
        colorize=True
    )
    
    # [Handler 2] File (复刻 ConcurrentTimedRotatingFileHandler)
    # 特性：记录所有日志(无过滤)，详细格式，每天轮转，保留3天(backupCount=3)
    log_file_path = settings.MEMOS_DIR / "logs" / "app.log"
    logger.add(
        str(log_file_path),
        rotation="00:00",    # midnight
        retention="3 days",  # backupCount=3
        level="DEBUG",
        format=file_fmt,     # standard format
        encoding="utf-8",
        enqueue=True
    )

    # [Handler 3] HTTP Sink (复刻 custom_logger)
    http_sink = AsyncHTTPSink()
    if http_sink.enabled:
        logger.add(http_sink.write, level="INFO", enqueue=True)

    # --- 全局配置 ---
    logger.configure(patcher=context_patcher)
    intercept_standard_logging()
    
    return logger

def get_logger(name: str | None = None):
    if name:
        return logger.bind(custom_name=name)
    return logger

setup_logger()
__all__ = ["logger", "get_logger"]
# context.py
import uuid
from contextvars import ContextVar
from typing import Optional

# 1. 定义 ContextVars
# 这些变量是“协程/线程局部”的。在不同的并发任务中，它们的值互不干扰。
# default 参数定义了没有设置时的默认值。
_trace_id_ctx = ContextVar("trace_id", default="no-trace-id")
_user_name_ctx = ContextVar("user_name", default="system")
_user_type_ctx = ContextVar("user_type", default="service")
_api_path_ctx = ContextVar("api_path", default="/")
_env_ctx = ContextVar("env", default="prod")

# 2. 定义 Getters (供 Logger 调用)
def get_current_trace_id() -> str:
    return _trace_id_ctx.get()

def get_current_user_name() -> str:
    return _user_name_ctx.get()

def get_current_user_type() -> str:
    return _user_type_ctx.get()

def get_current_api_path() -> str:
    return _api_path_ctx.get()

def get_current_env() -> str:
    return _env_ctx.get()

# 3. 定义 Setter/Configuration 工具 (供业务代码调用)
class LogContext:
    """
    上下文管理器，用于在一段代码执行期间设置日志上下文。
    支持 with 语句，退出时自动恢复旧值（防止污染）。
    """
    def __init__(
        self, 
        trace_id: Optional[str] = None,
        user_name: Optional[str] = None,
        user_type: Optional[str] = None,
        api_path: Optional[str] = None,
        env: Optional[str] = None
    ):
        self.trace_id = trace_id or str(uuid.uuid4()) # 如果没传 trace_id，自动生成一个
        self.user_name = user_name
        self.user_type = user_type
        self.api_path = api_path
        self.env = env
        
        self._tokens = {}

    def __enter__(self):
        # 设置新值，并保留 token 用于恢复
        if self.trace_id:
            self._tokens['trace_id'] = _trace_id_ctx.set(self.trace_id)
        if self.user_name:
            self._tokens['user_name'] = _user_name_ctx.set(self.user_name)
        if self.user_type:
            self._tokens['user_type'] = _user_type_ctx.set(self.user_type)
        if self.api_path:
            self._tokens['api_path'] = _api_path_ctx.set(self.api_path)
        if self.env:
            self._tokens['env'] = _env_ctx.set(self.env)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 恢复旧值
        if 'trace_id' in self._tokens:
            _trace_id_ctx.reset(self._tokens['trace_id'])
        if 'user_name' in self._tokens:
            _user_name_ctx.reset(self._tokens['user_name'])
        if 'user_type' in self._tokens:
            _user_type_ctx.reset(self._tokens['user_type'])
        if 'api_path' in self._tokens:
            _api_path_ctx.reset(self._tokens['api_path'])
        if 'env' in self._tokens:
            _env_ctx.reset(self._tokens['env'])

# 快捷设置函数 (如果你不想用 with)
def set_log_context(
    trace_id: str = None, 
    user_name: str = None, 
    api_path: str = None
):
    """直接设置当前上下文 (通常用于脚本或全局初始化)"""
    if trace_id:  _trace_id_ctx.set(trace_id)
    if user_name:  _user_name_ctx.set(user_name)
    if api_path:  _api_path_ctx.set(api_path)
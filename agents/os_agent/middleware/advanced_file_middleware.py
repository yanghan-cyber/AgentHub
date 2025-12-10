from datetime import datetime
import os
import asyncio
from typing import Callable, Awaitable, cast
from markitdown import MarkItDown
from openai import OpenAI
from async_lru import alru_cache
from langchain_core.tools import tool
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import (
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import SystemMessage
from deepagents.backends.protocol import BackendProtocol
from deepagents.backends import FilesystemBackend

from utils.logger import get_logger

logger = get_logger(__name__)

# --- Prompts ---

ADVANCED_READ_TOOL_DESCRIPTION = """
**[READ-ONLY / NON-CODE ONLY]**
Use this tool to convert BINARY or COMPLEX formats into readable Markdown for analysis.

## ✅ TARGET SCENARIOS (Use for):
1. **Office Docs**: PDF (`.pdf`), Word (`.docx`), PowerPoint (`.pptx`)
2. **Data Sheets**: Excel (`.xlsx`, `.csv`) -> Converts to Markdown tables
3. **Media**: Images (`.png`, `.jpg`) -> Generates AI descriptions/OCR

## ⛔️ STRICTLY PROHIBITED (DO NOT Use for):
- **Source Code**: `.py`, `.js`, `.java`, etc.
- **Config Files**: `.json`, `.yaml`, `.xml`
- **Editing Tasks**: If you plan to use `edit_file` later, YOU MUST USE standard `read_file`.
- **Reason**: This tool returns a *synthetic Markdown representation*. It loses original line numbers and strict syntax.

## Parameters
- file_path: The absolute path to the file.
- offset: Line number to start reading from (0-based index). Default is 0.
- limit: Maximum number of lines to return. Default is 2000.
"""

ADVANCED_FILE_SYSTEM_PROMPT = """## Advanced File Tools

You have access to an `advanced_read_file` tool that can read and understand almost any file format, including:

### `advanced_read_file` Usage  Policy
- When you need understand files, or documents, excels, images, or other binaries formats, use this tool.
- The files read by this tool are automatically converted into Markdown or text format for easy reading. 
- Use this tool in parallel to read multiple files at once. but NEVER exceed 5 files at once.

#### 🚦 Decision Matrix: Which tool should I use?

| Your Goal / Intent | Target File Type | **REQUIRED TOOL** |
| :--- | :--- | :--- |
| **EDIT / FIX / MODIFY** | **ANY** Code/Config (`.py`, `.js`, `.json`, `.md`) | `read_file` |
| **Read Code Context** | Code/Config (`.py`, `.json`, etc.) | `read_file` |
| **Analyze / Understand** | **PDF, DOCX, PPTX, XLSX** | `advanced_read_file` |
| **Vision / Describe** | **Images** (`.png`, `.jpg`, `.jpeg`) | `advanced_read_file` |

"""


class AdvancedFileMiddleware(AgentMiddleware):
    def __init__(self, backend: BackendProtocol):
        super().__init__()
        self.backend = backend
        self.md_converter = None
        self._md_converter_lock = asyncio.Lock()
        self.system_prompt = ADVANCED_FILE_SYSTEM_PROMPT
        self.tools = [self._create_advanced_read_tool()]

    async def _ensure_md_converter(self):
        """延迟初始化 MarkItDown，避免在构造函数中执行阻塞操作"""
        if self.md_converter is not None:
            return

        async with self._md_converter_lock:
            if self.md_converter is not None:
                return

            # 将可能阻塞的初始化移到线程中执行
            self.md_converter = await asyncio.to_thread(self._init_markitdown)

    def _init_markitdown(self):
        """在线程中执行 MarkItDown 的同步初始化"""
        api_key = os.environ.get("MARKITDOWN_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("MARKITDOWN_OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model_name = os.environ.get("MARKITDOWN_MODEL", "gpt-4o")
        llm_prompt = """You are an intelligent vision assistant. Perform the following steps:
## Step 1: Classify Determine if the image is TYPE A (Text-Heavy) or TYPE B (Visual-Heavy).
Type A: Screenshots, PDFs, receipts, pages of text, code snippets.
Type B: Photos of people, landscapes, animals, objects, or complex charts without dense text.

## Step 2: Execute
If TYPE A: Output the full text content verbatim. Maintain markdown formatting for headers or lists.
If TYPE B: Provide a descriptive summary of what is shown in the image.

!Important: Only Output Image content, Never output without any additional explanations or any irrelevant text.
!Important: If the image contains both a scene and significant text (e.g., a street sign in a landscape), describe the scene first, then quote the text found within it."""
        if api_key:
            try:
                logger.info(f"[MarkItDown] Initializing with LLM support (Model: {model_name})...")
                client = OpenAI(
                    api_key=api_key,
                    base_url=base_url
                )

                return MarkItDown(
                    llm_client=client,
                    llm_model=model_name,
                    llm_prompt=llm_prompt
                )
            except Exception as e:
                logger.warning(f"[MarkItDown] LLM Init Error: {e}. Falling back to basic mode.")
                return MarkItDown()
        else:
            logger.info("[MarkItDown] Running in basic mode (No Image/OCR support).")
            return MarkItDown()

    def _create_advanced_read_tool(self):
        """Create the advanced_read_file tool"""

        # 定义内部工作函数（可能包含阻塞调用）
        def _process_file_sync(file_path: str, offset: int, limit: int) -> str:
            """同步处理文件（可能包含阻塞操作）"""
            # 1. Backend Check & Path Resolution
            if isinstance(self.backend, FilesystemBackend):
                try:
                    # Reuse FilesystemBackend's secure path resolution
                    resolved_path = self.backend._resolve_path(file_path)

                    if not resolved_path.exists():
                        return f"<error>File '{file_path}' not found.</error>"

                    if not resolved_path.is_file():
                        return f"<error>Path '{file_path}' is not a file.</error>"

                    abs_path_str = str(resolved_path)

                except ValueError as e:
                    return f"<error>Invalid path or security violation: {str(e)}</error>"
            else:
                # Reject unsupported backends
                return f"<error>This tool requires a FilesystemBackend. Current backend: {type(self.backend).__name__}</error>"

            return abs_path_str

        @tool(description=ADVANCED_READ_TOOL_DESCRIPTION)
        async def advanced_read_file(file_path: str, offset: int = 0, limit: int = 2000) -> str:
            """
            Reads ANY file format (PDF, Excel, Word, PPT, Images, Code) and converts it to Markdown text.
            """

            # 使用 asyncio.to_thread 执行可能阻塞的文件系统操作
            try:
                abs_path_str = await asyncio.to_thread(_process_file_sync, file_path, offset, limit)
                if abs_path_str.startswith("<error>"):
                    return abs_path_str
            except Exception as e:
                return f"<error>File processing failed: {str(e)}</error>"

            # 2. Core Conversion Logic (Cached & Async)
            try:
                # 确保 md_converter 已初始化
                await self._ensure_md_converter()
                full_content = await self._convert_file_cached(abs_path_str)
            except Exception as e:
                return f"<error>Failed to convert file content: {str(e)}</error>"

            # 3. Empty Content Check
            if not full_content or not full_content.strip():
                return "<system-reminder>File exists but content is empty (or parsing returned no text).</system-reminder>"

            # 4. Pagination Logic
            full_len = len(full_content)
            if offset >= full_len:
                return f"<system-reminder>End of content. Total length: {full_len}.</system-reminder>"

            end_index = min(offset + limit, full_len)

            result_text = full_content[offset:end_index]

            # 5. Truncation Warning
            footer = ""
            if end_index < full_len:
                footer = f"\n\n<system-reminder>Truncated! Call tool again with start_index={end_index} to continue reading.</system-reminder>"
            else:
                footer = "\n\n<system-reminder>End of content.</system-reminder>"
            return f"### File Content: {file_path}\n{result_text}{footer}"

        return advanced_read_file
    # --- Internal Methods ---

    @alru_cache(maxsize=32, ttl=600)
    async def _convert_file_cached(self, abs_file_path: str) -> str:
        """
        Internal method: Run time-consuming MarkItDown conversion in thread pool and cache result.
        """
        # 确保 md_converter 已初始化
        await self._ensure_md_converter()

        loop = asyncio.get_running_loop()
        try:
            # MarkItDown.convert is synchronous/blocking, so run it in executor
            result = await loop.run_in_executor(
                None,
                self.md_converter.convert,
                abs_file_path
            )
            return result.text_content
        except Exception as e:
            raise RuntimeError(f"MarkItDown processing failed: {e}")

    # --- Lifecycle Hooks ---

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Inject System Prompt."""
        if request.system_message is not None:
            new_system_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{self.system_prompt}"},
            ]
        else:
            new_system_content = [{"type": "text", "text": self.system_prompt}]
            
        new_system_message = SystemMessage(
            content=cast("list[str | dict[str, str]]", new_system_content)
        )
        return handler(request.override(system_message=new_system_message))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Async Inject System Prompt."""
        if request.system_message is not None:
            new_system_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{self.system_prompt}"},
            ]
        else:
            new_system_content = [{"type": "text", "text": self.system_prompt}]
            
        new_system_message = SystemMessage(
            content=cast("list[str | dict[str, str]]", new_system_content)
        )
        return await handler(request.override(system_message=new_system_message))
    
    async def abefore_agent(self, state, runtime):
        content = ""
        content += f"""<env>Current Datetime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</env>\n"""

        if 'todos' not in state or all([todo['status'] == 'completed' for todo in state['todos']]):
            content += """<system-reminder>This is a reminder that your todo list is currently empty. DO NOT mention this to the user explicitly because they are already aware. If you are working on tasks that would benefit from a todo list please use the TodoWrite tool to create one. If not, please feel free to ignore. Again do not mention this message to the user.</system-reminder>\n"""

        return {
            "messages": [
                {"role": "user", "content": content}
            ]
        }

if __name__ == "__main__":
    backend = FilesystemBackend(
        "D:/ai_lab/langgraph-agents",
        virtual_mode=False,
        max_file_size_mb=200,
    )

    async def main():
        backend = FilesystemBackend(
            "D:/ai_lab/langgraph-agents",
            virtual_mode=True,
            max_file_size_mb=200,
        )
        mid = AdvancedFileMiddleware(backend=backend)

        res = await mid.tools[0].ainvoke(
            {
                "file_path": "D:/ai_lab/langgraph-agents/agent-store-space/.gitkeep"
            }
        )

        return res

    result = asyncio.run(main())
    print(result)
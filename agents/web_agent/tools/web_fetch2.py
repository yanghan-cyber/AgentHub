import os
import socket
import ipaddress
import asyncio
import aiohttp
import aiofiles
import tempfile
import mimetypes
from urllib.parse import urlparse
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from async_lru import alru_cache

# --- Crawl4AI 组件 ---
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

# --- MarkItDown 组件 ---
from markitdown import MarkItDown

from utils.logger import get_logger

logger = get_logger(__name__)

# =================配置区=================

# 1. 浏览器伪装头 (解决 403 Forbidden 关键)
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1"
}

# 2. 支持的文档类型映射 (解决 URL 无后缀问题)
# 当 URL 类似于 download.jsp?id=123 时，我们依靠 Content-Type 来决定保存为什么后缀
MIME_TO_EXT = {
    'application/pdf': '.pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
    'application/msword': '.doc',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
    'application/vnd.ms-excel': '.xls',
    'application/vnd.openxmlformats-officedocument.presentationml.presentation': '.pptx',
    'application/vnd.ms-powerpoint': '.ppt',
    'text/csv': '.csv',
    'application/rtf': '.rtf',
    'text/plain': '.txt'
}

# =================核心逻辑=================

def is_safe_url(url: str) -> bool:
    """防止 SSRF 攻击，禁止访问内网 IP"""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if not hostname: return False
        try:
            ip_str = socket.gethostbyname(hostname)
            ip = ipaddress.ip_address(ip_str)
            if ip.is_private or ip.is_loopback: return False
        except: return False
        return True
    except: return False

async def _process_with_markitdown(url: str, content_type: str = None) -> str:
    """
    文档处理通道：下载 -> 保存临时文件(带正确后缀) -> MarkItDown 转换
    """
    logger.info(f"[MarkItDown] Downloading doc: {url}")

    temp_path = None
    try:
        async with aiohttp.ClientSession(headers=BROWSER_HEADERS) as session:
            async with session.get(url) as response:
                if response.status != 200:
                    return f"<error>Download failed: HTTP {response.status}</error>"

                # --- 智能确定文件后缀 ---
                # 优先使用显式传入的类型，否则从响应头读取
                if not content_type:
                    content_type = response.headers.get('Content-Type', '').split(';')[0].strip().lower()

                # 1. 查表
                ext = MIME_TO_EXT.get(content_type)
                # 2. 猜测
                if not ext: ext = mimetypes.guess_extension(content_type)
                # 3. 从 URL 截取 (兜底)
                if not ext:
                    filename = url.split("/")[-1].split("?")[0]
                    if "." in filename: ext = "." + filename.split(".")[-1]
                # 4. 默认
                if not ext: ext = ".tmp"

                logger.info(f"[MarkItDown] Type: {content_type} | Ext: {ext}")

                # 创建临时文件 - 使用异步方式避免阻塞
                loop = asyncio.get_running_loop()
                temp_file = await loop.run_in_executor(
                    None,
                    lambda: tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                )
                temp_path = temp_file.name
                temp_file.close()  # 关闭文件句柄，准备异步写入

                # 使用 aiofiles 进行真正的异步文件写入
                async with aiofiles.open(temp_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(1024):
                        await f.write(chunk)
        
        # --- 调用 MarkItDown (同步代码放入线程池) ---
        def run_sync_convert(path):
            md = MarkItDown()
            # 如果你有 OpenAI Key 并想解析图片，可以这里初始化:
            # md = MarkItDown(llm_client=..., llm_model="gpt-4o") 
            result = md.convert(path)
            return result.text_content

        loop = asyncio.get_running_loop()
        markdown_text = await loop.run_in_executor(None, run_sync_convert, temp_path)

        return f"# Document Content (Source: {content_type})\n\n{markdown_text}"

    except Exception as e:
        return f"<error>MarkItDown failed: {str(e)}</error>"
    finally:
        # 异步清理临时文件
        if temp_path:
            async def cleanup_file():
                try:
                    if await asyncio.to_thread(os.path.exists, temp_path):
                        await asyncio.to_thread(os.remove, temp_path)
                except:
                    pass
            await cleanup_file()

@alru_cache(maxsize=32, ttl=600)
async def _crawl_url(url: str) -> str:
    """
    智能路由核心：
    1. 发送 HEAD 请求探测类型
    2. 如果是文档 -> MarkItDown
    3. 如果是网页 -> Crawl4AI
    """
    logger.info(f"[SmartFetch] Analyzing: {url}")

    # --- A. 探测阶段 ---
    is_document = False
    detected_type = ""

    try:
        async with aiohttp.ClientSession(headers=BROWSER_HEADERS) as session:
            # allow_redirects=True 必须开启，防止短链接误判
            async with session.head(url, allow_redirects=True, timeout=5) as resp:
                # 如果 HEAD 被屏蔽 (403/405)，我们假设它是普通网页，交给 Crawl4AI 强行处理
                if resp.status in [403, 404, 405]:
                    logger.info(f"[SmartFetch] Head request denied ({resp.status}), fallback to crawler.")
                else:
                    ctype = resp.headers.get('Content-Type', '').lower().split(';')[0].strip()
                    detected_type = ctype
                    
                    # 检查是否命中我们的文档列表
                    if ctype in MIME_TO_EXT:
                        is_document = True
                    # 额外检查：URL 是否以常见文档后缀结尾 (双重保险)
                    elif any(url.lower().endswith(x) for x in ['.pdf', '.docx', '.xlsx', '.pptx']):
                        is_document = True

    except Exception as e:
        logger.info(f"[SmartFetch] Detection error: {e}, proceeding as web page.")

    # --- B. 分发阶段 ---
    if is_document:
        return await _process_with_markitdown(url, detected_type)
    else:
        # === Crawl4AI 网页抓取配置 ===
        logger.info(f"[Crawl4AI] Starting browser for: {url}")
        
        browser_config = BrowserConfig(
            headless=True,
            # 这里的 headers 主要是给 Playwright 的
            headers=BROWSER_HEADERS, 
            verbose=False
        )

        run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            word_count_threshold=5,
            # 移除常见的干扰元素
            excluded_selector="nav, footer, header, aside, .ads, .cookie-banner",
            exclude_external_links=True,
            markdown_generator=DefaultMarkdownGenerator(
                options={"ignore_links": True},
                content_filter=PruningContentFilter(threshold=0.45, min_word_threshold=5)
            ),
            remove_overlay_elements=True,
            process_iframes=True,
            # 等待页面完全加载
            js_code="window.scrollTo(0, document.body.scrollHeight);",
            delay_before_return_html=2.0, 
        )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=run_config)
            
            if not result.success:
                # 错误处理
                err = result.error_message
                return f"<error>Crawl4AI failed: {err}</error>"
            
            return result.markdown

# --- 分页辅助函数 ---
def process_content(markdown_text: str, start_index: int = 0, max_length: int = 6000) -> str:
    if not markdown_text: return "No content found."
    full_len = len(markdown_text)
    if start_index >= full_len:
        return f"<system-reminder>End of content. Total length: {full_len}</system-reminder>"
    
    end_index = min(start_index + max_length, full_len)
    chunk = markdown_text[start_index:end_index]
    
    if end_index < full_len:
        chunk += f"\n\n<system-reminder>Truncated! Call tool again with start_index={end_index} to continue reading.</system-reminder>"
    return chunk

# =================Tool 定义=================

class FetchInput(BaseModel):
    url: str = Field(description="Target URL (webpage or document).")
    start_index: int = Field(default=0, description="Start position for reading.")
    max_length: int = Field(default=6000, description="Max characters to return.")

@tool(args_schema=FetchInput)
async def web_fetch(url: str, start_index: int = 0, max_length: int = 6000) -> str:
    """
    - Fetches the URL content, converts HTML to clean markdown
    - Use this tool when you need to retrieve and analyze web content

    Usage notes:
    - The URL must be a fully-formed valid URL
    - This tool is read-only and does not modify any files
    - Result may be truncated if it exceeds the max_length.
    - Supports pagination for long content, You can use the start_index parameter to fetch the next page.
    - Includes a self-cleaning 15-minute cache for faster responses whenrepeatedly accessing the same URL
    """
    if not is_safe_url(url):
        return "<error>Security Block: Private IP access denied.</error>"

    try:
        content = await _crawl_url(url)
        return process_content(content, start_index, max_length)
    except Exception as e:
        return f"<error>Fetch failed: {str(e)}</error>"

# =================测试运行=================
if __name__ == "__main__":
    async def main():
        # 1. 测试你的那个困难 URL (无后缀 + 可能 403)
        logger.info("--- Test 1: Complex Doc URL ---")
        complex_url = "https://gsy.hunnu.edu.cn/system/_content/download.jsp?urltype=news.DownloadAttachUrl&owner=1564207687&wbfileid=5407913"
        res = await web_fetch.ainvoke({"url": complex_url})
        logger.info(f"Result Preview:\n{res[:300]}...")

        # 2. 测试普通动态网页
        logger.info("--- Test 2: Dynamic Web Page ---")
        web_url = "https://github.com/trending"
        res2 = await web_fetch.ainvoke({"url": web_url})
        logger.info(f"Result Preview:\n{res2[:300]}...")

    asyncio.run(main())
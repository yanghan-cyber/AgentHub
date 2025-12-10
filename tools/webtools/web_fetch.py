import os
import socket
import ipaddress
import asyncio
from urllib.parse import urlparse
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from async_lru import alru_cache  # pip install async_lru

# 引入 crawl4ai 核心组件
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

from utils.logger import get_logger

logger = get_logger(__name__)
# --- 配置 ---
# 代理配置 (如果你需要通过代理访问)
# 注意：Crawl4AI (Playwright) 的代理格式通常是 "server": "http://127.0.0.1:7890"


# --- 1. 安全检查 (保持不变) ---
def is_safe_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            return False
        try:
            ip_str = socket.gethostbyname(hostname)
            ip = ipaddress.ip_address(ip_str)
            if ip.is_private or ip.is_loopback:
                return False
        except:
            return False
        return True
    except:
        return False


# --- 2. 核心抓取逻辑 (带缓存) ---
# 使用 async_lru 进行内存缓存，TTL=600秒 (10分钟)
# 这样翻页时不需要重新启动浏览器爬取
@alru_cache(maxsize=32, ttl=600)
async def _crawl_url(url: str) -> str:
    """
    使用 Crawl4AI 启动无头浏览器抓取网页，并返回 Markdown。
    """
    logger.info(f"[Crawl4AI] Starting crawl for: {url}")

    # A. 浏览器配置
    # 如果有代理，需要在这里注入
    browser_args = []
    PROXY_SERVER = os.getenv("WEB_FETCH_PROXY_SERVER")

    if PROXY_SERVER:
        browser_args.append(f"--proxy-server={PROXY_SERVER}")

    browser_config = BrowserConfig(
        headless=True,  # 无头模式，不弹出浏览器窗口
        verbose=False,  # 减少日志干扰
        extra_args=browser_args,  # 注入代理参数
    )
# --- 2. 抓取运行配置 (CrawlerRunConfig) ---
    # 这一层控制具体的页面处理逻辑
    

    # 定义内容过滤策略：基于算法修剪低信息密度的节点
    pruning_filter = PruningContentFilter(
        threshold=0.48,           # 阈值：越低越严格，过滤越多噪音
        threshold_type="fixed",
        min_word_threshold=5      # 忽略少于5个单词的块
    )
    
    #  定义 Markdown 生成策略：忽略链接以减少 token (可选)

    md_generator = DefaultMarkdownGenerator(
        options={"ignore_links": True} ,
        content_filter=pruning_filter,  # <--- 移到这里
    )

    # B. 运行配置
    run_config = CrawlerRunConfig(
        # CacheMode.BYPASS: 我们自己外层做了 async_lru，这里建议让 crawl4ai 每次都拿最新的
        # 或者使用 CacheMode.ENABLED 让 crawl4ai 管理文件缓存
        cache_mode=CacheMode.BYPASS,
        # --- 内容清洗核心 ---
        word_count_threshold=10,  # 忽略太短的文本块
        excluded_tags=['nav', 'footer', 'header', 'aside', 'script', 'style'], # 移除结构噪音
        excluded_selector="[class*='nav'],[class*='search'],[class*='header'],[class*='bar'],[class*='sidebar'],[class*='menu'],.ad,.ads,.advertisement,.cookie-banner,#ads,#popup,nav,footer,aside",
        exclude_domains=["ads.com", "googleads.com", "doubleclick.net"],
        exclude_external_links=True,
        markdown_generator=md_generator,
        exclude_all_images=True,
        # exclude_internal_links=True,
        # --- 移除覆盖层 (弹窗/Cookie同意框) ---
        remove_overlay_elements=True,
        process_iframes=True,
        js_code="window.scrollTo(0, document.body.scrollHeight);",
        delay_before_return_html=2.0,  # 滚动后等待 2 秒让内容渲染
        
        # 爬取策略
    )

    # C. 执行抓取
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=run_config)

        if not result.success:
            error_msg = result.error_message or "Unknown error"
            raise Exception(f"Crawl failed: {error_msg}")

        # crawl4ai 能够自动生成非常高质量的 Markdown
        return result.markdown


# --- 3. 内容处理与分页 ---
# 这部分是纯 CPU 逻辑，处理字符串
def process_content(
    markdown_text: str, start_index: int = 0, max_length: int = 6000
) -> str:
    if not markdown_text:
        return "No content found."

    full_len = len(markdown_text)

    # 如果起始位置超过了全文长度
    if start_index >= full_len:
        return f"<system-reminder>End of content. Total length: {full_len}</system-reminder>"

    end_index = min(start_index + max_length, full_len)
    chunk = markdown_text[start_index:end_index]

    # 添加截断提示
    if end_index < full_len:
        chunk += f"\n\n<system-reminder>Truncated! Call tool again with start_index={end_index} to continue reading.</system-reminder>"

    return chunk


# --- 4. Tool 定义 ---
class FetchInput(BaseModel):
    url: str = Field(description="Target URL to fetch.")
    start_index: int = Field(
        default=0, description="Offset for pagination (start position)."
    )
    max_length: int = Field(
        default=6000, description="Max characters to return per call."
    )


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
        # 1. 抓取 (带 async_lru 缓存)
        # 注意：因为 crawl4ai 启动浏览器较慢，缓存这里能极大提升"翻页"体验
        markdown_content = await _crawl_url(url)

        # 2. 分页处理
        return process_content(markdown_content, start_index, max_length)

    except Exception as e:
        return f"<error>Fetch failed: {str(e)}</error>"


# --- 测试代码 ---
if __name__ == "__main__":
    import time

    async def main():
        # 测试 URL: GitHub Trending (这是典型的动态网页，旧方法抓不到内容)
        test_url = "https://memos-docs.openmem.net/cn/dashboard/api/overview"

        logger.info(f"--- 1. Testing Crawl (Fresh) ---")
        t0 = time.time()
        # 第一次调用，会启动浏览器抓取
        res1 = await web_fetch.ainvoke({"url": test_url, "max_length": 50000})
        logger.info(f"Time: {time.time() - t0:.2f}s")
        logger.info(f"Preview:\n{res1}")  # 打印前200字符

        logger.info(f"--- 2. Testing Pagination (Cache Hit) ---")
        t1 = time.time()
        # 第二次调用，只是翻页，应该瞬间完成 (0.00s)
        res2 = await web_fetch.ainvoke(
            {"url": test_url, "start_index": 500, "max_length": 500}
        )
        logger.info(f"Time: {time.time() - t1:.4f}s")

        # 验证缓存是否生效
        logger.info(f"Cache Info: {_crawl_url.cache_info()}")

    asyncio.run(main())

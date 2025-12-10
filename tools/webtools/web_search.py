import os
from typing import Literal, Optional
from langchain.tools import tool
from pydantic import BaseModel, Field
from tavily import TavilyClient

from utils.logger import get_logger

logger = get_logger(__name__)


class SearchInput(BaseModel):
    query: str = Field(description="搜索关键词")
    
    topic: Literal["general", "news"] = Field(
        default="general", 
        description="搜索类型。'general' 适合普通查询，'news' 适合查找新闻。"
    )
    
    days: Optional[int] = Field(
        default=None, 
        description="[时间过滤器] 限制搜索最近 X 天的内容。例如：今天=1，本周=7，本月=30。仅在用户明确询问'最新'、'最近'或特定时间段的信息时使用。默认为 None (不限制时间)。"
    )
    
    search_depth: str = Field(
        default="basic", 
        description="搜索深度。'basic' (快速) 或 'advanced' (高质量)。"
    )
    
    max_results: int = Field(
        default=5, 
        description="返回结果的数量。默认返回5 条结果。"
    )
@tool(args_schema=SearchInput)
def web_search(query: str, topic: str = "general", days: Optional[int] = None, search_depth: str = "basic", max_results: int = 5) -> str:
    """
    使用 Tavily 搜索引擎查找互联网信息。
    返回结果包含：标题、URL、以及页面内容的简短摘要。
    """
    # --- 配置 ---

    # 建议放入环境变量: export TAVILY_API_KEY="tvly-..."
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

    # 初始化客户端
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None
    if not tavily_client:
        return "<error>Tavily API key is missing. Please set TAVILY_API_KEY env var.</error>"

    try:
        logger.info(f"[Tavily] Searching: {query} (Depth: {search_depth})")

        # 执行搜索
        response = tavily_client.search(
            query=query,
            days=days,
            topic=topic,
            search_depth=search_depth, # "basic" or "advanced"
            max_results=max_results,
            include_answer=True,       # 让 Tavily 尝试直接生成一个简短回答
            include_domains=None,      # 可以限制只搜 github.com 等
            include_raw_content=False  # 我们有 fetch 工具，所以这里不需要 raw_html
        )
        
        # --- 格式化输出给 LLM ---
        # 1. 这种格式非常节省 Token，且清晰易读
        output = []
        
        # A. 如果 Tavily 生成了直接回答，放在最前面
        if response.get("answer"):
            output.append(f"### Direct Answer (AI Generated)\n{response['answer']}\n")
            
        # B. 罗列搜索结果
        output.append(f"### Search Results ({len(response.get('results', []))})")
        
        for i, result in enumerate(response.get("results", [])):
            title = result.get("title", "No Title")
            url = result.get("url", "#")
            content = result.get("content", "")
            
            # 使用 Markdown 列表格式
            item = (
                f"{i+1}. **[{title}]({url})**\n"
                f"   *Snippet*: {content[:500]}..." # 截断摘要，防止太长
            )
            output.append(item)
            
        return "\n".join(output)

    except Exception as e:
        return f"<error>Tavily search failed: {str(e)}</error>"

# --- 测试 ---
if __name__ == "__main__":
    # 确保你有环境变量，或者在这里临时写死 key 测试
    # os.environ["TAVILY_API_KEY"] = "tvly-你的key"
    
    # 重新初始化一下以便测试读到 key
    import dotenv
    dotenv.load_dotenv()
    if os.getenv("TAVILY_API_KEY"):
         tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    result = web_search.invoke({"query": "Python requests vs httpx difference", "search_depth": "basic"})
    print(result)
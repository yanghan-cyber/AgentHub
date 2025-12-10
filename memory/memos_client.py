import os
import json
import asyncio
from typing import List, Dict, Any, Optional
import aiohttp

from utils.logger import get_logger

logger = get_logger(__name__)

class MemosClient:
    def __init__(self, api_key: str = None, base_url: str = None, session: aiohttp.ClientSession = None):
        """
        初始化 Memos 客户端
        :param api_key: API Key (默认从环境变量 MEMOS_API_KEY 读取)
        :param base_url: API 基础地址
        :param session: aiohttp.ClientSession 实例(可选,如果不提供则每次调用创建新会话)
        """

        self.api_key = api_key or os.environ.get("MEMOS_API_KEY")
        if not self.api_key:
            raise ValueError("API Key is required. Set MEMOS_API_KEY env var or pass it explicitly.")
        self.base_url = base_url or os.environ.get("MEMOS_BASE_URL") or "https://memos.memtensor.cn/api/openmem/v1"
        self.base_url = self.base_url.rstrip('/')
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {self.api_key}"
        }
        self._session = session

    async def __aenter__(self):
        """异步上下文管理器入口"""
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """内部通用异步 POST 请求处理"""
        url = f"{self.base_url}/{endpoint}"

        # 如果没有 session 或 session 已关闭,创建新 session
        session = self._session
        if session is None or session.closed:
            session = aiohttp.ClientSession()
            created_new_session = True
        else:
            created_new_session = False

        try:
            async with session.post(url=url, headers=self.headers, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                return result
        except aiohttp.ClientError as e:
            return {"code": -1, "message": f"Request failed: {str(e)}"}
        finally:
            # 如果创建了新的 session,需要关闭它
            if created_new_session and not session.closed:
                await session.close()

    async def add_messages(self, user_id: str, conversation_id: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        1. 上报对话消息 (Add Message)
        :param messages: 格式 [{"role": "user", "content": "..."}, ...]
        """
        data = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "messages": messages
        }
        return await self._post("add/message", data)

    async def get_history(self, user_id: str, conversation_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """
        2. 获取历史消息 (Get Message)
        :return: 返回大模型可用的简洁列表 [{"role": "...", "content": "..."}]
        """
        data = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "message_limit_number": limit
        }

        result = await self._post("get/message", data)

        if result.get("code") == 0:
            raw_list = result.get("data", {}).get("message_detail_list", [])
            # 格式化清洗：只保留 role 和 content
            return [{"role": m["role"], "content": m["content"]} for m in raw_list]
        else:
            logger.warning(f"Error getting history: {result.get('message')}")
            return []

    async def search_memory(
            self,
            user_id: str,
            query: str,
            conversation_id: Optional[str] = None,
            memory_limit: int = 6,
            include_preference: bool = True,
            preference_limit: int = 6
        ) -> Dict[str, Any]:
            """
            搜索用户记忆 (支持偏好召回和数量控制)

            :param user_id: 用户唯一标识
            :param query: 查询语句
            :param conversation_id: 会话ID (传入则提升本会话记忆权重；不传则进行全局检索)
            :param memory_limit: 事实记忆返回条数 (默认6，最大25)
            :param include_preference: 是否启用偏好记忆召回 (默认True)
            :param preference_limit: 偏好记忆返回条数 (默认6，最大25)
            """
            data = {
                "user_id": user_id,
                "query": query,
                "memory_limit_number": memory_limit,
                "include_preference": include_preference,
                "preference_limit_number": preference_limit
            }

            # 只有在有 conversation_id 时才传，否则视为全局检索
            if conversation_id:
                data["conversation_id"] = conversation_id

            result = await self._post("search/memory", data)

            if result.get("code") == 0:
                return result.get("data", {})
            else:
                logger.warning(f"[MemosClient] Search Error: {result.get('message')}")
                return {}

# ================= 使用示例 =================

async def main():
    # 1. 初始化
    # os.environ["MEMOS_API_KEY"] = "YOUR_REAL_KEY" # 确保环境变量已设置
    async with MemosClient() as client: # 自动读取环境变量
        user_id = "memos_user_demo"
        conv_id = "conv_001"

        logger.info("--- 1. 上报消息 ---")
        new_msgs = [
            {"role": "user", "content": "我下周要去成都出差，有什么推荐的火锅吗？"},
            {"role": "assistant", "content": "成都有很多不错的火锅，比如蜀大侠、小龙坎。"}
        ]
        res_add = await client.add_messages(user_id, conv_id, new_msgs)
        logger.info(f"Add Result: {res_add}")

        logger.info("\n--- 2. 获取历史 (清洗版) ---")
        history = await client.get_history(user_id, conv_id, limit=5)
        logger.info(json.dumps(history, ensure_ascii=False, indent=2))

        logger.info("\n--- 3. 搜索记忆 ---")
        # 假设用户之前说过不喜欢吃太辣的
        query = "还是给我推荐点不辣的吧，我肠胃不好。"
        memory_context = await client.search_memory(user_id, query, conv_id)

        # 打印检索到的偏好 (如果有)
        preferences = memory_context.get("preference_detail_list", [])
        if preferences:
            logger.info("发现用户偏好:")
            for pref in preferences:
                logger.info(f"- {pref.get('preference')}")
        else:
            logger.info("未发现相关偏好记忆。")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(main())
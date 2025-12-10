import os
from langchain.agents import create_agent

from agents.os_agent.prompt import OS_AGENT_SYSTEM_PROMPT

from .middleware import AdvancedFileMiddleware
from langchain.agents.middleware import TodoListMiddleware, ShellToolMiddleware, HostExecutionPolicy
from deepagents.middleware import FilesystemMiddleware
from langchain_openai import ChatOpenAI
from deepagents.backends import FilesystemBackend
from dotenv import load_dotenv
from deepagents import CompiledSubAgent
from utils.logger import get_logger

load_dotenv(override=True)

logger = get_logger(__name__)

backend = FilesystemBackend(
    "D:/ai_lab/langgraph-agents/agent-store-space", virtual_mode=True
)

os_agent = create_agent(
    model=ChatOpenAI(model=os.getenv("OPENAI_MODEL")),
    system_prompt=OS_AGENT_SYSTEM_PROMPT,
    middleware=[
        TodoListMiddleware(),
        AdvancedFileMiddleware(backend=backend),
        FilesystemMiddleware(backend=backend),
    ],
)



if __name__ == "__main__":
    import asyncio
    context =  {"thread_id": "user_123", "user_id": "user_default"}
    config1 = {"configurable": context}
    async def main(user_input):
        async for mode, chunk in os_agent.astream({"messages": [{"role": "user", "content": user_input}]},config=config1, context=context,stream_mode=["values"]):
            if 'messages' in chunk:
                message = chunk['messages'][-1]
                logger.info(f"Agent response:\n{message}")
    asyncio.run(main("读取一下文件夹里的所有文件看看，并获取里面的内容，总结一下里面的内容给我, 并且把内容增加到temp文件中去。"))


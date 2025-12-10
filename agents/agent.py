from dataclasses import dataclass

from deepagents import CompiledSubAgent
from deepagents.backends import FilesystemBackend

from deepagents.middleware import (
    FilesystemMiddleware,
    SubAgentMiddleware,
)
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from agents.main_agent.middleware import MainAgentMiddleware
from agents.main_agent.prompt import MAIN_AGENT_SYSTEM_PROMPT
from agents.os_agent.middleware.advanced_file_middleware import AdvancedFileMiddleware
from agents.os_agent.prompt import OS_AGENT_SYSTEM_PROMPT
from agents.web_agent.middleware.base import WebAgentMiddleware
from agents.web_agent.prompt import RESEARCHER_SYSTEM_PROMPT
from agents.web_agent.tools import web_fetch, web_search
from memory.middleware import MemOSMiddleware
from utils.logger import get_logger

load_dotenv(override=True)

logger = get_logger(__name__)


@dataclass
class Context:
    thread_id: str 
    user_id: str


default_model = ChatOpenAI(model="glm-4.6")

backend = FilesystemBackend(
    "D:/ai_lab/langgraph-agents/agent-store-space", virtual_mode=True
)

os_agent = create_agent(
    model=default_model,
    system_prompt=OS_AGENT_SYSTEM_PROMPT,
    middleware=[
        AdvancedFileMiddleware(backend=backend),
        FilesystemMiddleware(backend=backend),
    ],
)


web_agent = create_agent(
    model=default_model,
    tools=[web_fetch, web_search],
    system_prompt=RESEARCHER_SYSTEM_PROMPT,
    middleware=[WebAgentMiddleware()],
)


agent = create_agent(
    model=default_model,
    system_prompt=MAIN_AGENT_SYSTEM_PROMPT,
    middleware=[
        MemOSMiddleware(),
        MainAgentMiddleware(),
        SummarizationMiddleware(
            model=default_model,
            max_tokens_before_summary=170000,
            messages_to_keep=6,
        ),
        PatchToolCallsMiddleware(),
        TodoListMiddleware(),
        SubAgentMiddleware(
            default_model=default_model,
            subagents=[
                CompiledSubAgent(
                    name="Web-Searcher",
                    description=(
                        "A specialized research agent for EXTERNAL information retrieval. "
                        "Delegate tasks here when you need to search the internet, verify facts, "
                        "find up-to-date documentation/news, or answer questions requiring knowledge "
                        "outside the local environment."
                    ),
                    runnable=web_agent,
                ),
                CompiledSubAgent(
                    name="File-Agent",
                    description=(
                        "A specialized engineering agent for LOCAL file system. "
                        "Delegate tasks here when you need to explore files (`ls`, `grep`), "
                        "read files(include: Binary files, PDF, Excel, Markdown, Words, PPT, Images etc...), modify code/files."
                    ),
                    runnable=os_agent,
                ),
            ],
            default_middleware=[
                TodoListMiddleware(),
                SummarizationMiddleware(
                    model=default_model,
                    max_tokens_before_summary=170000,
                    messages_to_keep=6,
                ),
                PatchToolCallsMiddleware(),
            ],
        ),
    ],
    # checkpointer=InMemorySaver(),
    # context_schema=Context,
)


if __name__ == "__main__":
    import asyncio

    context = {"thread_id": "user_123", "user_id": "user_default"}
    config1 = {"configurable": context}

    async def main(user_input):
        async for mode, chunk in agent.astream(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config1,
            # context=context,
            stream_mode=["values"],
        ):
            if "messages" in chunk:
                chunk["messages"][-1].pretty_print()
                # message = chunk["messages"][-1]
                # logger.info(f"Agent response:\n{message}")
    asyncio.run(main("今天星期几啦"))

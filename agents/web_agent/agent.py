import os
from langchain.agents import create_agent

from agents.web_agent.middleware import WebAgentMiddleware

from langchain.agents.middleware import TodoListMiddleware
from agents.web_agent.tools import web_fetch, web_search
from langchain_openai import ChatOpenAI
from agents.web_agent.prompt import RESEARCHER_SYSTEM_PROMPT
from utils.logger import get_logger

logger = get_logger(__name__)

web_agent = create_agent(
    model=ChatOpenAI(model=os.getenv("OPENAI_MODEL")),
    tools=[web_fetch, web_search],
    system_prompt=RESEARCHER_SYSTEM_PROMPT,
    middleware=[TodoListMiddleware(), WebAgentMiddleware()],
)



web_agent_config = {
    "name": "Web-Searcher",
    "description": "Specialized agent for web search. ",
    "system_prompt": RESEARCHER_SYSTEM_PROMPT,
    "tools": [web_search, web_fetch],
    "middleware": [WebAgentMiddleware()],
}


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(override=True)
    import asyncio

    async def main():
        async for mode, chunk in web_agent.astream(
            {"messages": "Deepseek v3.2的创新技术有哪些？"}, stream_mode=["values"]
        ):
            if "messages" in chunk:
                message = chunk["messages"][-1]
                logger.info(f"Agent response:\n{message}")


    asyncio.run(main())

import asyncio
import os

from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()

async def main():
    model_client = OpenAIChatCompletionClient(
        model="openai/gpt-oss-20b",
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
            "structured_output": True,
        },
    )

    agent = AssistantAgent(
        name="data_scientist",
        model_client=model_client,
        system_message=(
            "You are a senior data scientist reviewing machine-learning models."
        ),
    )

    result = await agent.run(
        task=(
            "A classifier has precision 0.84, recall 0.71, "
            "F1 0.77, and ROC-AUC 0.86. "
            "Give a short assessment."
        )
    )

    print(result.messages[-1].content)

    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())

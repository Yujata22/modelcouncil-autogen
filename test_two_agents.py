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

    # -------------------------
    # Agent 1: Data Scientist
    # -------------------------
    data_scientist = AssistantAgent(
        name="data_scientist",
        model_client=model_client,
        system_message="""
        You are a Senior Data Scientist on an ML Model Review Council.

        Your responsibility is to:
        - evaluate model performance
        - identify strengths and weaknesses
        - consider precision/recall tradeoffs
        - recommend whether a model deserves further consideration

        Be concise and evidence-based.
        """,
    )

    # -------------------------
    # Agent 2: Model Evaluator
    # -------------------------
    evaluator = AssistantAgent(
        name="model_evaluator",
        model_client=model_client,
        system_message="""
        You are an independent ML Model Evaluation specialist.

        Your job is to critically review another data scientist's
        assessment.

        Look for:
        - unsupported conclusions
        - metric interpretation mistakes
        - missing evaluation metrics
        - class imbalance concerns
        - threshold-related issues
        - validation or generalization risks

        Do not automatically agree with the Data Scientist.
        Challenge conclusions when appropriate.
        """,
    )

    model_metrics = """
    Binary classification model:

    Precision: 0.84
    Recall: 0.71
    F1 Score: 0.77
    ROC-AUC: 0.86
    """

    print("\n========== DATA SCIENTIST ==========\n")

    ds_result = await data_scientist.run(
        task=f"""
        Review the following model:

        {model_metrics}

        Give your recommendation and reasoning.
        """
    )

    ds_assessment = ds_result.messages[-1].content

    print(ds_assessment)

    print("\n========== MODEL EVALUATOR ==========\n")

    evaluator_result = await evaluator.run(
        task=f"""
        Here are the model metrics:

        {model_metrics}

        The Data Scientist gave the following assessment:

        {ds_assessment}

        Critically review this assessment.

        Tell me:
        1. What you agree with
        2. What you disagree with
        3. What evidence is missing
        4. Whether you support the recommendation
        """
    )

    print(evaluator_result.messages[-1].content)

    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())

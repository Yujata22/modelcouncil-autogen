import asyncio
import json
import os

from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


load_dotenv()


async def main():

    # -----------------------------------
    # Load real model evaluation results
    # -----------------------------------
    with open("ml/model_metrics.json", "r") as f:
        model_metrics = json.load(f)

    metrics_text = json.dumps(model_metrics, indent=2)

    # -----------------------------------
    # Groq model client
    # -----------------------------------
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

    # -----------------------------------
    # Agent 1: Data Scientist
    # -----------------------------------
    data_scientist = AssistantAgent(
        name="data_scientist",
        model_client=model_client,
        system_message="""
        You are a Senior Data Scientist on an ML Model Review Council.

        Compare candidate models using the supplied evaluation metrics.

        Your job:
        - compare model performance
        - reason about precision/recall tradeoffs
        - identify the strongest candidate
        - explain why
        - mention any concerns

        Do not simply pick the highest accuracy.
        """,
    )

    # -----------------------------------
    # Agent 2: Independent Evaluator
    # -----------------------------------
    evaluator = AssistantAgent(
        name="model_evaluator",
        model_client=model_client,
        system_message="""
        You are an independent ML evaluation specialist.

        Your job is to challenge the Data Scientist's recommendation.

        Check:
        - whether the metric interpretation is correct
        - whether another model may be preferable
        - whether accuracy is misleading
        - whether recall or precision creates risk
        - what evidence is still missing

        Do not automatically agree.
        """,
    )

    # -----------------------------------
    # Agent 1 review
    # -----------------------------------
    print("\n========== DATA SCIENTIST ==========\n")

    ds_result = await data_scientist.run(
        task=f"""
        We trained three churn classification models.

        Here are the actual test-set metrics:

        {metrics_text}

        Compare Logistic Regression, Random Forest, and XGBoost.

        Recommend ONE model for further review.

        Return:
        1. Recommended model
        2. Why
        3. Main tradeoffs
        4. Remaining concerns
        """
    )

    ds_recommendation = ds_result.messages[-1].content

    print(ds_recommendation)

    # -----------------------------------
    # Agent 2 critique
    # -----------------------------------
    print("\n========== MODEL EVALUATOR ==========\n")

    evaluator_result = await evaluator.run(
        task=f"""
        Here are the actual model metrics:

        {metrics_text}

        The Data Scientist recommended:

        {ds_recommendation}

        Critically evaluate this recommendation.

        Return:
        1. What you agree with
        2. What you disagree with
        3. Missing evidence
        4. Your preferred model
        5. Whether the Data Scientist's recommendation should proceed
        """
    )

    evaluator_review = evaluator_result.messages[-1].content

    print(evaluator_review)

    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())

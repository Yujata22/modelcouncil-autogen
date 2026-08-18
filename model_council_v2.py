import asyncio
import json
import os

from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


load_dotenv()


async def main():

    # =========================================================
    # 1. LOAD REAL MODEL EVIDENCE
    # =========================================================

    with open("ml/model_metrics.json", "r") as f:
        model_metrics = json.load(f)

    with open("ml/segment_metrics.json", "r") as f:
        segment_metrics = json.load(f)

    metrics_text = json.dumps(model_metrics, indent=2)
    segment_text = json.dumps(segment_metrics, indent=2)

    # =========================================================
    # 2. CREATE GROQ MODEL CLIENT
    # =========================================================

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

    # =========================================================
    # 3. DEFINE AUTOGEN AGENTS
    # =========================================================

    # ---------------------------------------------------------
    # Agent 1: Senior Data Scientist
    # ---------------------------------------------------------

    data_scientist = AssistantAgent(
        name="data_scientist",
        model_client=model_client,
        system_message="""
        You are a Senior Data Scientist serving on an ML Model Review Council.

        You are responsible for comparing candidate machine-learning models
        using objective evidence.

        Evaluate:
        - accuracy
        - precision
        - recall
        - F1 score
        - ROC-AUC
        - precision/recall tradeoffs
        - likely business impact

        Do NOT simply choose the model with the highest accuracy.

        Your recommendation must be evidence-based.

        You may recommend only ONE model for further review.

        Clearly explain:
        - why you selected it
        - what tradeoffs exist
        - what additional evidence may still be required
        """,
    )

    # ---------------------------------------------------------
    # Agent 2: Independent Model Evaluator
    # ---------------------------------------------------------

    model_evaluator = AssistantAgent(
        name="model_evaluator",
        model_client=model_client,
        system_message="""
        You are an independent Machine Learning Model Evaluation specialist.

        Your job is NOT to automatically agree with the Data Scientist.

        Critically evaluate the recommendation.

        Look for:
        - unsupported assumptions
        - misleading metric interpretations
        - differences that may not be practically significant
        - missing validation evidence
        - class imbalance concerns
        - threshold-selection concerns
        - overfitting risk
        - calibration requirements
        - deployment considerations

        Compare competing models when appropriate.

        You should explicitly challenge claims that are not supported
        by the available evidence.
        """,
    )

    # ---------------------------------------------------------
    # Agent 3: Risk / Governance Specialist
    # ---------------------------------------------------------

    risk_agent = AssistantAgent(
        name="risk_agent",
        model_client=model_client,
        system_message="""
        You are an ML Risk and Model Governance specialist.

        Your responsibility is to evaluate whether strong overall model
        performance hides weaknesses across business segments.

        Inspect subgroup and segment-level performance carefully.

        Look for:
        - unusually low recall
        - unusually low precision
        - weak F1 scores
        - large performance gaps between segments
        - segment performance substantially worse than overall performance
        - insufficient sample sizes
        - operational or customer risks

        IMPORTANT:

        A performance difference between business segments is NOT
        automatically evidence of algorithmic bias.

        Do not call a result "bias" unless the evidence supports that claim.

        Instead distinguish among:
        - performance disparity
        - model risk
        - potential fairness concern
        - demonstrated bias

        Your job is to determine whether the model should:

        PROCEED

        PROCEED_WITH_CONDITIONS

        STOP
        """,
    )

    # =========================================================
    # 4. DATA SCIENTIST REVIEW
    # =========================================================

    print("\n")
    print("=" * 70)
    print("DATA SCIENTIST")
    print("=" * 70)
    print()

    ds_result = await data_scientist.run(
        task=f"""
        Three machine-learning models were trained for a churn
        classification problem.

        The following metrics were calculated on the test dataset:

        {metrics_text}

        Compare:

        - Logistic Regression
        - Random Forest
        - XGBoost

        Recommend ONE model for further review.

        Structure your response as:

        1. Recommended Model

        2. Evidence Supporting the Recommendation

        3. Precision vs Recall Tradeoff

        4. Comparison Against the Other Models

        5. Remaining Concerns

        6. Recommended Next Validation Step
        """
    )

    ds_review = ds_result.messages[-1].content

    print(ds_review)

    # =========================================================
    # 5. INDEPENDENT EVALUATOR REVIEW
    # =========================================================

    print("\n")
    print("=" * 70)
    print("MODEL EVALUATOR")
    print("=" * 70)
    print()

    evaluator_result = await model_evaluator.run(
        task=f"""
        You are independently reviewing a model recommendation.

        Actual model metrics:

        {metrics_text}


        The Senior Data Scientist provided the following recommendation:

        -----------------------------

        {ds_review}

        -----------------------------

        Critically evaluate this recommendation.

        Do not automatically agree with it.

        Structure your response as:

        1. What You Agree With

        2. What You Disagree With

        3. Unsupported Assumptions

        4. Missing Evidence

        5. Your Preferred Model

        6. Should the Recommendation Proceed?

        Your final answer for item 6 must be one of:

        YES

        YES_WITH_ADDITIONAL_VALIDATION

        NO
        """
    )

    evaluator_review = evaluator_result.messages[-1].content

    print(evaluator_review)

    # =========================================================
    # 6. RISK / GOVERNANCE REVIEW
    # =========================================================

    print("\n")
    print("=" * 70)
    print("RISK / GOVERNANCE AGENT")
    print("=" * 70)
    print()

    risk_result = await risk_agent.run(
        task=f"""
        You are conducting an ML model-risk and governance review.

        The candidate models have these overall performance metrics:

        {metrics_text}


        The current XGBoost candidate has the following segment-level
        evaluation results:

        {segment_text}


        The Senior Data Scientist concluded:

        -----------------------------

        {ds_review}

        -----------------------------


        The Independent Model Evaluator concluded:

        -----------------------------

        {evaluator_review}

        -----------------------------


        Evaluate the available evidence carefully.

        Pay special attention to differences between overall model
        performance and segment-level performance.

        Structure your response as:

        1. Highest-Risk Segment

        Identify the segment that creates the greatest concern.


        2. Evidence

        Provide the exact precision, recall and F1 values that support
        your concern.


        3. Overall vs Segment Comparison

        Compare the risky segment against the overall XGBoost metrics.


        4. Performance Gap

        Explain how large or important the performance difference appears.


        5. Business Impact

        Explain what could happen if this weakness appears in production.


        6. Bias vs Performance Disparity

        Explicitly state whether the available evidence demonstrates:

        - performance disparity
        - potential fairness concern
        - demonstrated algorithmic bias

        Do NOT claim algorithmic bias without sufficient evidence.


        7. Missing Evidence

        Identify what additional evidence should be collected.


        8. Risk Rating

        Choose exactly one:

        LOW
        MEDIUM
        HIGH


        9. Governance Recommendation

        Choose exactly one:

        PROCEED

        PROCEED_WITH_CONDITIONS

        STOP


        10. Required Actions

        List concrete steps that should be completed before production
        deployment.
        """
    )

    risk_review = risk_result.messages[-1].content

    print(risk_review)

    # =========================================================
    # 7. SAVE COUNCIL OUTPUTS
    # =========================================================

    council_output = {
        "data_scientist_review": ds_review,
        "model_evaluator_review": evaluator_review,
        "risk_governance_review": risk_review,
    }

    with open("ml/model_council_v2_output.json", "w") as f:
        json.dump(
            council_output,
            f,
            indent=4
        )

    print("\n")
    print("=" * 70)
    print("MODELCOUNCIL V2 COMPLETE")
    print("=" * 70)

    print(
        "\nSaved council output to "
        "ml/model_council_v2_output.json"
    )

    # =========================================================
    # 8. CLOSE MODEL CLIENT
    # =========================================================

    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())

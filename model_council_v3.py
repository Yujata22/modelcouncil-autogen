import argparse
import asyncio
import json
import os

from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import (
    MaxMessageTermination,
    TextMentionTermination,
)
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console

from autogen_core.model_context import BufferedChatCompletionContext

from autogen_ext.models.openai import OpenAIChatCompletionClient


load_dotenv()


# =========================================================
# 1. COMMAND-LINE ARGUMENTS
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ModelCouncil against an ML governance scenario."
    )

    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help=(
            "Scenario from evaluation/scenarios.json. "
            "If omitted, use locally generated ML outputs."
        ),
    )

    return parser.parse_args()


# =========================================================
# 2. LOAD EVIDENCE
# =========================================================

def load_evidence(scenario_name=None):

    if scenario_name:

        with open("evaluation/scenarios.json", "r") as f:
            scenarios = json.load(f)

        if scenario_name not in scenarios:

            available = ", ".join(scenarios.keys())

            raise ValueError(
                f"Unknown scenario: {scenario_name}\n"
                f"Available scenarios: {available}"
            )

        scenario = scenarios[scenario_name]

        return {
            "scenario_name": scenario_name,
            "description": scenario.get("description", ""),
            "overall_metrics": scenario["overall_metrics"],
            "segment_metrics": scenario["segment_metrics"],
            "expected_decision": scenario.get("expected_decision"),
        }

    # -----------------------------------------------------
    # Real locally generated ML evidence
    # -----------------------------------------------------

    with open("ml/model_metrics.json", "r") as f:
        model_metrics = json.load(f)

    with open("ml/segment_metrics.json", "r") as f:
        segment_metrics = json.load(f)

    return {
        "scenario_name": "real_model",
        "description": "Evaluation using locally generated ML metrics.",
        "overall_metrics": model_metrics,
        "segment_metrics": segment_metrics,
        "expected_decision": "APPROVE_WITH_CONDITIONS",
    }


# =========================================================
# 3. MAIN
# =========================================================

async def main():

    args = parse_args()

    evidence = load_evidence(args.scenario)

    scenario_name = evidence["scenario_name"]
    description = evidence["description"]
    expected_decision = evidence["expected_decision"]
    model_metrics = evidence["overall_metrics"]
    segment_metrics = evidence["segment_metrics"]

    # Compact JSON: avoids wasting tokens on indentation/whitespace
    metrics_text = json.dumps(
        model_metrics,
        separators=(",", ":")
    )

    segment_text = json.dumps(
        segment_metrics,
        separators=(",", ":")
    )

    print("\n" + "=" * 70)
    print("MODELCOUNCIL SCENARIO")
    print("=" * 70)

    print(f"\nScenario: {scenario_name}")
    print(f"Description: {description}")

    if expected_decision:
        print(
            f"Benchmark expected decision: {expected_decision}"
        )

    # =====================================================
    # 4. GROQ MODEL CLIENT
    # =====================================================

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:

        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Add it to your .env file."
        )

    model_client = OpenAIChatCompletionClient(

        model="qwen/qwen3.6-27b",

        base_url="https://api.groq.com/openai/v1",

        api_key=api_key,

        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
            "structured_output": False,
        },

        # =================================================
        # IMPORTANT FIX #1
        # Disable Qwen reasoning tokens.
        # =================================================
        extra_body={
            "reasoning_effort": "none"
        },
    )

    # =====================================================
    # CONTEXT STRATEGY
    # =====================================================
    #
    # AutoGen normally uses an unbounded context.
    #
    # That means later agents may receive:
    #
    # task
    # + DS response
    # + evaluator response
    # + risk response
    # + critic response
    #
    # which caused the final call to exceed Groq's 8K TPM.
    #
    # Each agent below therefore uses a small rolling buffer.
    #
    # =====================================================


    # =====================================================
    # 5. DATA SCIENTIST
    # =====================================================

    data_scientist = AssistantAgent(

        name="data_scientist",

        description="Selects the strongest candidate ML model.",

        model_client=model_client,

        # =================================================
        # IMPORTANT FIX #2
        # Task/current evidence only.
        # =================================================
        model_context=BufferedChatCompletionContext(
            buffer_size=1
        ),

        system_message="""
You are a Senior Data Scientist.

Evaluate candidate models using precision, recall, F1 and ROC-AUC.

Return only:

Recommended Model: <model>
Evidence: <one sentence>
Tradeoff: <one sentence>
Concern: <one sentence>

Maximum 70 words.
Do not reproduce tables.
""",
    )

    # =====================================================
    # 6. MODEL EVALUATOR
    # =====================================================

    model_evaluator = AssistantAgent(

        name="model_evaluator",

        description="Independently challenges model selection.",

        model_client=model_client,

        # Current task + previous response is sufficient.
        model_context=BufferedChatCompletionContext(
            buffer_size=2
        ),

        system_message="""
You are an independent ML evaluator.

Challenge the previous recommendation using validation,
threshold, calibration and business tradeoff considerations.

Return only:

Agreement: <one sentence>
Challenge: <one sentence>
Missing Evidence: <one sentence>
Evaluator Position: <model or uncertainty>

Maximum 70 words.
""",
    )

    # =====================================================
    # 7. RISK AGENT
    # =====================================================

    risk_agent = AssistantAgent(

        name="risk_agent",

        description="Reviews segment-level model risk.",

        model_client=model_client,

        model_context=BufferedChatCompletionContext(
            buffer_size=2
        ),

        system_message="""
You are an ML Risk and Governance reviewer.

Focus on segment performance:
recall, precision, F1, segment gaps and operational impact.

Return only:

Highest-Risk Segment: <segment>
Performance Gap: <one sentence>
Risk Rating: <LOW | MEDIUM | HIGH>
Recommended Action: <one sentence>

Do not call disparity bias without evidence.
Maximum 65 words.
""",
    )

    # =====================================================
    # 8. CRITIC
    # =====================================================

    critic = AssistantAgent(

        name="critic",

        description="Challenges unsupported governance conclusions.",

        model_client=model_client,

        model_context=BufferedChatCompletionContext(
            buffer_size=2
        ),

        system_message="""
You are an adversarial ML governance critic.

Identify unsupported assumptions or missing evidence.

Return only:

Weakness 1: <brief>
Weakness 2: <brief>
Missing Test: <brief>
Conclusion: <brief>

Maximum 55 words.
""",
    )

    # =====================================================
    # 9. FINAL REVIEWER
    # =====================================================

    final_reviewer = AssistantAgent(

        name="final_reviewer",

        description="Issues the final model governance decision.",

        model_client=model_client,

        # =================================================
        # IMPORTANT:
        # Keep the last 4 specialist messages.
        # This gives the chair enough evidence without
        # sending unlimited history.
        # =================================================
        model_context=BufferedChatCompletionContext(
            buffer_size=4
        ),

        system_message="""
You are the ModelCouncil Chair.

Synthesize the specialist reviews.

Policy:

APPROVE:
Strong performance with no material segment degradation
or critical unresolved risk.

APPROVE_WITH_CONDITIONS:
Model is viable but material concerns remain.

REJECT:
Overall performance is inadequate or deployment risk is severe.

Do not use any benchmark expected answer.

You MUST return:

FINAL_DECISION

Recommended Model: <model>

Governance Decision: <APPROVE | APPROVE_WITH_CONDITIONS | REJECT>

Risk Level: <LOW | MEDIUM | HIGH>

Primary Reason: <one sentence>

Required Actions:
1. <brief>
2. <brief>

TERMINATE

Maximum 90 words.
""",
    )

    # =====================================================
    # 10. TERMINATION
    # =====================================================

    termination = (
        TextMentionTermination("TERMINATE")
        |
        MaxMessageTermination(max_messages=7)
    )

    # =====================================================
    # 11. ROUND ROBIN TEAM
    # =====================================================

    team = RoundRobinGroupChat(

        participants=[
            data_scientist,
            model_evaluator,
            risk_agent,
            critic,
            final_reviewer,
        ],

        termination_condition=termination,

        max_turns=5,
    )

    # =====================================================
    # 12. GOVERNANCE TASK
    # =====================================================

    task = f"""
ML governance review.

Scenario:
{description}

Candidate metrics:
{metrics_text}

XGBoost segment metrics:
{segment_text}

Review:
1. best model,
2. model tradeoffs,
3. segment risk,
4. missing evidence,
5. deployment decision.

Allowed decisions:
APPROVE
APPROVE_WITH_CONDITIONS
REJECT

Use only supplied evidence.
"""

    # =====================================================
    # 13. RUN
    # =====================================================

    print("\n" + "=" * 70)
    print("MODELCOUNCIL — GOVERNANCE RUN")
    print("=" * 70)

    print("\nRouting: RoundRobinGroupChat")
    print(
        "Sequence: "
        "data_scientist -> "
        "model_evaluator -> "
        "risk_agent -> "
        "critic -> "
        "final_reviewer"
    )

    print(
        "Context strategy: BufferedChatCompletionContext"
    )

    print(
        "Reasoning: disabled for Qwen"
    )

    print()

    try:

        result = await Console(
            team.run_stream(task=task),
            output_stats=True,
        )

        # =================================================
        # 14. OUTPUT DIRECTORY
        # =================================================

        os.makedirs(
            "outputs",
            exist_ok=True
        )

        safe_name = (
            scenario_name
            .replace(" ", "_")
            .lower()
        )

        # =================================================
        # 15. TRANSCRIPT
        # =================================================

        transcript = []

        for message in result.messages:

            source = getattr(
                message,
                "source",
                "unknown"
            )

            content = getattr(
                message,
                "content",
                ""
            )

            if not isinstance(content, str):
                content = str(content)

            transcript.append(
                {
                    "speaker": source,
                    "content": content,
                }
            )

        transcript_path = (
            f"outputs/{safe_name}_transcript.json"
        )

        transcript_output = {

            "scenario": scenario_name,

            "description": description,

            "expected_decision":
                expected_decision,

            "model_provider":
                "Groq",

            "model":
                "qwen/qwen3.6-27b",

            "reasoning_effort":
                "none",

            "routing_mode":
                "RoundRobinGroupChat",

            "context_strategy":
                "BufferedChatCompletionContext",

            "stop_reason":
                result.stop_reason,

            "message_count":
                len(result.messages),

            "conversation":
                transcript,
        }

        with open(
            transcript_path,
            "w"
        ) as f:

            json.dump(
                transcript_output,
                f,
                indent=2
            )

        # =================================================
        # 16. SPEAKER ORDER
        # =================================================

        speaker_order = []

        for message in result.messages:

            source = getattr(
                message,
                "source",
                None
            )

            if (
                source
                and source != "user"
            ):

                speaker_order.append(
                    source
                )

        speaker_path = (
            f"outputs/{safe_name}_speaker_order.json"
        )

        with open(
            speaker_path,
            "w"
        ) as f:

            json.dump(
                {
                    "scenario":
                        scenario_name,

                    "routing_mode":
                        "RoundRobinGroupChat",

                    "speaker_order":
                        speaker_order,
                },
                f,
                indent=2
            )

        # =================================================
        # 17. FINAL REVIEWER RESPONSE
        # =================================================

        final_decision = None

        for message in reversed(
            result.messages
        ):

            source = getattr(
                message,
                "source",
                ""
            )

            content = getattr(
                message,
                "content",
                ""
            )

            if (
                source == "final_reviewer"
                and isinstance(content, str)
            ):

                final_decision = content
                break

        # =================================================
        # 18. SAVE FINAL DECISION
        # =================================================

        decision_path = (
            f"outputs/{safe_name}_final_decision.txt"
        )

        if final_decision:

            with open(
                decision_path,
                "w"
            ) as f:

                f.write(final_decision)

        # =================================================
        # 19. SUMMARY
        # =================================================

        print("\n" + "=" * 70)
        print("RUN SUMMARY")
        print("=" * 70)

        print(
            f"\nScenario: {scenario_name}"
        )

        print(
            "Routing: RoundRobinGroupChat"
        )

        print(
            "Model: qwen/qwen3.6-27b"
        )

        print(
            "Reasoning: none"
        )

        print(
            "Speaker order: "
            + " -> ".join(
                speaker_order
            )
        )

        print(
            f"Messages: "
            f"{len(result.messages)}"
        )

        print(
            f"Stop reason: "
            f"{result.stop_reason}"
        )

        print(
            f"\nTranscript: "
            f"{transcript_path}"
        )

        print(
            f"Speaker order: "
            f"{speaker_path}"
        )

        if final_decision:

            print(
                f"Final decision: "
                f"{decision_path}"
            )

            print(
                "\n" + "=" * 70
            )

            print(
                "FINAL REVIEWER OUTPUT"
            )

            print(
                "=" * 70
            )

            print(
                f"\n{final_decision}"
            )

        else:

            print(
                "\nWARNING: "
                "No final reviewer response found."
            )

    except Exception as e:

        print("\n" + "=" * 70)
        print("MODELCOUNCIL ERROR")
        print("=" * 70)

        print(f"\n{e}")

    finally:

        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
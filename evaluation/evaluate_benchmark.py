import json
import os
import re


SCENARIOS_PATH = "evaluation/scenarios.json"
OUTPUT_DIR = "outputs"

BENCHMARK_OUTPUT = (
    "outputs/benchmark_results.json"
)


# =========================================================
# LOAD HELPERS
# =========================================================

def load_json(path):

    with open(
        path,
        "r"
    ) as f:

        return json.load(f)


def load_text(path):

    with open(
        path,
        "r"
    ) as f:

        return f.read()


# =========================================================
# DECISION PARSING
# =========================================================

def extract_decision(text):

    if not text:
        return None

    match = re.search(
        r"Governance Decision:\s*"
        r"(APPROVE_WITH_CONDITIONS|APPROVE|REJECT)",
        text,
        re.IGNORECASE,
    )

    if match:

        return (
            match
            .group(1)
            .upper()
        )

    return None


def extract_risk_level(text):

    if not text:
        return None

    match = re.search(
        r"Risk Level:\s*"
        r"(LOW|MEDIUM|HIGH)",
        text,
        re.IGNORECASE,
    )

    if match:

        return (
            match
            .group(1)
            .upper()
        )

    return None


def extract_model(text):

    if not text:
        return None

    match = re.search(
        r"Recommended Model:\s*([^\n]+)",
        text,
        re.IGNORECASE,
    )

    if match:

        return (
            match
            .group(1)
            .strip()
        )

    return None


# =========================================================
# ROUTING EVALUATION
# =========================================================

def evaluate_routing(speaker_order):

    specialists = {
        "data_scientist",
        "model_evaluator",
        "risk_agent",
        "critic",
    }

    speakers = set(
        speaker_order
    )

    specialists_present = (
        specialists.issubset(
            speakers
        )
    )

    final_present = (
        "final_reviewer"
        in speakers
    )

    routing_complete = (
        specialists_present
        and final_present
    )

    final_position = None

    if final_present:

        final_position = (
            speaker_order.index(
                "final_reviewer"
            )
        )

    premature_final = False

    if final_position is not None:

        speakers_before_final = set(
            speaker_order[
                :final_position
            ]
        )

        premature_final = (
            not specialists.issubset(
                speakers_before_final
            )
        )

    return {
        "routing_complete":
            routing_complete,

        "specialists_present":
            specialists_present,

        "final_reviewer_present":
            final_present,

        "premature_final_reviewer":
            premature_final,

        "agent_turns":
            len(
                speaker_order
            ),
    }


# =========================================================
# MAIN BENCHMARK
# =========================================================

def main():

    scenarios = load_json(
        SCENARIOS_PATH
    )

    scenario_results = []

    total_scenarios = 0
    correct_decisions = 0

    completed_routing = 0
    non_premature_reviews = 0

    total_agent_turns = 0

    decisions_found = 0

    # =====================================================
    # EVALUATE EACH SCENARIO
    # =====================================================

    for scenario_name, scenario in scenarios.items():

        total_scenarios += 1

        expected_decision = (
            scenario[
                "expected_decision"
            ]
        )

        decision_path = (
            f"{OUTPUT_DIR}/"
            f"{scenario_name}_"
            f"final_decision.txt"
        )

        speaker_path = (
            f"{OUTPUT_DIR}/"
            f"{scenario_name}_"
            f"speaker_order.json"
        )

        transcript_path = (
            f"{OUTPUT_DIR}/"
            f"{scenario_name}_"
            f"transcript.json"
        )

        # -------------------------------------------------
        # Missing scenario output
        # -------------------------------------------------

        if (
            not os.path.exists(
                decision_path
            )
            or not os.path.exists(
                speaker_path
            )
        ):

            scenario_results.append(
                {
                    "scenario":
                        scenario_name,

                    "status":
                        "MISSING_OUTPUT",

                    "expected_decision":
                        expected_decision,
                }
            )

            continue

        # -------------------------------------------------
        # Load outputs
        # -------------------------------------------------

        final_text = load_text(
            decision_path
        )

        speaker_data = load_json(
            speaker_path
        )

        speaker_order = (
            speaker_data.get(
                "speaker_order",
                []
            )
        )

        # -------------------------------------------------
        # Parse decision
        # -------------------------------------------------

        actual_decision = (
            extract_decision(
                final_text
            )
        )

        risk_level = (
            extract_risk_level(
                final_text
            )
        )

        recommended_model = (
            extract_model(
                final_text
            )
        )

        if actual_decision:

            decisions_found += 1

        decision_correct = (
            actual_decision
            == expected_decision
        )

        if decision_correct:

            correct_decisions += 1

        # -------------------------------------------------
        # Routing
        # -------------------------------------------------

        routing = (
            evaluate_routing(
                speaker_order
            )
        )

        if routing[
            "routing_complete"
        ]:

            completed_routing += 1

        if not routing[
            "premature_final_reviewer"
        ]:

            non_premature_reviews += 1

        total_agent_turns += (
            routing[
                "agent_turns"
            ]
        )

        # -------------------------------------------------
        # Transcript metadata
        # -------------------------------------------------

        message_count = None
        stop_reason = None

        if os.path.exists(
            transcript_path
        ):

            transcript = load_json(
                transcript_path
            )

            message_count = (
                transcript.get(
                    "message_count"
                )
            )

            stop_reason = (
                transcript.get(
                    "stop_reason"
                )
            )

        # -------------------------------------------------
        # Scenario result
        # -------------------------------------------------

        scenario_results.append(
            {
                "scenario":
                    scenario_name,

                "description":
                    scenario.get(
                        "description"
                    ),

                "status":
                    "COMPLETE",

                "expected_decision":
                    expected_decision,

                "actual_decision":
                    actual_decision,

                "decision_correct":
                    decision_correct,

                "recommended_model":
                    recommended_model,

                "risk_level":
                    risk_level,

                "speaker_order":
                    speaker_order,

                "routing_complete":
                    routing[
                        "routing_complete"
                    ],

                "premature_final_reviewer":
                    routing[
                        "premature_final_reviewer"
                    ],

                "agent_turns":
                    routing[
                        "agent_turns"
                    ],

                "message_count":
                    message_count,

                "stop_reason":
                    stop_reason,
            }
        )

    # =====================================================
    # AGGREGATE METRICS
    # =====================================================

    governance_accuracy = (
        correct_decisions
        / total_scenarios
        if total_scenarios
        else 0
    )

    routing_completion_rate = (
        completed_routing
        / total_scenarios
        if total_scenarios
        else 0
    )

    reviewer_timing_accuracy = (
        non_premature_reviews
        / total_scenarios
        if total_scenarios
        else 0
    )

    decision_completion_rate = (
        decisions_found
        / total_scenarios
        if total_scenarios
        else 0
    )

    average_agent_turns = (
        total_agent_turns
        / total_scenarios
        if total_scenarios
        else 0
    )

    # =====================================================
    # FINAL OUTPUT
    # =====================================================

    benchmark = {

        "summary": {

            "total_scenarios":
                total_scenarios,

            "correct_decisions":
                correct_decisions,

            "governance_accuracy":
                round(
                    governance_accuracy,
                    4
                ),

            "governance_accuracy_percent":
                round(
                    governance_accuracy
                    * 100,
                    2
                ),

            "decision_completion_rate":
                round(
                    decision_completion_rate,
                    4
                ),

            "routing_completion_rate":
                round(
                    routing_completion_rate,
                    4
                ),

            "reviewer_timing_accuracy":
                round(
                    reviewer_timing_accuracy,
                    4
                ),

            "average_agent_turns":
                round(
                    average_agent_turns,
                    2
                ),
        },

        "scenarios":
            scenario_results,
    }

    # =====================================================
    # SAVE
    # =====================================================

    with open(
        BENCHMARK_OUTPUT,
        "w"
    ) as f:

        json.dump(
            benchmark,
            f,
            indent=4
        )

    # =====================================================
    # PRINT SUMMARY
    # =====================================================

    print("\n")
    print("=" * 70)
    print("MODELCOUNCIL BENCHMARK")
    print("=" * 70)

    print(
        f"\nScenarios: "
        f"{total_scenarios}"
    )

    print(
        f"Correct decisions: "
        f"{correct_decisions}/"
        f"{total_scenarios}"
    )

    print(
        f"Governance accuracy: "
        f"{round(governance_accuracy * 100, 2)}%"
    )

    print(
        f"Decision completion: "
        f"{round(decision_completion_rate * 100, 2)}%"
    )

    print(
        f"Routing completion: "
        f"{round(routing_completion_rate * 100, 2)}%"
    )

    print(
        f"Reviewer timing accuracy: "
        f"{round(reviewer_timing_accuracy * 100, 2)}%"
    )

    print(
        f"Average agent turns: "
        f"{round(average_agent_turns, 2)}"
    )

    print("\n--- SCENARIOS ---")

    for result in scenario_results:

        print(
            f"\n{result['scenario']}"
        )

        print(
            "Expected:",
            result.get(
                "expected_decision"
            )
        )

        print(
            "Actual:",
            result.get(
                "actual_decision"
            )
        )

        print(
            "Correct:",
            result.get(
                "decision_correct"
            )
        )

    print(
        f"\nSaved benchmark to: "
        f"{BENCHMARK_OUTPUT}"
    )


if __name__ == "__main__":
    main()

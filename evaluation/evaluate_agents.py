import json
import os
import re


TRANSCRIPT_PATH = "outputs/model_council_v3_transcript.json"
SPEAKER_PATH = "outputs/speaker_order.json"
OUTPUT_PATH = "outputs/agent_evaluation.json"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def extract_final_decision_from_transcript(transcript):
    """
    Find the most recent final_reviewer message containing FINAL_DECISION.
    """

    for message in reversed(transcript):

        speaker = message.get("speaker", "")
        content = message.get("content", "")

        if (
            speaker == "final_reviewer"
            and isinstance(content, str)
            and "FINAL_DECISION" in content
        ):
            return content

    return None


def extract_governance_decision(text):

    if not text:
        return None

    patterns = [
        r"Governance Decision:\s*(APPROVE_WITH_CONDITIONS|APPROVE|REJECT)",
        r"\b(APPROVE_WITH_CONDITIONS|APPROVE|REJECT)\b",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE,
        )

        if match:
            return match.group(1).upper()

    return None


def extract_risk_level(text):

    if not text:
        return None

    match = re.search(
        r"Risk Level:\s*(LOW|MEDIUM|HIGH)",
        text,
        re.IGNORECASE,
    )

    if match:
        return match.group(1).upper()

    return None


def extract_recommended_model(text):

    if not text:
        return None

    match = re.search(
        r"Recommended Model:\s*([^\n]+)",
        text,
        re.IGNORECASE,
    )

    if match:
        return match.group(1).strip()

    return None


# =========================================================
# ROUTING EVALUATION
# =========================================================

def evaluate_routing(speaker_order):

    required_agents = {
        "data_scientist",
        "model_evaluator",
        "risk_agent",
        "critic",
        "final_reviewer",
    }

    present_agents = set(speaker_order)

    missing_agents = sorted(
        required_agents - present_agents
    )

    all_required_present = (
        len(missing_agents) == 0
    )

    final_reviewer_positions = [
        i
        for i, speaker in enumerate(speaker_order)
        if speaker == "final_reviewer"
    ]

    premature_final_reviewer = False

    if final_reviewer_positions:

        first_final_position = (
            final_reviewer_positions[0]
        )

        speakers_before_final = set(
            speaker_order[:first_final_position]
        )

        required_before_final = {
            "data_scientist",
            "model_evaluator",
            "risk_agent",
            "critic",
        }

        if not required_before_final.issubset(
            speakers_before_final
        ):
            premature_final_reviewer = True

    else:
        premature_final_reviewer = True

    unique_agent_count = len(
        set(speaker_order)
    )

    return {
        "all_required_agents_present":
            all_required_present,

        "missing_agents":
            missing_agents,

        "premature_final_reviewer":
            premature_final_reviewer,

        "unique_agent_count":
            unique_agent_count,

        "speaker_order":
            speaker_order,
    }


# =========================================================
# RISK DETECTION EVALUATION
# =========================================================

def evaluate_risk_detection(transcript):

    risk_messages = [
        message
        for message in transcript
        if message.get("speaker") == "risk_agent"
    ]

    if not risk_messages:

        return {
            "risk_agent_participated": False,
            "high_engagement_detected": False,
            "recall_045_detected": False,
            "risk_rating_detected": False,
        }

    combined_text = " ".join(
        message.get("content", "")
        for message in risk_messages
    ).lower()

    high_engagement_detected = (
        "high-engagement" in combined_text
        or "high_engagement" in combined_text
        or "high engagement" in combined_text
    )

    recall_045_detected = any(
        pattern in combined_text
        for pattern in [
            "0.45",
            "45%",
            "45 %",
        ]
    )

    risk_rating_detected = any(
        rating in combined_text
        for rating in [
            "risk rating: high",
            "risk rating: medium",
            "risk rating: low",
            "high risk",
            "medium risk",
            "low risk",
        ]
    )

    return {
        "risk_agent_participated":
            True,

        "high_engagement_detected":
            high_engagement_detected,

        "recall_045_detected":
            recall_045_detected,

        "risk_rating_detected":
            risk_rating_detected,
    }


# =========================================================
# CRITIC EVALUATION
# =========================================================

def evaluate_critic(transcript):

    critic_messages = [
        message
        for message in transcript
        if message.get("speaker") == "critic"
    ]

    if not critic_messages:

        return {
            "critic_participated": False,
            "challenged_reasoning": False,
        }

    text = " ".join(
        message.get("content", "")
        for message in critic_messages
    ).lower()

    challenge_terms = [
        "unsupported",
        "assumption",
        "contradiction",
        "missing",
        "evidence",
        "weakness",
        "premature",
        "insufficient",
        "cannot conclude",
        "not justified",
    ]

    challenged_reasoning = any(
        term in text
        for term in challenge_terms
    )

    return {
        "critic_participated":
            True,

        "challenged_reasoning":
            challenged_reasoning,
    }


# =========================================================
# FINAL DECISION EVALUATION
# =========================================================

def evaluate_final_decision(final_text):

    governance_decision = (
        extract_governance_decision(
            final_text
        )
    )

    risk_level = (
        extract_risk_level(
            final_text
        )
    )

    recommended_model = (
        extract_recommended_model(
            final_text
        )
    )

    # For this specific scenario, we deliberately
    # introduced a serious segment-level recall problem.
    expected_decision = (
        "APPROVE_WITH_CONDITIONS"
    )

    decision_correct = (
        governance_decision
        == expected_decision
    )

    valid_risk_level = (
        risk_level
        in {
            "LOW",
            "MEDIUM",
            "HIGH",
        }
    )

    final_decision_found = (
        final_text is not None
    )

    return {
        "final_decision_found":
            final_decision_found,

        "recommended_model":
            recommended_model,

        "governance_decision":
            governance_decision,

        "expected_governance_decision":
            expected_decision,

        "decision_correct":
            decision_correct,

        "risk_level":
            risk_level,

        "valid_risk_level":
            valid_risk_level,
    }


# =========================================================
# SCORE
# =========================================================

def calculate_score(
    routing,
    risk,
    critic,
    decision,
):

    checks = {
        "all_required_agents_present":
            routing[
                "all_required_agents_present"
            ],

        "reviewer_not_premature":
            not routing[
                "premature_final_reviewer"
            ],

        "risk_agent_participated":
            risk[
                "risk_agent_participated"
            ],

        "high_engagement_detected":
            risk[
                "high_engagement_detected"
            ],

        "recall_045_detected":
            risk[
                "recall_045_detected"
            ],

        "critic_participated":
            critic[
                "critic_participated"
            ],

        "critic_challenged_reasoning":
            critic[
                "challenged_reasoning"
            ],

        "final_decision_found":
            decision[
                "final_decision_found"
            ],

        "correct_governance_decision":
            decision[
                "decision_correct"
            ],

        "valid_risk_level":
            decision[
                "valid_risk_level"
            ],
    }

    passed = sum(
        1
        for value in checks.values()
        if value
    )

    total = len(checks)

    score = round(
        passed / total,
        4
    )

    return {
        "checks":
            checks,

        "passed_checks":
            passed,

        "total_checks":
            total,

        "score":
            score,

        "score_percent":
            round(
                score * 100,
                2
            ),
    }


# =========================================================
# MAIN
# =========================================================

def main():

    required_files = [
        TRANSCRIPT_PATH,
        SPEAKER_PATH,
    ]

    for path in required_files:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Required file missing: {path}"
            )

    # -----------------------------------------------------
    # Load saved AutoGen run
    # -----------------------------------------------------

    transcript_data = load_json(
        TRANSCRIPT_PATH
    )

    speaker_data = load_json(
        SPEAKER_PATH
    )

    transcript = transcript_data.get(
        "conversation",
        []
    )

    speaker_order = speaker_data.get(
        "speaker_order",
        []
    )

    # -----------------------------------------------------
    # Extract final reviewer directly from transcript
    # -----------------------------------------------------

    final_text = (
        extract_final_decision_from_transcript(
            transcript
        )
    )

    # -----------------------------------------------------
    # Run evaluations
    # -----------------------------------------------------

    routing_eval = evaluate_routing(
        speaker_order
    )

    risk_eval = evaluate_risk_detection(
        transcript
    )

    critic_eval = evaluate_critic(
        transcript
    )

    decision_eval = (
        evaluate_final_decision(
            final_text
        )
    )

    score = calculate_score(
        routing_eval,
        risk_eval,
        critic_eval,
        decision_eval,
    )

    # -----------------------------------------------------
    # Build final evaluation output
    # -----------------------------------------------------

    output = {
        "routing_evaluation":
            routing_eval,

        "risk_detection_evaluation":
            risk_eval,

        "critic_evaluation":
            critic_eval,

        "decision_evaluation":
            decision_eval,

        "overall_evaluation":
            score,

        "conversation_metadata": {
            "message_count":
                transcript_data.get(
                    "message_count"
                ),

            "stop_reason":
                transcript_data.get(
                    "stop_reason"
                ),
        },
    }

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w"
    ) as f:

        json.dump(
            output,
            f,
            indent=4
        )

    # =====================================================
    # PRINT RESULTS
    # =====================================================

    print("\n")
    print("=" * 70)
    print("MODELCOUNCIL AGENT EVALUATION")
    print("=" * 70)

    print(
        f"\nAgent Evaluation Score: "
        f"{score['score_percent']}%"
    )

    print(
        f"Passed Checks: "
        f"{score['passed_checks']}"
        f"/{score['total_checks']}"
    )

    # -----------------------------------------------------
    # Routing
    # -----------------------------------------------------

    print("\n--- ROUTING ---")

    print(
        "All required agents present:",
        routing_eval[
            "all_required_agents_present"
        ],
    )

    print(
        "Missing agents:",
        routing_eval[
            "missing_agents"
        ],
    )

    print(
        "Premature final reviewer:",
        routing_eval[
            "premature_final_reviewer"
        ],
    )

    print(
        "Speaker order:",
        " -> ".join(
            speaker_order
        ),
    )

    # -----------------------------------------------------
    # Risk
    # -----------------------------------------------------

    print("\n--- RISK DETECTION ---")

    print(
        "Risk agent participated:",
        risk_eval[
            "risk_agent_participated"
        ],
    )

    print(
        "High-engagement segment detected:",
        risk_eval[
            "high_engagement_detected"
        ],
    )

    print(
        "Recall 0.45 detected:",
        risk_eval[
            "recall_045_detected"
        ],
    )

    # -----------------------------------------------------
    # Critic
    # -----------------------------------------------------

    print("\n--- CRITIC ---")

    print(
        "Critic participated:",
        critic_eval[
            "critic_participated"
        ],
    )

    print(
        "Critic challenged reasoning:",
        critic_eval[
            "challenged_reasoning"
        ],
    )

    # -----------------------------------------------------
    # Decision
    # -----------------------------------------------------

    print("\n--- FINAL DECISION ---")

    print(
        "Final decision found:",
        decision_eval[
            "final_decision_found"
        ],
    )

    print(
        "Recommended model:",
        decision_eval[
            "recommended_model"
        ],
    )

    print(
        "Governance decision:",
        decision_eval[
            "governance_decision"
        ],
    )

    print(
        "Expected decision:",
        decision_eval[
            "expected_governance_decision"
        ],
    )

    print(
        "Decision correct:",
        decision_eval[
            "decision_correct"
        ],
    )

    print(
        "Risk level:",
        decision_eval[
            "risk_level"
        ],
    )

    print(
        f"\nSaved evaluation to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
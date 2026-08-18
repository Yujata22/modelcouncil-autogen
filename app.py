import json
import os
import re

import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="ModelCouncil",
    page_icon="🧠",
    layout="wide",
)


# =========================================================
# CONSTANTS
# =========================================================

SCENARIOS_PATH = "evaluation/scenarios.json"
BENCHMARK_PATH = "outputs/benchmark_results.json"
OUTPUT_DIR = "outputs"


# =========================================================
# HELPERS
# =========================================================

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_text(path):
    with open(path, "r") as f:
        return f.read()


def extract_field(text, field_name):
    if not text:
        return None

    pattern = rf"{re.escape(field_name)}:\s*([^\n]+)"

    match = re.search(
        pattern,
        text,
        re.IGNORECASE,
    )

    if match:
        return match.group(1).strip()

    return None


def get_scenario_output_paths(scenario_name):

    safe_name = (
        scenario_name
        .replace(" ", "_")
        .lower()
    )

    return {
        "transcript":
            f"{OUTPUT_DIR}/{safe_name}_transcript.json",

        "speaker_order":
            f"{OUTPUT_DIR}/{safe_name}_speaker_order.json",

        "final_decision":
            f"{OUTPUT_DIR}/{safe_name}_final_decision.txt",
    }


def load_scenario_run(scenario_name):

    paths = get_scenario_output_paths(
        scenario_name
    )

    transcript = None
    speaker_order = []
    final_decision = None

    if os.path.exists(
        paths["transcript"]
    ):
        transcript = load_json(
            paths["transcript"]
        )

    if os.path.exists(
        paths["speaker_order"]
    ):
        speaker_data = load_json(
            paths["speaker_order"]
        )

        speaker_order = speaker_data.get(
            "speaker_order",
            []
        )

    if os.path.exists(
        paths["final_decision"]
    ):
        final_decision = load_text(
            paths["final_decision"]
        )

    return {
        "transcript": transcript,
        "speaker_order": speaker_order,
        "final_decision": final_decision,
    }


def build_model_metric_df(overall_metrics):

    rows = []

    for model_name, metrics in overall_metrics.items():

        row = {
            "Model": model_name.replace(
                "_",
                " "
            ).title()
        }

        row.update(metrics)

        rows.append(row)

    return pd.DataFrame(rows)


def build_segment_df(segment_metrics):

    rows = []

    # -----------------------------------------------------
    # Flat benchmark structure
    # -----------------------------------------------------

    flat_structure = all(
        isinstance(value, dict)
        and "precision" in value
        for value in segment_metrics.values()
    )

    if flat_structure:

        for segment_name, metrics in segment_metrics.items():

            rows.append(
                {
                    "Segment":
                        segment_name.replace(
                            "_",
                            " "
                        ).title(),

                    "Precision":
                        metrics.get(
                            "precision"
                        ),

                    "Recall":
                        metrics.get(
                            "recall"
                        ),

                    "F1":
                        metrics.get(
                            "f1"
                        ),

                    "Sample Size":
                        metrics.get(
                            "sample_size"
                        ),
                }
            )

    # -----------------------------------------------------
    # Nested real-model structure
    # -----------------------------------------------------

    else:

        for group_name, segments in segment_metrics.items():

            for segment_name, metrics in segments.items():

                rows.append(
                    {
                        "Segment Group":
                            group_name.replace(
                                "_",
                                " "
                            ).title(),

                        "Segment":
                            segment_name.replace(
                                "_",
                                " "
                            ).title(),

                        "Precision":
                            metrics.get(
                                "precision"
                            ),

                        "Recall":
                            metrics.get(
                                "recall"
                            ),

                        "F1":
                            metrics.get(
                                "f1"
                            ),

                        "Sample Size":
                            metrics.get(
                                "sample_size"
                            ),
                    }
                )

    return pd.DataFrame(rows)


# =========================================================
# LOAD DATA
# =========================================================

if not os.path.exists(
    SCENARIOS_PATH
):
    st.error(
        "evaluation/scenarios.json was not found."
    )
    st.stop()


scenarios = load_json(
    SCENARIOS_PATH
)


benchmark = None

if os.path.exists(
    BENCHMARK_PATH
):
    benchmark = load_json(
        BENCHMARK_PATH
    )


# =========================================================
# HEADER
# =========================================================

st.title(
    "🧠 ModelCouncil"
)

st.caption(
    "AutoGen-powered multi-agent ML model governance and evaluation"
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header(
    "Review Controls"
)

scenario_name = st.sidebar.selectbox(
    "Select Scenario",
    list(
        scenarios.keys()
    ),
)


scenario = scenarios[
    scenario_name
]


run_data = load_scenario_run(
    scenario_name
)


st.sidebar.markdown(
    "---"
)

st.sidebar.markdown(
    "**Scenario Description**"
)

st.sidebar.write(
    scenario.get(
        "description",
        ""
    )
)

st.sidebar.markdown(
    "**Expected Governance Decision**"
)

st.sidebar.code(
    scenario.get(
        "expected_decision",
        "N/A"
    )
)


# =========================================================
# FINAL DECISION FIELDS
# =========================================================

final_text = run_data[
    "final_decision"
]

recommended_model = extract_field(
    final_text,
    "Recommended Model"
)

governance_decision = extract_field(
    final_text,
    "Governance Decision"
)

risk_level = extract_field(
    final_text,
    "Risk Level"
)


# =========================================================
# TOP METRICS
# =========================================================

col1, col2, col3, col4 = st.columns(
    4
)

with col1:

    st.metric(
        "Recommended Model",
        recommended_model
        or "Not Run"
    )

with col2:

    st.metric(
        "Governance Decision",
        governance_decision
        or "Not Run"
    )

with col3:

    st.metric(
        "Risk Level",
        risk_level
        or "Not Run"
    )

with col4:

    if benchmark:

        accuracy = (
            benchmark
            .get(
                "summary",
                {}
            )
            .get(
                "governance_accuracy_percent"
            )
        )

        value = (
            f"{accuracy}%"
            if accuracy is not None
            else "N/A"
        )

    else:
        value = "N/A"

    st.metric(
        "Benchmark Accuracy",
        value
    )


st.markdown(
    "---"
)


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Model Review",
        "🤖 Agent Conversation",
        "🧪 Benchmark",
        "🏗 Architecture",
    ]
)


# =========================================================
# TAB 1 — MODEL REVIEW
# =========================================================

with tab1:

    st.subheader(
        "Candidate Model Performance"
    )

    model_df = build_model_metric_df(
        scenario[
            "overall_metrics"
        ]
    )

    st.dataframe(
        model_df,
        use_container_width=True,
        hide_index=True,
    )

    metric_choice = st.selectbox(
        "Compare Metric",
        [
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "accuracy",
        ],
        index=2,
    )

    fig = px.bar(
        model_df,
        x="Model",
        y=metric_choice,
        text_auto=".3f",
        title=(
            f"Model Comparison — "
            f"{metric_choice.upper()}"
        ),
    )

    fig.update_layout(
        yaxis_range=[
            0,
            1
        ]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.subheader(
        "Segment-Level Risk"
    )

    segment_df = build_segment_df(
        scenario[
            "segment_metrics"
        ]
    )

    st.dataframe(
        segment_df,
        use_container_width=True,
        hide_index=True,
    )


    if (
        not segment_df.empty
        and "Recall"
        in segment_df.columns
    ):

        segment_chart_df = (
            segment_df
            .dropna(
                subset=[
                    "Recall"
                ]
            )
        )

        fig_segment = px.bar(
            segment_chart_df,
            x="Segment",
            y="Recall",
            text_auto=".3f",
            title=(
                "Recall by Segment"
            ),
        )

        fig_segment.update_layout(
            yaxis_range=[
                0,
                1
            ]
        )

        st.plotly_chart(
            fig_segment,
            use_container_width=True
        )


    st.subheader(
        "Final Governance Decision"
    )

    if final_text:

        if (
            governance_decision
            == "APPROVE"
        ):
            st.success(
                governance_decision
            )

        elif (
            governance_decision
            == "APPROVE_WITH_CONDITIONS"
        ):
            st.warning(
                governance_decision
            )

        elif (
            governance_decision
            == "REJECT"
        ):
            st.error(
                governance_decision
            )

        st.markdown(
            final_text
        )

    else:

        st.info(
            "This scenario has not been run yet. "
            "Run ModelCouncil from the terminal first."
        )

        st.code(
            f"python model_council_v3.py "
            f"--scenario {scenario_name}"
        )


# =========================================================
# TAB 2 — AGENT CONVERSATION
# =========================================================

with tab2:

    st.subheader(
        "AutoGen Speaker Routing"
    )

    speaker_order = run_data[
        "speaker_order"
    ]

    if speaker_order:

        route_string = "  →  ".join(
            speaker_order
        )

        st.info(
            route_string
        )

        st.caption(
            f"Total agent turns: "
            f"{len(speaker_order)}"
        )

    else:

        st.warning(
            "No speaker-order output found."
        )


    st.subheader(
        "Council Transcript"
    )

    transcript_data = run_data[
        "transcript"
    ]

    if transcript_data:

        conversation = (
            transcript_data.get(
                "conversation",
                []
            )
        )

        for message in conversation:

            speaker = message.get(
                "speaker",
                "unknown"
            )

            content = message.get(
                "content",
                ""
            )

            if speaker == "user":
                continue

            title = (
                speaker
                .replace(
                    "_",
                    " "
                )
                .title()
            )

            with st.expander(
                f"🤖 {title}"
            ):

                st.markdown(
                    content
                )

        st.caption(
            "Stop reason: "
            + str(
                transcript_data.get(
                    "stop_reason"
                )
            )
        )

    else:

        st.info(
            "No transcript found for this scenario."
        )


# =========================================================
# TAB 3 — BENCHMARK
# =========================================================

with tab3:

    st.subheader(
        "ModelCouncil Evaluation Benchmark"
    )

    if not benchmark:

        st.warning(
            "benchmark_results.json was not found."
        )

        st.code(
            "python evaluation/evaluate_benchmark.py"
        )

    else:

        summary = benchmark.get(
            "summary",
            {}
        )

        b1, b2, b3, b4 = st.columns(
            4
        )

        with b1:

            st.metric(
                "Governance Accuracy",
                f"{summary.get('governance_accuracy_percent', 0)}%"
            )

        with b2:

            routing_rate = (
                summary.get(
                    "routing_completion_rate",
                    0
                )
                * 100
            )

            st.metric(
                "Routing Completion",
                f"{routing_rate:.0f}%"
            )

        with b3:

            completion_rate = (
                summary.get(
                    "decision_completion_rate",
                    0
                )
                * 100
            )

            st.metric(
                "Decision Completion",
                f"{completion_rate:.0f}%"
            )

        with b4:

            st.metric(
                "Avg Agent Turns",
                summary.get(
                    "average_agent_turns",
                    0
                )
            )


        benchmark_rows = []

        for result in benchmark.get(
            "scenarios",
            []
        ):

            benchmark_rows.append(
                {
                    "Scenario":
                        result.get(
                            "scenario"
                        ),

                    "Expected":
                        result.get(
                            "expected_decision"
                        ),

                    "Actual":
                        result.get(
                            "actual_decision"
                        ),

                    "Correct":
                        result.get(
                            "decision_correct"
                        ),

                    "Model":
                        result.get(
                            "recommended_model"
                        ),

                    "Risk":
                        result.get(
                            "risk_level"
                        ),

                    "Agent Turns":
                        result.get(
                            "agent_turns"
                        ),
                }
            )

        benchmark_df = pd.DataFrame(
            benchmark_rows
        )

        st.dataframe(
            benchmark_df,
            use_container_width=True,
            hide_index=True,
        )


# =========================================================
# TAB 4 — ARCHITECTURE
# =========================================================

with tab4:

    st.subheader(
        "System Architecture"
    )

    st.code(
        """
                 ┌─────────────────────┐
                 │   Scenario / Data   │
                 └─────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │ Deterministic ML Layer │
              │ sklearn + XGBoost      │
              └───────────┬────────────┘
                          │
                Metrics + Segment Risk
                          │
                          ▼
              ┌────────────────────────┐
              │ AutoGen SelectorGroup  │
              └───────────┬────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
 Data Scientist     Model Evaluator     Risk Agent
        │                 │                 │
        └─────────────────┼─────────────────┘
                          ▼
                        Critic
                          │
                          ▼
                   Final Reviewer
                          │
                          ▼
                Governance Decision
                          │
                          ▼
              ┌────────────────────────┐
              │ Evaluation Framework   │
              │ routing / decisions    │
              └───────────┬────────────┘
                          │
                          ▼
                   Streamlit UI
        """
    )

    st.markdown(
        """
### Core stack

- **AutoGen AgentChat** — multi-agent orchestration
- **SelectorGroupChat** — dynamic speaker selection
- **Groq** — LLM inference
- **GPT-OSS 20B** — agent reasoning model
- **scikit-learn** — deterministic ML evaluation
- **XGBoost** — candidate churn model
- **Pandas** — data processing
- **Streamlit** — application layer
- **Plotly** — interactive charts

### Governance roles

**Data Scientist**  
Selects the strongest candidate model.

**Model Evaluator**  
Challenges metric interpretation and validation quality.

**Risk Agent**  
Looks for segment-level degradation and model risk.

**Critic**  
Challenges unsupported assumptions.

**Final Reviewer**  
Issues the final governance decision.
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    "---"
)

st.caption(
    "ModelCouncil — Multi-Agent ML Governance with AutoGen"
)

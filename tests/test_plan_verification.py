"""
Unit tests for PlannerAgent plan verification, reordering, retry/fallback, and HPT runtime guardrails.
"""

import pytest
import os
import sys
import tempfile

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from agents.planner_agent import (
    TaskItem,
    ExecutionPlan,
    PlanStatus,
    infer_task_type,
    validate_plan_prerequisites,
    reorder_plan_tasks,
    insert_missing_prerequisites_fallback,
    _extract_json,
    extract_response_message,
    extract_message_content,
)


def test_infer_task_type():
    assert infer_task_type("Data Cleaning and Inspection", "Clean raw data", "DataAnalystAgent") == "cleaning"
    assert infer_task_type("Feature Engineering", "Construct ratios", "DataAnalystAgent") == "feature_engineering"
    assert infer_task_type("Descriptive Statistics", "Plot churn charts", "DataAnalystAgent") == "stats_and_viz"
    assert infer_task_type("Model Training & Comparison", "Train 3 candidate models", "MLAgent") == "model_comparison"
    assert infer_task_type("Hyperparameter Tuning", "Optimize best model with GridSearchCV", "MLAgent") == "hyperparameter_tuning"
    assert infer_task_type("Executive Report", "Write final markdown", "ReporterAgent") == "report"


def test_valid_plan_prerequisites():
    tasks = [
        TaskItem(task_id=1, title="Data Cleaning", task_type="cleaning", assigned_agent="DataAnalystAgent"),
        TaskItem(task_id=2, title="Feature Engineering", task_type="feature_engineering", assigned_agent="DataAnalystAgent", depends_on=[1]),
        TaskItem(task_id=3, title="Descriptive Statistics", task_type="stats_and_viz", assigned_agent="DataAnalystAgent", depends_on=[2]),
        TaskItem(task_id=4, title="Model Training", task_type="model_comparison", assigned_agent="MLAgent", depends_on=[2]),
        TaskItem(task_id=5, title="Hyperparameter Tuning", task_type="hyperparameter_tuning", assigned_agent="MLAgent", depends_on=[4]),
        TaskItem(task_id=6, title="Executive Report", task_type="report", assigned_agent="ReporterAgent", depends_on=[1, 2, 3, 4, 5]),
    ]
    plan = ExecutionPlan(
        plan_id="plan_valid",
        query="Complete pipeline",
        selected_agents=["DataAnalystAgent", "MLAgent", "ReporterAgent"],
        tasks=tasks
    )

    is_valid, errors, details = validate_plan_prerequisites(plan)
    assert is_valid is True
    assert len(errors) == 0
    assert details["needs_reorder"] is False
    assert details["missing_cleaning"] is False
    assert details["missing_model_comparison"] is False


def test_missing_cleaning_for_feature_engineering():
    tasks = [
        TaskItem(task_id=1, title="Feature Engineering", task_type="feature_engineering", assigned_agent="DataAnalystAgent"),
        TaskItem(task_id=2, title="Report", task_type="report", assigned_agent="ReporterAgent", depends_on=[1]),
    ]
    plan = ExecutionPlan(
        plan_id="plan_fe_no_clean",
        query="Engineer features without cleaning",
        selected_agents=["DataAnalystAgent", "ReporterAgent"],
        tasks=tasks
    )

    is_valid, errors, details = validate_plan_prerequisites(plan)
    assert is_valid is False
    assert len(errors) >= 1
    assert details["missing_cleaning"] is True
    assert "cleaned_dataset.csv" in errors[0]


def test_missing_model_for_hyperparameter_tuning():
    tasks = [
        TaskItem(task_id=1, title="Data Cleaning", task_type="cleaning", assigned_agent="DataAnalystAgent"),
        TaskItem(task_id=2, title="Hyperparameter Tuning", task_type="hyperparameter_tuning", assigned_agent="MLAgent", depends_on=[1]),
        TaskItem(task_id=3, title="Report", task_type="report", assigned_agent="ReporterAgent", depends_on=[2]),
    ]
    plan = ExecutionPlan(
        plan_id="plan_hpt_no_model",
        query="Tune model without prior comparison",
        selected_agents=["DataAnalystAgent", "MLAgent", "ReporterAgent"],
        tasks=tasks
    )

    is_valid, errors, details = validate_plan_prerequisites(plan)
    assert is_valid is False
    assert len(errors) >= 1
    assert details["missing_model_comparison"] is True
    assert "model_comparison.csv" in errors[0]


def test_reorder_when_prerequisites_placed_after():
    tasks = [
        TaskItem(task_id=1, title="Feature Engineering", task_type="feature_engineering", assigned_agent="DataAnalystAgent"),
        TaskItem(task_id=2, title="Hyperparameter Tuning", task_type="hyperparameter_tuning", assigned_agent="MLAgent"),
        TaskItem(task_id=3, title="Data Cleaning", task_type="cleaning", assigned_agent="DataAnalystAgent"),
        TaskItem(task_id=4, title="Model Training", task_type="model_comparison", assigned_agent="MLAgent"),
    ]
    plan = ExecutionPlan(
        plan_id="plan_reversed",
        query="Reversed order pipeline",
        selected_agents=["DataAnalystAgent", "MLAgent"],
        tasks=tasks
    )

    is_valid, errors, details = validate_plan_prerequisites(plan)
    assert is_valid is True  # Both cleaning and model exist, so no fatal missing prerequisite
    assert details["needs_reorder"] is True

    reordered = reorder_plan_tasks(plan)
    types_in_order = [t.task_type for t in reordered.tasks]
    assert types_in_order == ["cleaning", "feature_engineering", "model_comparison", "hyperparameter_tuning"]
    assert [t.task_id for t in reordered.tasks] == [1, 2, 3, 4]
    # Check FE depends on cleaning
    fe_task = reordered.tasks[1]
    assert 1 in fe_task.depends_on
    assert fe_task.inputs == ["cleaned_dataset.csv"]
    # Check HPT depends on model comparison
    hpt_task = reordered.tasks[3]
    assert 3 in hpt_task.depends_on
    assert "model_comparison.csv" in hpt_task.inputs


def test_fallback_insertion_after_retry_exceeded():
    tasks = [
        TaskItem(task_id=1, title="Feature Engineering", task_type="feature_engineering", assigned_agent="DataAnalystAgent"),
        TaskItem(task_id=2, title="Hyperparameter Tuning", task_type="hyperparameter_tuning", assigned_agent="MLAgent"),
    ]
    plan = ExecutionPlan(
        plan_id="plan_fallback",
        query="Fallback insertion test",
        selected_agents=["DataAnalystAgent", "MLAgent"],
        tasks=tasks
    )

    is_valid, errors, details = validate_plan_prerequisites(plan)
    assert is_valid is False
    assert details["missing_cleaning"] is True
    assert details["missing_model_comparison"] is True

    repaired_plan = insert_missing_prerequisites_fallback(plan, details)
    types = [t.task_type for t in repaired_plan.tasks]
    assert "cleaning" in types
    assert "model_comparison" in types
    assert types.index("cleaning") < types.index("feature_engineering")
    assert types.index("model_comparison") < types.index("hyperparameter_tuning")
    assert "Fallback insertion applied" in repaired_plan.notes


def test_orchestrator_skips_hpt_when_artifacts_missing(monkeypatch):
    from agents.orchestrate import WorkflowOrchestrator

    # Create a plan with an HPT task
    hpt_task = TaskItem(
        task_id=1,
        title="Hyperparameter Tuning",
        task_type="hyperparameter_tuning",
        assigned_agent="MLAgent",
        description="Tune Random Forest hyperparameters"
    )
    plan = ExecutionPlan(
        plan_id="plan_hpt_skip",
        query="Tune model",
        selected_agents=["MLAgent"],
        tasks=[hpt_task]
    )

    orchestrator = WorkflowOrchestrator()
    # Mock reporter agent to avoid remote LLM calls
    monkeypatch.setattr(orchestrator.reporter_agent, "generate_report", lambda **kwargs: "# Final Report\nHPT skipped.")

    with tempfile.TemporaryDirectory() as tmpdir:
        orchestrator.current_output_dir = tmpdir
        orchestrator.generated_artifacts = []

        # Execute plan
        res = orchestrator.execute_plan(plan=plan, user_query="Tune model")

        # Verify HPT task was skipped with warning
        assert hpt_task.status == "skipped"
        assert "Skipped:" in str(hpt_task.result)
        assert any(r.get("status") == "skipped" for r in orchestrator.completed_task_results)


def test_priority_and_dependency_combined():
    """Verify that tasks strictly respect depends_on and use priority for tie-breaking."""
    tasks = [
        TaskItem(task_id=1, title="Data Cleaning", task_type="cleaning", assigned_agent="DataAnalystAgent"),
        TaskItem(task_id=2, title="Feature Engineering", task_type="feature_engineering", assigned_agent="DataAnalystAgent", depends_on=[1]),
        TaskItem(task_id=3, title="Descriptive Statistics", task_type="stats_and_viz", assigned_agent="DataAnalystAgent", depends_on=[2]),
        TaskItem(task_id=4, title="Unsupervised Analysis", task_type="unsupervised_analysis", assigned_agent="MLAgent", depends_on=[2]),
        TaskItem(task_id=5, title="Model Training", task_type="model_comparison", assigned_agent="MLAgent", depends_on=[2]),
        TaskItem(task_id=6, title="Hyperparameter Tuning", task_type="hyperparameter_tuning", assigned_agent="MLAgent", depends_on=[5]),
        TaskItem(task_id=7, title="Validation Threshold Selection", task_type="model_evaluation", assigned_agent="MLAgent", depends_on=[6]),
        TaskItem(task_id=8, title="Executive Report", task_type="report", assigned_agent="ReporterAgent", depends_on=[1, 2, 3, 4, 5, 6, 7]),
    ]
    plan = ExecutionPlan(
        plan_id="plan_full_pipeline",
        query="End-to-end pipeline with unsupervised learning and threshold evaluation",
        selected_agents=["DataAnalystAgent", "MLAgent", "ReporterAgent"],
        tasks=tasks
    )

    reordered = reorder_plan_tasks(plan)
    titles = [t.title for t in reordered.tasks]
    assert titles == [
        "Data Cleaning",
        "Feature Engineering",
        "Descriptive Statistics",
        "Unsupervised Analysis",
        "Model Training",
        "Hyperparameter Tuning",
        "Validation Threshold Selection",
        "Executive Report",
    ]
    assert [t.task_id for t in reordered.tasks] == [1, 2, 3, 4, 5, 6, 7, 8]
    # Task 7 (Validation Threshold Selection) must depend on Task 6 (Hyperparameter Tuning)
    assert 6 in reordered.tasks[6].depends_on


def test_user_reported_fraud_pipeline_no_order_mismatch():
    """Verify that the user's exact 7-task plan never swaps Task 5 and Task 6."""
    tasks = [
        TaskItem(task_id=1, title="Data Inspection & Cleaning", task_type="cleaning", assigned_agent="DataAnalystAgent", depends_on=[]),
        TaskItem(task_id=2, title="Feature Engineering", task_type="feature_engineering", assigned_agent="DataAnalystAgent", depends_on=[1]),
        TaskItem(task_id=3, title="EDA & Class Imbalance Analysis", task_type="stats_and_viz", assigned_agent="DataAnalystAgent", depends_on=[2]),
        TaskItem(task_id=4, title="Model Training & Comparison", task_type="model_comparison", assigned_agent="MLAgent", depends_on=[2]),
        TaskItem(task_id=5, title="Hyperparameter Tuning", task_type="hyperparameter_tuning", assigned_agent="MLAgent", depends_on=[4]),
        # Even if task_type was given as model_comparison by LLM, title indicates threshold evaluation and depends on 5
        TaskItem(task_id=6, title="Validation Threshold Selection & Test Evaluation", task_type="model_comparison", assigned_agent="MLAgent", depends_on=[5]),
        TaskItem(task_id=7, title="Executive Report & Verification", task_type="report", assigned_agent="ReporterAgent", depends_on=[1, 2, 3, 4, 5, 6]),
    ]
    plan = ExecutionPlan(
        plan_id="plan_fraud_01",
        query="Supervised fraud detection on creditcard.csv",
        selected_agents=["DataAnalystAgent", "MLAgent", "ReporterAgent"],
        tasks=tasks
    )

    reordered = reorder_plan_tasks(plan)
    reordered_titles = [t.title for t in reordered.tasks]

    assert reordered_titles[3] == "Model Training & Comparison"
    assert reordered_titles[4] == "Hyperparameter Tuning"
    assert reordered_titles[5] == "Validation Threshold Selection & Test Evaluation"
    assert reordered_titles[6] == "Executive Report & Verification"
    assert [t.task_id for t in reordered.tasks] == [1, 2, 3, 4, 5, 6, 7]
    assert 5 in reordered.tasks[5].depends_on


def test_exact_user_credit_card_fraud_pipeline_ordering():
    """Verify that the user's full realistic credit card fraud pipeline maintains strict procedure."""
    plan_dict = {
        "plan_id": "plan_ccfd_001",
        "user_goal": "Complete supervised binary classification investigation to detect fraudulent credit card transactions (Class=1) with rigorous leakage prevention, class-imbalance handling, validation-based threshold selection, and held-out test evaluation.",
        "selected_agents": ["DataAnalystAgent", "MLAgent", "ReporterAgent"],
        "status": "draft",
        "notes": "SCOPE CLASSIFICATION...",
        "tasks": [
            {
                "task_id": 1,
                "title": "Data Inspection & Cleaning",
                "assigned_agent": "DataAnalystAgent",
                "task_type": "cleaning",
                "description": "Inspect creditcard.csv: record row/column counts, dtypes, target distribution (Class 0/1), imbalance ratio, missing values, duplicate rows, invalid values (logical bounds on Time>=0, Amount>=0), summary stats for Time/Amount/V1-V28. Investigate duplicates (exact row duplicates) and distributional anomalies without deleting legitimate outliers. Preserve all columns. Export cleaned_dataset.csv and cleaning_summary.json with transformation log and row counts.",
                "depends_on": [],
                "inputs": ["creditcard.csv"],
                "expected_output": "cleaned_dataset.csv, cleaning_summary.json",
                "est_tool_calls": 5
            },
            {
                "task_id": 2,
                "title": "Feature Engineering",
                "assigned_agent": "DataAnalystAgent",
                "task_type": "feature_engineering",
                "description": "From cleaned_dataset.csv, construct ONLY these justified derived features: (1) Amount_log = log1p(Amount) to handle extreme skew; (2) Time_hour = (Time / 3600) % 24 for hour-of-day cyclical pattern; (3) Time_day = (Time / 86400) % 7 for day-of-week pattern. Do NOT apply PCA to V1-V28 (already PCA-transformed). Preserve all original columns plus new features. Export feature_engineered_dataset.csv and feature_engineering_summary.json with feature list, transformations, and rationale.",
                "depends_on": [1],
                "inputs": ["cleaned_dataset.csv"],
                "expected_output": "feature_engineered_dataset.csv, feature_engineering_summary.json",
                "est_tool_calls": 4
            },
            {
                "task_id": 3,
                "title": "Descriptive Statistics & Visualization",
                "assigned_agent": "DataAnalystAgent",
                "task_type": "stats_and_viz",
                "description": "From feature_engineered_dataset.csv, compute and visualize EXACTLY these 5 charts (bounded for budget): (1) Class balance bar chart (count + %); (2) Amount_log distribution by Class (overlay histograms); (3) Fraud rate by Time_hour (line plot); (4) Correlation heatmap of V1-V10 + Amount_log + Class (sampled 10k rows for readability); (5) Precision-Recall curve placeholder (empty axes with labels, to be populated by MLAgent later). Compute summary stats: class proportions, Amount_log stats by class, Time_hour fraud rates. Export analyst_summary.json and the 5 PNG files.",
                "depends_on": [2],
                "inputs": ["feature_engineered_dataset.csv"],
                "expected_output": "analyst_summary.json, class_balance.png, amount_log_by_class.png, fraud_rate_by_hour.png, correlation_heatmap.png, pr_curve_placeholder.png",
                "est_tool_calls": 7
            },
            {
                "task_id": 4,
                "title": "Model Training & Comparison",
                "assigned_agent": "MLAgent",
                "task_type": "model_comparison",
                "description": "From feature_engineered_dataset.csv, perform leakage-safe chronological split: earliest 60% train, next 20% validation, latest 20% test (verify class counts in each). Fit StandardScaler on train only (exclude Time, Time_hour, Time_day from scaling if they are not Gaussian; scale Amount_log and V1-V28). Train and compare 4 models with class_weight='balanced' where supported: (1) DummyClassifier(strategy='most_frequent') baseline; (2) LogisticRegression(max_iter=1000, class_weight='balanced'); (3) RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1); (4) GradientBoostingClassifier(n_estimators=200) or HistGradientBoostingClassifier if available. Use 5-fold stratified CV on training partition for each. Evaluate on validation set: precision, recall, F1, PR-AUC, ROC-AUC, balanced accuracy, confusion matrix, fraud-class support at default 0.5 threshold. Select champion by validation recall (primary_metric). Export ALL core artifacts: model_comparison.csv, baseline_score.json, feature_importance.csv (coefficients for LR, impurity for RF/GB), confusion_matrix.json, predictions.csv (val set with proba/pred), ml_summary.json, model_registry.json, champion_model.json, and versioned .pkl files (e.g., logistic_regression_p0.pkl).",
                "depends_on": [2],
                "inputs": ["feature_engineered_dataset.csv"],
                "expected_output": "model_comparison.csv, baseline_score.json, feature_importance.csv, confusion_matrix.json, predictions.csv, ml_summary.json, model_registry.json, champion_model.json, *_p0.pkl",
                "est_tool_calls": 9,
                "primary_metric": "recall"
            },
            {
                "task_id": 5,
                "title": "Hyperparameter Tuning",
                "assigned_agent": "MLAgent",
                "task_type": "hyperparameter_tuning",
                "description": "Using champion model from task 4 (per champion_model.json), run RandomizedSearchCV (n_iter=20, cv=3, scoring=recall, random_state=42) on training partition ONLY with the following BOUNDED grid (product >= n_iter): For GradientBoosting/HistGradientBoosting: {learning_rate: [0.01, 0.05, 0.1, 0.2], max_depth: [3, 5, 7, None], n_estimators: [100, 200, 300, 400], min_samples_leaf: [1, 5, 10], subsample: [0.8, 1.0]} (4*4*4*3*2=384 combos). For RandomForest: {n_estimators: [100, 200, 300, 400, 500], max_depth: [5, 10, 15, None], min_samples_leaf: [1, 2, 5], max_features: ['sqrt', 'log2']} (5*4*3*2=120 combos). For LogisticRegression: {C: [0.01, 0.1, 1, 10, 100], penalty: ['l2'], solver: ['lbfgs']} (5 combos - use GridSearchCV instead). Refit best on full train, evaluate on validation, compare tuned vs baseline metrics. Export tuned_model_comparison.csv, best_params.json, hyperparameter_tuning_summary.json, and update champion_model.json ONLY if tuned model beats baseline on validation recall.",
                "depends_on": [4],
                "inputs": ["model_comparison.csv", "champion_model.json", "model_registry.json"],
                "expected_output": "tuned_model_comparison.csv, best_params.json, hyperparameter_tuning_summary.json, champion_model.json (updated if tuned wins)",
                "est_tool_calls": 7,
                "primary_metric": "recall"
            },
            {
                "task_id": 6,
                "title": "Validation Threshold Selection & Final Test Evaluation",
                "assigned_agent": "MLAgent",
                "task_type": "model_comparison",
                "description": "Using the final champion model (from task 4 or 5), perform validation-only threshold selection: compute precision/recall/F1 across thresholds 0.01-0.99 on validation set, select threshold maximizing F1 (or meeting min precision=0.5 if specified). Record validation_threshold_analysis.csv. Then evaluate ONCE on held-out test set at both default 0.5 and selected threshold. Export test_metrics.json (all metrics at both thresholds), test_predictions.csv (true label, predicted proba, predicted class at selected threshold), confusion matrix and PR curve visualizations for test set. Verify no test data used in any prior step.",
                "depends_on": [4, 5],
                "inputs": ["feature_engineered_dataset.csv", "champion_model.json", "model_registry.json"],
                "expected_output": "validation_threshold_analysis.csv, test_metrics.json, test_predictions.csv, test_confusion_matrix.png, test_pr_curve.png",
                "est_tool_calls": 6
            },
            {
                "task_id": 7,
                "title": "Executive Report Synthesis",
                "assigned_agent": "ReporterAgent",
                "task_type": "report",
                "description": "Synthesize all verified artifacts into final_report.md covering: dataset characteristics/quality, class imbalance implications, cleaning/feature-engineering decisions, chronological splitting strategy and leakage safeguards, baseline and model-comparison results (validation), validation-based threshold selection, final held-out test performance (both thresholds), false-positive/false-negative implications, limitations/deployment risks, reproducibility instructions and artifact manifest. Distinguish validation vs test results. Do not claim production readiness. Include artifact manifest listing all generated files and purposes.",
                "depends_on": [1, 2, 3, 4, 5, 6],
                "inputs": ["cleaning_summary.json", "feature_engineering_summary.json", "analyst_summary.json", "ml_summary.json", "hyperparameter_tuning_summary.json", "validation_threshold_analysis.csv", "test_metrics.json", "test_predictions.csv"],
                "expected_output": "final_report.md, artifact_manifest.json",
                "est_tool_calls": 3
            }
        ]
    }

    plan = ExecutionPlan.from_dict(plan_dict)

    # 1. Validation check
    is_valid, errors, details = validate_plan_prerequisites(plan)
    assert is_valid is True
    assert len(errors) == 0
    assert details["needs_reorder"] is False

    # 2. Check task_type resolution before reordering
    assert plan.tasks[3].task_type == "model_comparison"
    assert plan.tasks[4].task_type == "hyperparameter_tuning"
    assert plan.tasks[5].task_type == "model_evaluation"

    # 3. Check reordering strictly preserves procedural sequence:
    reordered = reorder_plan_tasks(plan)
    reordered_titles = [t.title for t in reordered.tasks]

    assert reordered_titles == [
        "Data Inspection & Cleaning",
        "Feature Engineering",
        "Descriptive Statistics & Visualization",
        "Model Training & Comparison",
        "Hyperparameter Tuning",
        "Validation Threshold Selection & Final Test Evaluation",
        "Executive Report Synthesis",
    ]
    assert [t.task_id for t in reordered.tasks] == [1, 2, 3, 4, 5, 6, 7]

    # Task 4 (Model Training) must strictly precede Task 5 (Hyperparameter Tuning)
    model_task = reordered.tasks[3]
    hpt_task = reordered.tasks[4]
    eval_task = reordered.tasks[5]
    report_task = reordered.tasks[6]

    assert model_task.task_id == 4
    assert hpt_task.task_id == 5
    assert eval_task.task_id == 6
    assert report_task.task_id == 7

    # Hyperparameter Tuning must depend on Model Training
    assert 4 in hpt_task.depends_on
    # Validation Threshold Selection must depend on HPT (and Model Training)
    assert 5 in eval_task.depends_on


def test_extract_response_message_safe():
    """Verify extract_response_message safely handles dict envelopes, objects, and empty/None choices."""
    class DummyChoice:
        def __init__(self, message):
            self.message = message

    class DummyResponse:
        def __init__(self, choices):
            self.choices = choices

    class DummyMessage:
        def __init__(self, content):
            self.content = content

    # 1. Normal object response
    msg_obj = DummyMessage("hello")
    resp_obj = DummyResponse([DummyChoice(msg_obj)])
    extracted = extract_response_message(resp_obj)
    assert extracted == msg_obj
    assert extract_message_content(extracted) == "hello"

    # 2. Dict response
    dict_resp = {
        "choices": [
            {"message": {"content": "world"}}
        ]
    }
    extracted_dict = extract_response_message(dict_resp)
    assert extracted_dict == {"content": "world"}
    assert extract_message_content(extracted_dict) == "world"

    # 3. None or missing choices raises ValueError instead of subscripting None
    with pytest.raises(ValueError):
        extract_response_message(None)

    with pytest.raises(ValueError):
        extract_response_message(DummyResponse(None))

    with pytest.raises(ValueError):
        extract_response_message(DummyResponse([]))

    with pytest.raises(ValueError):
        extract_response_message({"choices": None})

    with pytest.raises(ValueError):
        extract_response_message({"choices": []})

    with pytest.raises(RuntimeError):
        extract_response_message({"error": {"message": "Service temporarily overloaded", "code": 503}})


def test_extract_json_null_and_malformed_cases():
    """Verify _extract_json returns a dict even for None, non-dict, or raw JSON lists."""
    assert _extract_json("") == {}
    assert _extract_json(None) == {}
    assert _extract_json("not json") == {}
    assert _extract_json("null") == {}

    # JSON array should be wrapped in {"tasks": [...]}
    res_list = _extract_json('[{"task_id": 1, "title": "Inspection"}]')
    assert isinstance(res_list, dict)
    assert "tasks" in res_list
    assert len(res_list["tasks"]) == 1

    # Standard JSON object
    res_obj = _extract_json('{"notes": "test", "tasks": []}')
    assert res_obj == {"notes": "test", "tasks": []}


def test_planner_null_selected_agents_and_malformed_tasks():
    """Verify that ExecutionPlan and TaskItem handle null fields gracefully without subscript errors."""
    raw_dict = {
        "selected_agents": None,
        "notes": None,
        "tasks": [
            {
                "task_id": 1,
                "title": "Clean Data",
                "assigned_agent": None,
                "task_type": None,
                "depends_on": None,
                "inputs": None,
            },
            {
                "task_id": 2,
                "title": None,
                "assigned_agent": "MLAgent",
                "task_type": "model_comparison",
                "depends_on": [1],
                "inputs": None,
            }
        ]
    }

    # Verify task parsing from raw dict with nulls
    tasks = []
    default_agent = "DataAnalystAgent"
    for idx, task_raw in enumerate(raw_dict.get("tasks") or [], start=1):
        assigned = task_raw.get("assigned_agent") or default_agent
        title = str(task_raw.get("title") or f"Task {idx}").strip()
        task = TaskItem(
            task_id=task_raw.get("task_id", idx),
            title=title,
            assigned_agent=assigned,
            task_type=task_raw.get("task_type") or infer_task_type(title, "", assigned),
            depends_on=task_raw.get("depends_on") or [],
            inputs=task_raw.get("inputs") or [],
        )
        tasks.append(task)

    plan = ExecutionPlan(
        plan_id="plan_null_safe",
        query="Test query with nulls",
        selected_agents=["DataAnalystAgent", "MLAgent"],
        tasks=tasks,
    )

    # validate_plan_prerequisites must not crash on None titles or null lists
    is_valid, errors, details = validate_plan_prerequisites(plan)
    assert isinstance(is_valid, bool)

    # reorder_plan_tasks must not crash on None titles or null lists
    reordered = reorder_plan_tasks(plan)
    assert len(reordered.tasks) == 2
    assert reordered.tasks[0].task_id == 1
    assert reordered.tasks[1].task_id == 2


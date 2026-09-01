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



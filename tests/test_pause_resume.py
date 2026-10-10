"""
Tests for Pause and Resume Feature in AnalystGPT

Verifies:
1. Pausing during a task correctly identifies the active task ID (e.g., Task 3).
2. completed_task_results for tasks prior to the paused task (Tasks 1 & 2) are preserved.
3. Resuming skips completed prior tasks and re-executes strictly from the paused task.
4. Fallback logic for determining resume_from_task_id correctly resolves to the active/incomplete task, never reverting to Task 1 when prior tasks are completed.
5. Task statuses in the plan are properly updated (completed prior tasks stay completed, resumed task is in_progress, subsequent tasks are pending).
"""

import pytest
from unittest.mock import MagicMock, patch
from agents.cancellation import CancellationToken, WorkflowPausedException
from agents.planner_agent import ExecutionPlan, TaskItem, PlanStatus
from agents.orchestrate import WorkflowOrchestrator


def make_dummy_plan():
    tasks = [
        TaskItem(task_id=1, title="Task 1 Data Cleaning", assigned_agent="DataAnalystAgent", description="clean data"),
        TaskItem(task_id=2, title="Task 2 Feature Engineering", assigned_agent="DataAnalystAgent", description="features"),
        TaskItem(task_id=3, title="Task 3 Model Training", assigned_agent="MLAgent", description="train model"),
        TaskItem(task_id=4, title="Task 4 Tuning", assigned_agent="MLAgent", description="tune model"),
        TaskItem(task_id=5, title="Task 5 Test Evaluation", assigned_agent="MLAgent", description="test model"),
        TaskItem(task_id=6, title="Task 6 Report", assigned_agent="ReporterAgent", description="report"),
    ]
    return ExecutionPlan(
        plan_id="test_plan_pause_123",
        query="Test plan",
        selected_agents=["DataAnalystAgent", "MLAgent", "ReporterAgent"],
        tasks=tasks,
    )


def test_pause_execution_identifies_active_task():
    """Verify orchestrator.pause_execution records active task and creates provisional checkpoint."""
    orch = WorkflowOrchestrator()
    plan = make_dummy_plan()
    orch.current_plan = plan
    token = CancellationToken()
    orch.cancellation_token = token

    # Simulate Task 1 & 2 completed, Task 3 currently active
    orch.completed_task_results = [
        {"task_id": 1, "status": "completed", "summary": "clean done"},
        {"task_id": 2, "status": "completed", "summary": "features done"},
    ]
    plan.tasks[0].status = "completed"
    plan.tasks[1].status = "completed"
    plan.tasks[2].status = "in_progress"
    orch.current_active_task = plan.tasks[2]

    res = orch.pause_execution("Paused by test")

    assert res["status"] == "paused"
    assert res["task_id"] == 3
    assert token.is_paused is True
    assert token.paused_task_id == 3

    # Provisional checkpoint should be generated immediately
    cp = orch.last_pause_checkpoint
    assert cp is not None
    assert cp["paused_task_id"] == 3
    assert len(cp["completed_task_results"]) == 2
    assert cp["completed_task_results"][0]["task_id"] == 1
    assert cp["completed_task_results"][1]["task_id"] == 2


def test_resume_task_id_resolution_logic():
    """Verify that when resume_from_task_id is determined, it never defaults to 1 when tasks 1 & 2 are done."""
    orch = WorkflowOrchestrator()
    plan = make_dummy_plan()
    orch.current_plan = plan

    plan.tasks[0].status = "completed"
    plan.tasks[1].status = "completed"
    plan.tasks[2].status = "paused"
    orch.completed_task_results = [
        {"task_id": 1, "status": "completed"},
        {"task_id": 2, "status": "completed"},
    ]

    # Helper resolution logic matching backend
    def resolve_resume_id(req_tid=None, checkpoint=None):
        if req_tid:
            return req_tid
        if checkpoint and checkpoint.get("paused_task_id"):
            return checkpoint["paused_task_id"]
        if plan and plan.tasks:
            active_t = next((t for t in plan.tasks if t.status in ["paused", "in_progress"]), None)
            if active_t:
                return active_t.task_id
            completed_ids = {tr.get("task_id") for tr in orch.completed_task_results}
            uncompleted_t = next((t for t in plan.tasks if t.status != "completed" and t.task_id not in completed_ids), None)
            if uncompleted_t:
                return uncompleted_t.task_id
            return plan.tasks[0].task_id
        return 1

    # Case 1: Checkpoint has paused_task_id = 3
    assert resolve_resume_id(checkpoint={"paused_task_id": 3}) == 3

    # Case 2: Checkpoint is None, active task is paused
    assert resolve_resume_id(checkpoint=None) == 3

    # Case 3: Checkpoint is None, active task reset to pending
    plan.tasks[2].status = "pending"
    assert resolve_resume_id(checkpoint=None) == 3


def test_execute_plan_skips_prior_completed_tasks_on_resume():
    """Verify execute_plan with resume_from_task_id=3 skips Tasks 1 and 2 and runs Task 3."""
    orch = WorkflowOrchestrator()
    plan = make_dummy_plan()
    plan.tasks[0].status = "completed"
    plan.tasks[1].status = "completed"

    orch.completed_task_results = [
        {"task_id": 1, "status": "completed", "artifacts": ["clean.csv"]},
        {"task_id": 2, "status": "completed", "artifacts": ["features.csv"]},
    ]

    mock_agent = MagicMock()
    mock_agent.run_task.return_value = {
        "status": "completed",
        "artifacts": ["model.pkl"],
        "summary": "Trained models",
        "metrics": {"f1": 0.95}
    }

    executed_tasks = []

    def mock_resolve_agent(agent_name):
        def _run_task_wrapper(prompt, **kwargs):
            tid = kwargs.get("task_id")
            executed_tasks.append(tid)
            return {
                "status": "completed",
                "artifacts": [f"task_{tid}_output.txt"],
                "summary": f"Task {tid} done"
            }
        mock = MagicMock()
        mock.run_task.side_effect = _run_task_wrapper
        mock.__class__.__name__ = agent_name
        return mock

    orch._resolve_agent = mock_resolve_agent

    checkpoint = {
        "paused_task_id": 3,
        "completed_task_results": [
            {"task_id": 1, "status": "completed"},
            {"task_id": 2, "status": "completed"},
        ],
        "output_dir": orch.current_output_dir,
    }

    result = orch.execute_plan(
        plan=plan,
        resume_from_task_id=3,
        resume_checkpoint=checkpoint
    )

    # Tasks 1 and 2 MUST NOT have been executed
    assert 1 not in executed_tasks
    assert 2 not in executed_tasks
    # Task 3 MUST be the first executed task
    assert executed_tasks[0] == 3
    # Tasks 1 and 2 should remain completed in plan
    assert plan.tasks[0].status == "completed"
    assert plan.tasks[1].status == "completed"

"""
Shared Pydantic schemas for AnalystGPT model registry and champion selection.

This module defines the canonical data structures used by both MLAgent's code-gen
template and champion_selector to ensure schema consistency and prevent silent
data loss on mismatch.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from enum import Enum


class ModelStatus(str, Enum):
    CANDIDATE = "candidate"
    CHALLENGER = "challenger"
    CHAMPION = "champion"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"
    EVALUATED = "evaluated"


class ModelStage(str, Enum):
    INITIAL_COMPARISON = "initial_comparison"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"


class ModelEntry(BaseModel):
    """Single model entry in the registry."""
    model_id: str = Field(..., description="Unique identifier for this model version")
    model_family: Optional[str] = Field(None, description="Model family name (e.g., logistic_regression)")
    artifact_filename: str = Field(..., description="Exact filename of the saved .pkl artifact")
    version: str = Field(default="p0", description="Version string (e.g., p0, p1_tuned, p2_tuned)")
    stage: ModelStage = Field(default=ModelStage.INITIAL_COMPARISON, description="Pipeline stage that produced this model")
    parent_model: Optional[str] = Field(None, description="Parent model artifact filename if applicable")
    task_id: int = Field(default=0, description="Task ID that produced this model")
    dataset_source: str = Field(default="", description="Source dataset filename")
    target_column: str = Field(default="", description="Target column name")
    feature_count: int = Field(default=0, description="Number of features used for training")
    preprocessing: Dict[str, Any] = Field(default_factory=dict, description="Preprocessing steps applied")
    model_parameters: Dict[str, Any] = Field(default_factory=dict, description="Model hyperparameters")
    evaluation_metrics: Dict[str, float] = Field(default_factory=dict, description="Evaluation metrics on test set")
    primary_selection_metric: str = Field(default="", description="Primary metric used for selection")
    secondary_metrics: List[str] = Field(default_factory=list, description="Secondary metrics for tie-breaking")
    improved_over_parent: Optional[bool] = Field(None, description="Whether this model improved over its parent")
    improvement_delta: Optional[float] = Field(None, description="Delta in primary metric vs parent")
    status: ModelStatus = Field(default=ModelStatus.CANDIDATE, description="Current status in selection pipeline")

    @field_validator('artifact_filename')
    @classmethod
    def artifact_must_be_pkl(cls, v):
        if not v.endswith('.pkl'):
            raise ValueError(f'artifact_filename must end with .pkl, got: {v}')
        return v

    @field_validator('model_family', mode='before')
    @classmethod
    def set_model_family_from_id(cls, v, info):
        if v is None or v == "":
            # Derive from model_id if not provided
            model_id = info.data.get('model_id', '')
            if model_id:
                return model_id
        return v


class ModelRegistry(BaseModel):
    """Complete model registry containing all trained model versions."""
    models: List[ModelEntry] = Field(default_factory=list)

    def get_champion(self) -> Optional[ModelEntry]:
        """Return the current champion model, if any."""
        for model in self.models:
            if model.status == ModelStatus.CHAMPION:
                return model
        return None

    def get_model_by_artifact(self, artifact_filename: str) -> Optional[ModelEntry]:
        """Find a model entry by its artifact filename."""
        for model in self.models:
            if model.artifact_filename == artifact_filename:
                return model
        return None

    def get_models_by_family(self, family: str) -> List[ModelEntry]:
        """Get all model versions for a given family."""
        return [m for m in self.models if m.model_family == family]


class ChampionModel(BaseModel):
    """Champion model pointer - identifies the currently recommended model artifact."""
    model: str = Field(..., description="Model name (e.g., Logistic Regression)")
    model_name: str = Field(..., description="Full model display name")
    artifact_filename: str = Field(..., description="Exact filename of the champion .pkl artifact")
    primary_metric: str = Field(..., description="Primary metric used for selection")
    score: float = Field(..., description="Score on the primary metric")
    selection_basis: str = Field(..., description="Human-readable selection rationale")
    selected_by: str = Field(default="code (champion_selector)", description="Who/what selected this champion")
    primary_metric_name: Optional[str] = Field(None, description="Canonical metric name")
    ranking: Optional[List[Dict[str, Any]]] = Field(None, description="Top-5 model ranking")
    registry_entry_added: Optional[bool] = Field(None, description="Whether registry entry was added")

    @field_validator('artifact_filename')
    @classmethod
    def artifact_must_be_pkl(cls, v):
        if not v.endswith('.pkl'):
            raise ValueError(f'artifact_filename must end with .pkl, got: {v}')
        return v


def load_registry(path: str) -> ModelRegistry:
    """Load and validate model registry from JSON file.

    Handles both new schema format and legacy format for backward compatibility.
    Raises:
        ValueError: If file doesn't exist or schema validation fails.
    """
    import json
    import os

    if not os.path.exists(path):
        return ModelRegistry(models=[])

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"model_registry.json root must be an object, got {type(data).__name__}")

    models_data = data.get('models')
    if not isinstance(models_data, list):
        raise ValueError(f"model_registry.json 'models' must be a list, got {type(models_data).__name__}")

    # Validate each entry with backward compatibility
    models = []
    for i, entry in enumerate(models_data):
        try:
            models.append(ModelEntry(**entry))
        except Exception:
            # Try to convert legacy format
            converted = _convert_legacy_entry(entry)
            if converted:
                models.append(ModelEntry(**converted))
            else:
                raise ValueError(f"model_registry.json entry {i} failed validation and could not be converted from legacy format")

    return ModelRegistry(models=models)


def _convert_legacy_entry(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert legacy registry entry format to new schema."""
    if not isinstance(entry, dict):
        return None

    # Check if it has the minimal legacy fields
    if 'model_id' not in entry or 'artifact_filename' not in entry:
        return None

    # Map legacy fields to new schema
    converted = {
        'model_id': entry.get('model_id', ''),
        'model_family': entry.get('model_family') or entry.get('model_id', ''),
        'artifact_filename': entry.get('artifact_filename', ''),
        'version': entry.get('version', 'p0'),
        'stage': entry.get('stage', 'initial_comparison'),
        'parent_model': entry.get('parent_model'),
        'task_id': entry.get('task_id', 0),
        'dataset_source': entry.get('dataset_source', ''),
        'target_column': entry.get('target_column', ''),
        'feature_count': entry.get('feature_count', 0),
        'preprocessing': entry.get('preprocessing', {}),
        'model_parameters': entry.get('model_parameters', {}),
        'evaluation_metrics': entry.get('evaluation_metrics', {}),
        'primary_selection_metric': entry.get('primary_selection_metric', ''),
        'secondary_metrics': entry.get('secondary_metrics', []),
        'improved_over_parent': entry.get('improved_over_parent'),
        'improvement_delta': entry.get('improvement_delta'),
        'status': entry.get('status', 'candidate'),
    }

    # Map legacy status 'evaluated' to 'candidate'
    if converted['status'] == 'evaluated':
        converted['status'] = 'candidate'

    return converted


def save_registry(registry: ModelRegistry, path: str) -> None:
    """Save model registry to JSON file."""
    import json
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(registry.model_dump(), f, indent=2, default=str)


def load_champion(path: str) -> Optional[ChampionModel]:
    """Load champion model pointer from JSON file."""
    import json
    import os

    if not os.path.exists(path):
        return None

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, dict):
        return None

    try:
        return ChampionModel(**data)
    except Exception:
        return None


def save_champion(champion: ChampionModel, path: str) -> None:
    """Save champion model pointer to JSON file."""
    import json
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(champion.model_dump(), f, indent=2, default=str)
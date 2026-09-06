import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.config import MAX_COMMITS

RepoStatus = Literal["pending", "processing", "ready", "error"]
JobStatus = Literal[
    "queued",
    "cloning",
    "analyzing",
    "building_graph",
    "computing_bus_factor",
    "ready",
    "error",
    "cancelled",
]
PromptType = Literal["explain_drop", "predict_merge"]
RiskLevel = Literal["low", "medium", "high", "critical"]
GraphEdgeType = Literal["import", "co_change"]
HealthColor = Literal["green", "yellow", "orange", "red"]


class ApiError(BaseModel):
    detail: str
    code: str | None = None
    message: str | None = None


class IngestRequest(BaseModel):
    repo_url: str = Field(..., min_length=3, max_length=300)
    branch: str | None = None
    max_commits: int = Field(default=MAX_COMMITS, ge=1, le=MAX_COMMITS)
    exclude_merges: bool = False

    @field_validator("repo_url")
    @classmethod
    def validate_repo_url(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Repository URL is required")
        if value.startswith("https://github.com/") or value.startswith("http://github.com/"):
            normalized = value.replace("http://github.com/", "https://github.com/")
            path = normalized.removeprefix("https://github.com/")
            while path.endswith("/") or path.endswith(".git"):
                path = path[:-1] if path.endswith("/") else path[:-4]
            parts = path.split("/")
            name_pattern = re.compile(r"^[\w.-]+$")
            if (
                len(parts) == 2
                and all(part and part not in {".", ".."} for part in parts)
                and all(name_pattern.fullmatch(part) for part in parts)
            ):
                return normalized
            raise ValueError("Must be a GitHub URL or owner/repo path")
        if "/" in value and not value.startswith("http"):
            owner, repo = value.split("/", 1)
            name_pattern = re.compile(r"^[\w.-]+$")
            if (
                owner
                and repo
                and owner not in {".", ".."}
                and repo not in {".", ".."}
                and name_pattern.fullmatch(owner)
                and name_pattern.fullmatch(repo)
            ):
                return f"https://github.com/{owner}/{repo}"
        raise ValueError("Must be a GitHub URL or owner/repo path")


class IngestResponse(BaseModel):
    repo_id: int
    repo_slug: str
    status: str
    job_id: int
    message: str


class RescanResponse(BaseModel):
    repo_id: int
    repo_slug: str
    status: str
    job_id: int
    message: str
    new_commits_found: int = 0


class RepoOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    url: str
    name: str
    owner: str
    repo_slug: str
    default_branch: str
    ingested_at: datetime | None
    last_updated_at: datetime | None
    last_job_completed_at: datetime | None = None
    total_commits: int
    analyzed_commits: int
    status: RepoStatus
    error_message: str | None
    max_commits_setting: int
    github_stars: int | None
    github_language: str | None
    github_description: str | None
    active_contributors_count: int = 0


class CommitOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    repo_id: int
    sha: str
    full_sha: str
    message: str | None
    author_name: str | None
    author_email: str | None
    committed_at: datetime
    insertions: int
    deletions: int
    files_changed: int
    parent_sha: str | None


class TopFileMetric(BaseModel):
    path: str
    complexity: float
    loc: int


class RiskReason(BaseModel):
    code: str
    severity: str
    label: str
    detail: str
    impact: float


class PersistentHotspot(BaseModel):
    path: str
    recent_commit_count: int
    complexity: float
    loc: int


class HealthSnapshotOut(BaseModel):
    id: int | None = None
    repo_id: int | None = None
    commit_id: int | None = None
    sha: str
    full_sha: str
    message: str | None
    author: str | None
    author_email: str | None = None
    committed_at: datetime
    health_score: float
    avg_complexity: float
    max_complexity: float
    total_loc: int
    churn_rate: float
    num_files_changed: int
    insertions: int | None = None
    deletions: int | None = None
    bus_factor_min: int
    health_delta: float | None
    cc_score: float
    churn_score: float
    bus_score: float
    loc_score: float
    subscores: dict[str, float] = Field(default_factory=dict)
    dependency_density: float = 0.0
    has_cycles: bool = False
    hotspot_count: int = 0
    avg_semantic_drift: float = 0.0
    semantic_health_score: float = 100.0
    high_drift_files: int = 0
    semantic_drift_method: str = "none"
    risk_reasons: list[RiskReason] = Field(default_factory=list)
    hotspot_persistence_score: float = 0.0
    persistent_hotspots: list[PersistentHotspot] = Field(default_factory=list)
    top_files: list[TopFileMetric] = Field(default_factory=list)
    computed_at: datetime | None = None


class TimelineResponse(BaseModel):
    repo_id: int
    commits: list[HealthSnapshotOut]


class GraphNodeOut(BaseModel):
    id: str
    file: str
    module: str | None
    loc: int
    health: float
    health_color: HealthColor
    is_entry_point: bool
    semantic_drift_score: float = 0.0
    drift_method: str = "none"


class GraphEdgeOut(BaseModel):
    source: str
    target: str
    type: GraphEdgeType
    weight: int
    cochange_count: int | None = None


class GraphResponse(BaseModel):
    repo_id: int
    commit_sha: str
    nodes: list[GraphNodeOut]
    edges: list[GraphEdgeOut]


class HotspotItem(BaseModel):
    file: str
    complexity: float
    churn_count: int
    risk_score: float


class HotspotResponse(BaseModel):
    repo_id: int
    commit_sha: str
    hotspots: list[HotspotItem]
    total: int
    limit: int
    offset: int


class BusFactorEntryOut(BaseModel):
    module_path: str
    contributor_count: int
    top_contributor: str | None
    top_contributor_email: str | None
    top_contributor_pct: float
    total_commits_to_module: int
    risk_level: RiskLevel
    last_commit_sha: str | None
    last_updated_at: datetime | None = None


class BusFactorWrapper(BaseModel):
    repo_id: int
    modules: list[BusFactorEntryOut]


class NarrativeRequest(BaseModel):
    repo_id: int
    commit_sha: str = Field(..., min_length=7, max_length=40)
    prompt_type: PromptType = "explain_drop"


class PredictRequest(BaseModel):
    repo_id: int
    commit_sha: str = Field(..., min_length=7, max_length=40)
    prompt_type: Literal["predict_merge"] = "predict_merge"


class NarrativeResponse(BaseModel):
    repo_id: int
    commit_sha: str
    prompt_type: PromptType
    explanation: str
    tokens_used: int
    cost_usd: float
    cached: bool
    model: str
    provider: str | None = None
    demo_mode: bool = False


class NarrativeStreamChunk(BaseModel):
    token: str | None = None
    done: bool
    explanation: str | None = None
    tokens_total: int | None = None
    cost_usd: float | None = None
    cached: bool | None = None
    model: str | None = None
    provider: str | None = None
    demo_mode: bool | None = None
    error: str | None = None


class LLMUsageOut(BaseModel):
    repo_id: int
    total_calls: int
    cache_hits: int = 0
    anthropic_calls: int = 0
    gemini_calls: int = 0
    total_tokens: int
    total_cost_usd: float
    cache_savings_usd: float = 0.0
    budget_remaining: int
    max_calls: int


class CommitDetailResponse(BaseModel):
    repo: RepoOut
    commit: CommitOut
    snapshot: HealthSnapshotOut
    graph: GraphResponse
    bus_factor: BusFactorWrapper
    has_narrative: bool
    narrative: NarrativeResponse | None = None


class JobProgressOut(BaseModel):
    current: int
    total: int
    current_sha: str | None
    stage: str | None
    progress_pct: float
    status: JobStatus
    error_message: str | None = None


class DeploymentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    repo_id: int
    provider: str
    environment: str
    status: str
    ref: str | None = None
    sha: str | None = None
    pipeline_id: str | None = None
    deployed_at: datetime | None = None


class WebhookResponse(BaseModel):
    status: str
    message: str
    deployment_id: int | None = None
    repo_id: int | None = None


class RepoCompareMetrics(BaseModel):
    health_score: float = 0.0
    avg_complexity: float = 0.0
    max_complexity: float = 0.0
    churn_rate: float = 0.0
    total_loc: int = 0
    bus_factor_min: int = 1
    hotspot_count: int = 0
    active_contributors: int = 0
    total_commits: int = 0
    analyzed_commits: int = 0
    dependency_density: float = 0.0
    has_cycles: bool = False
    avg_semantic_drift: float = 0.0
    cc_score: float = 0.0
    churn_score: float = 0.0
    bus_score: float = 0.0
    loc_score: float = 0.0
    semantic_health_score: float = 100.0


class RepoCompareItem(BaseModel):
    repo: RepoOut
    latest_snapshot: HealthSnapshotOut | None = None
    metrics_summary: RepoCompareMetrics
    bus_factor: BusFactorWrapper
    timeline_summary: list[HealthSnapshotOut] = Field(default_factory=list)


class RepoCompareDelta(BaseModel):
    health_score_delta: float = 0.0
    avg_complexity_delta: float = 0.0
    max_complexity_delta: float = 0.0
    churn_rate_delta: float = 0.0
    total_loc_delta: int = 0
    bus_factor_min_delta: int = 0
    hotspot_count_delta: int = 0
    active_contributors_delta: int = 0
    total_commits_delta: int = 0


class RepoCompareInsight(BaseModel):
    category: str
    winner: str | None = None
    summary: str


class RepoCompareResponse(BaseModel):
    base: RepoCompareItem
    head: RepoCompareItem
    deltas: RepoCompareDelta
    insights: list[RepoCompareInsight] = Field(default_factory=list)
    verdict: str

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from backend.database import Base

ACTIVE_JOB_STATUSES = {
    "queued",
    "pending",
    "cloning",
    "analyzing",
    "building_graph",
    "computing_bus_factor",
    "extracting",
    "bus_factor",
    "finalizing",
}


class Repo(Base):
    __tablename__ = "repos"

    id = Column(Integer, primary_key=True, autoincrement=True)
    url = Column(String, nullable=False, unique=True)
    name = Column(String, nullable=False)
    owner = Column(String, nullable=False, default="")
    repo_slug = Column(String, nullable=False, unique=True)
    default_branch = Column(String, nullable=False, default="main")
    ingested_at = Column(DateTime(timezone=True), server_default=func.now())
    last_updated_at = Column(DateTime(timezone=True), nullable=True)
    total_commits = Column(Integer, nullable=False, default=0)
    analyzed_commits = Column(Integer, nullable=False, default=0)
    status = Column(String, nullable=False, default="pending")
    error_message = Column(Text, nullable=True)
    max_commits_setting = Column(Integer, nullable=False, default=150)
    github_stars = Column(Integer, nullable=True)
    github_language = Column(String, nullable=True)
    github_description = Column(Text, nullable=True)
    total_file_count = Column(Integer, nullable=False, default=0)
    total_repo_loc = Column(Integer, nullable=False, default=0)

    commits = relationship("Commit", back_populates="repo", cascade="all, delete-orphan")
    analysis_jobs = relationship("AnalysisJob", back_populates="repo", cascade="all, delete-orphan")
    bus_factor = relationship("BusFactor", back_populates="repo", cascade="all, delete-orphan")
    deployments = relationship("Deployment", back_populates="repo", cascade="all, delete-orphan")
    pull_requests = relationship("PullRequest", back_populates="repo", cascade="all, delete-orphan")
    report_schedules = relationship(
        "ReportSchedule", back_populates="repo", cascade="all, delete-orphan"
    )


class Deployment(Base):
    __tablename__ = "deployments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    repo_id = Column(Integer, ForeignKey("repos.id", ondelete="CASCADE"), nullable=False)
    provider = Column(String, nullable=False, default="gitlab")
    environment = Column(String, nullable=False, default="production")
    status = Column(String, nullable=False)
    ref = Column(String, nullable=True)
    sha = Column(String, nullable=True)
    pipeline_id = Column(String, nullable=True)
    deployed_at = Column(DateTime(timezone=True), server_default=func.now())

    repo = relationship("Repo", back_populates="deployments")

    __table_args__ = (Index("idx_deployments_repo_time", "repo_id", "deployed_at"),)


class Commit(Base):
    __tablename__ = "commits"

    id = Column(Integer, primary_key=True, autoincrement=True)
    repo_id = Column(Integer, ForeignKey("repos.id", ondelete="CASCADE"), nullable=False)
    sha = Column(String, nullable=False)
    full_sha = Column(String, nullable=False)
    message = Column(Text, nullable=True)
    author_name = Column(String, nullable=True)
    author_email = Column(String, nullable=True)
    committed_at = Column(DateTime(timezone=True), nullable=False)
    insertions = Column(Integer, nullable=False, default=0)
    deletions = Column(Integer, nullable=False, default=0)
    files_changed = Column(Integer, nullable=False, default=0)
    parent_sha = Column(String, nullable=True)

    repo = relationship("Repo", back_populates="commits")
    health_snapshot = relationship(
        "HealthSnapshot", back_populates="commit", uselist=False, cascade="all, delete-orphan"
    )
    graph_nodes = relationship("GraphNode", back_populates="commit", cascade="all, delete-orphan")
    graph_edges = relationship("GraphEdge", back_populates="commit", cascade="all, delete-orphan")
    llm_narratives = relationship(
        "LLMNarrative", back_populates="commit", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("repo_id", "full_sha", name="uq_commits_repo_fullsha"),
        Index("idx_commits_repo_time", "repo_id", "committed_at"),
        Index("idx_commits_repo_sha", "repo_id", "sha"),
    )


class HealthSnapshot(Base):
    __tablename__ = "health_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    repo_id = Column(Integer, ForeignKey("repos.id", ondelete="CASCADE"), nullable=False)
    commit_id = Column(
        Integer, ForeignKey("commits.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    full_sha = Column(String, nullable=False)
    health_score = Column(Float, nullable=False)
    avg_complexity = Column(Float, nullable=False, default=0.0)
    max_complexity = Column(Float, nullable=False, default=0.0)
    total_loc = Column(Integer, nullable=False, default=0)
    churn_rate = Column(Float, nullable=False, default=0.0)
    num_files_changed = Column(Integer, nullable=False, default=0)
    bus_factor_min = Column(Integer, nullable=False, default=1)
    health_delta = Column(Float, nullable=True)
    cc_score = Column(Float, nullable=False, default=0.0)
    churn_score = Column(Float, nullable=False, default=0.0)
    bus_score = Column(Float, nullable=False, default=0.0)
    loc_score = Column(Float, nullable=False, default=0.0)
    complexity_drift_score = Column(Float, nullable=False, default=0.0)
    churn_risk_score = Column(Float, nullable=False, default=0.0)
    bus_factor_risk_score = Column(Float, nullable=False, default=0.0)
    dependency_health_score = Column(Float, nullable=False, default=0.0)
    dependency_density = Column(Float, nullable=False, default=0.0)
    has_cycles = Column(Boolean, nullable=False, default=False)
    hotspot_count = Column(Integer, nullable=False, default=0)
    avg_semantic_drift = Column(Float, nullable=False, default=0.0)
    semantic_health_score = Column(Float, nullable=False, default=100.0)
    high_drift_files = Column(Integer, nullable=False, default=0)
    semantic_drift_method = Column(String, nullable=False, default="none")
    risk_reasons_json = Column(Text, nullable=True)
    hotspot_persistence_score = Column(Float, nullable=False, default=0.0)
    persistent_hotspots_json = Column(Text, nullable=True)
    top_files_json = Column(Text, nullable=True)
    computed_at = Column(DateTime(timezone=True), server_default=func.now())

    commit = relationship("Commit", back_populates="health_snapshot")

    __table_args__ = (
        Index("idx_snapshots_repo", "repo_id"),
        Index("idx_snapshots_fullsha", "repo_id", "full_sha"),
    )


class GraphNode(Base):
    __tablename__ = "graph_nodes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    repo_id = Column(Integer, ForeignKey("repos.id", ondelete="CASCADE"), nullable=False)
    commit_id = Column(Integer, ForeignKey("commits.id", ondelete="CASCADE"), nullable=False)
    full_sha = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    module_name = Column(String, nullable=True)
    loc = Column(Integer, nullable=False, default=0)
    avg_complexity = Column(Float, nullable=False, default=0.0)
    health_color = Column(String, nullable=False, default="green")
    is_entry_point = Column(Boolean, nullable=False, default=False)
    semantic_drift_score = Column(Float, nullable=False, default=0.0)
    drift_method = Column(String, nullable=False, default="none")

    commit = relationship("Commit", back_populates="graph_nodes")

    __table_args__ = (
        UniqueConstraint("repo_id", "commit_id", "file_path"),
        Index("idx_nodes_commit", "repo_id", "commit_id"),
        Index("idx_graph_nodes_drift", "repo_id", "semantic_drift_score"),
    )


class GraphEdge(Base):
    __tablename__ = "graph_edges"

    id = Column(Integer, primary_key=True, autoincrement=True)
    repo_id = Column(Integer, ForeignKey("repos.id", ondelete="CASCADE"), nullable=False)
    commit_id = Column(Integer, ForeignKey("commits.id", ondelete="CASCADE"), nullable=False)
    full_sha = Column(String, nullable=False)
    source_file = Column(String, nullable=False)
    target_file = Column(String, nullable=False)
    edge_type = Column(String, nullable=False)  # 'import' | 'co_change'
    weight = Column(Integer, nullable=False, default=1)
    cochange_count = Column(Integer, nullable=True)

    commit = relationship("Commit", back_populates="graph_edges")

    __table_args__ = (
        UniqueConstraint("repo_id", "commit_id", "source_file", "target_file", "edge_type"),
        Index("idx_edges_commit", "repo_id", "commit_id"),
        Index("idx_edges_sha", "repo_id", "full_sha"),
        Index("idx_edges_source", "repo_id", "source_file"),
    )


class BusFactor(Base):
    __tablename__ = "bus_factor"

    id = Column(Integer, primary_key=True, autoincrement=True)
    repo_id = Column(Integer, ForeignKey("repos.id", ondelete="CASCADE"), nullable=False)
    module_path = Column(String, nullable=False)
    contributor_count = Column(Integer, nullable=False)
    top_contributor = Column(String, nullable=True)
    top_contributor_email = Column(String, nullable=True)
    top_contributor_pct = Column(Float, nullable=False, default=0.0)
    total_commits_to_module = Column(Integer, nullable=False, default=0)
    risk_level = Column(String, nullable=False, default="low")
    last_commit_sha = Column(String, nullable=True)
    last_updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    repo = relationship("Repo", back_populates="bus_factor")

    __table_args__ = (
        UniqueConstraint("repo_id", "module_path"),
        Index("idx_bus_risk", "repo_id", "risk_level"),
    )


class LLMNarrative(Base):
    __tablename__ = "llm_narratives"

    id = Column(Integer, primary_key=True, autoincrement=True)
    repo_id = Column(Integer, ForeignKey("repos.id", ondelete="CASCADE"), nullable=False)
    commit_id = Column(Integer, ForeignKey("commits.id", ondelete="CASCADE"), nullable=False)
    full_sha = Column(String, nullable=False)
    prompt_type = Column(String, nullable=False)  # 'explain_drop' | 'predict_merge'
    cache_key = Column(String, nullable=False, unique=True)
    prompt_input = Column(Text, nullable=False)
    response_text = Column(Text, nullable=False)
    tokens_input = Column(Integer, nullable=False, default=0)
    tokens_output = Column(Integer, nullable=False, default=0)
    cost_usd = Column(Float, nullable=False, default=0.0)
    model_used = Column(String, nullable=False, default="claude-sonnet-4-20250514")
    is_pre_cached = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    commit = relationship("Commit", back_populates="llm_narratives")

    __table_args__ = (
        UniqueConstraint("repo_id", "commit_id", "prompt_type"),
        Index("idx_narratives_cache_key", "cache_key"),
        Index("idx_narratives_commit", "repo_id", "commit_id", "prompt_type"),
    )


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    repo_id = Column(Integer, ForeignKey("repos.id", ondelete="CASCADE"), nullable=False)
    status = Column(String, nullable=False, default="queued")
    total_commits = Column(Integer, nullable=False, default=0)
    processed_commits = Column(Integer, nullable=False, default=0)
    current_sha = Column(String, nullable=True)
    current_stage = Column(String, nullable=True)
    progress_pct = Column(Float, nullable=False, default=0.0)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    duration_seconds = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    triggered_by = Column(String, nullable=False, default="user")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    repo = relationship("Repo", back_populates="analysis_jobs")

    __table_args__ = (Index("idx_jobs_repo_created", "repo_id", "created_at"),)


class PullRequest(Base):
    __tablename__ = "pull_requests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    repo_id = Column(Integer, ForeignKey("repos.id", ondelete="CASCADE"), nullable=False)
    pr_number = Column(Integer, nullable=False)
    title = Column(String, nullable=False)
    state = Column(String, nullable=False)  # 'open', 'closed', 'merged'
    author = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)
    first_review_at = Column(DateTime(timezone=True), nullable=True)
    merged_at = Column(DateTime(timezone=True), nullable=True)
    closed_at = Column(DateTime(timezone=True), nullable=True)

    # Optional metrics
    coding_time_sec = Column(Integer, nullable=True)
    pickup_time_sec = Column(Integer, nullable=True)
    review_time_sec = Column(Integer, nullable=True)

    repo = relationship("Repo", back_populates="pull_requests")

    __table_args__ = (
        UniqueConstraint("repo_id", "pr_number", name="uq_pr_repo_number"),
        Index("idx_prs_repo_created", "repo_id", "created_at"),
    )


class ReportSchedule(Base):
    __tablename__ = "report_schedules"

    id = Column(Integer, primary_key=True, autoincrement=True)
    repo_id = Column(Integer, ForeignKey("repos.id", ondelete="CASCADE"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    cron_expression = Column(String, nullable=False)  # e.g. "0 9 * * MON" for weekly Monday 9am
    timezone = Column(String, nullable=False, default="UTC")
    report_type = Column(String, nullable=False, default="health_summary")
    # report_type values: health_summary, dora_metrics, team_health, full_analysis
    is_active = Column(Boolean, nullable=False, default=True)
    webhook_url = Column(Text, nullable=True)  # Slack / custom webhook URL
    webhook_secret = Column(String, nullable=True)  # HMAC signing secret for webhook
    notification_email = Column(String, nullable=True)
    include_narrative = Column(Boolean, nullable=False, default=False)
    last_run_at = Column(DateTime(timezone=True), nullable=True)
    next_run_at = Column(DateTime(timezone=True), nullable=True)
    last_delivery_status = Column(String, nullable=True)  # 'success', 'failed', 'pending'
    consecutive_failures = Column(Integer, nullable=False, default=0)
    max_retry_count = Column(Integer, nullable=False, default=3)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    repo = relationship("Repo", back_populates="report_schedules")
    deliveries = relationship(
        "ReportDelivery", back_populates="schedule", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_report_schedules_repo", "repo_id"),
        Index("idx_report_schedules_next_run", "next_run_at", "is_active"),
    )


class ReportDelivery(Base):
    __tablename__ = "report_deliveries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    schedule_id = Column(
        Integer, ForeignKey("report_schedules.id", ondelete="CASCADE"), nullable=False
    )
    repo_id = Column(Integer, ForeignKey("repos.id", ondelete="CASCADE"), nullable=False)
    status = Column(String, nullable=False, default="pending")  # pending, running, success, failed
    report_type = Column(String, nullable=False)
    triggered_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    duration_seconds = Column(Float, nullable=True)
    report_payload = Column(Text, nullable=True)  # JSON-encoded report summary data
    webhook_status_code = Column(Integer, nullable=True)
    webhook_response_body = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)
    snapshot_health_score = Column(Float, nullable=True)
    snapshot_commits_analyzed = Column(Integer, nullable=True)
    snapshot_latest_sha = Column(String, nullable=True)

    schedule = relationship("ReportSchedule", back_populates="deliveries")
    repo = relationship("Repo")

    __table_args__ = (
        Index("idx_report_deliveries_schedule", "schedule_id"),
        Index("idx_report_deliveries_repo_time", "repo_id", "triggered_at"),
    )

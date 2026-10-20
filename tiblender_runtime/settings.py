"""Runtime configuration loaded from environment variables and optional .env."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from .models import Severity
from .pivots import max_resolutions_per_ip


class MissingLLMConfigurationError(RuntimeError):
    """Raised when report generation is requested without an LLM configuration."""


def _load_dotenv(path: Path) -> None:
    # Preserve explicitly supplied environment values while still supporting a
    # local .env file for repeatable runs.
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


# Enumerated settings. An unrecognized value used to degrade silently -- e.g.
# TIBLENDER_MODE=ofline is neither "offline" nor "live", so is_offline() and
# is_live() both returned False and the run took the `auto` path, quietly
# making live network calls in what the operator believed was an offline run.
# from_env() now rejects the value outright instead.
MODES = ("auto", "live", "offline")
CLOCK_MODES = ("auto", "fixed", "wall")
RL_POLICY_BACKENDS = ("disabled", "multiaction_dqn")


def _choice_env(name: str, default: str, allowed: tuple[str, ...]) -> str:
    """Read an enumerated environment variable, case-insensitively.

    Raises ``ValueError`` naming both the offending value and the allowed set
    when the variable is set to something outside ``allowed``; an unset or
    empty variable falls back to ``default``.
    """
    raw = os.getenv(name, default)
    value = raw.strip().lower() or default
    if value not in allowed:
        raise ValueError(
            f"invalid {name}={raw.strip()!r}: expected one of {', '.join(allowed)}"
        )
    return value


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# Passive-DNS resolutions kept per lookup, matching the deployed system's
# 20-item truncation of the VirusTotal payload (`TIBLENDER_PASSIVE_DNS_LIMIT`).
DEFAULT_PASSIVE_DNS_LIMIT = 20


def _positive_int_env(name: str, default: int) -> int:
    """Read a chunk-size style variable, clamped to at least 1.

    A batch size of 0 or a negative value would make the chunking loop
    produce no requests at all, so an out-of-range or unparseable value is
    clamped instead of silently disabling the stage.
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return max(1, default)
    try:
        return max(1, int(raw.strip()))
    except ValueError as error:
        raise ValueError(f"invalid {name}={raw.strip()!r}: expected an integer") from error


def _first_env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def _path_env(*names: str) -> str:
    # Production .env files sometimes annotate file paths inline. Keep the
    # value shell-compatible while tolerating comments copied from deployment.
    value = _first_env(*names)
    if " #" in value:
        value = value.split(" #", 1)[0]
    return value.strip()


@dataclass(slots=True)
class Settings:
    # Defaults run the bundled sample input and mirror the production pipeline
    # knobs closely enough for one-cycle reproduction without private services.
    input_path: Path = Path("sample_runs/collector_experiment_sample/sample_run_posts.jsonl")
    output_dir: Path = Path("output")
    mode: str = "auto"  # auto | offline | live
    seed: int = 20260909
    clock_mode: str = "auto"  # auto | fixed | wall (auto: fixed when offline)
    fixtures_dir: Path = Path("fixtures")
    config_dir: Path = Path("config")
    history_dir: Path | None = None  # None -> resolved_history_dir() falls back to output_dir/"history"
    cycle_interval_seconds: int = 3600
    collect_window_hours: float = 1.0
    max_dynamic_queries_per_platform: int = 30
    max_posts_per_query: int = 100
    telegram_api_id: str = ""
    telegram_api_hash: str = ""
    telegram_session: str = ""
    telegram_max_channels: int = 0  # 0 = every joined channel
    discord_bot_token: str = ""
    discord_user_token: str = ""
    discord_max_guilds: int = 0  # 0 = every joined guild
    discord_max_channels_per_guild: int = 0  # 0 = every text channel
    # Passive DNS provider used in production. VirusTotal needs a key (free
    # tier: 4 requests/minute, 500/day).
    virustotal_api_key: str = ""
    ocr_enabled: bool = True
    pivot_depth: int = 1
    # Resolutions considered per IP before that IP counts as a shared-hosting
    # environment. Above it the co-resident names are other people's sites, so
    # they are dropped instead of shipped as campaign indicators. No
    # confidence bump follows: crowded hosting is a false-positive risk, not
    # corroboration (see pivots.py and docs/FIDELITY.md, Step 3).
    pivot_max_resolutions_per_ip: int = 5
    # Passive-DNS resolutions kept per lookup. Matches the deployed system's
    # 20-item truncation of the provider payload; the live VirusTotal request
    # asks for exactly this many, and recorded payloads captured under an
    # older, larger limit are truncated to it on replay so the two paths agree
    # without re-recording.
    passive_dns_limit: int = 20
    perspective_workers: int = 4
    llm_provider: str = "azure_openai"
    llm_model: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_deployment: str = ""
    azure_openai_api_version: str = ""
    llm_temperature: float = 0.0
    llm_max_output_tokens: int = 4096
    llm_timeout_seconds: int = 60
    # Batch agents (post normalization, cluster refinement) send their input in
    # chunks of this size so one provider reply never has to carry the whole
    # cycle: a single oversized request is what a provider truncates.
    llm_batch_size: int = 20
    llm_cluster_batch_size: int = 10
    # Chunked batch agents send their chunks concurrently; cluster-level
    # Step 3/4 work runs in a worker pool of this size. Both preserve
    # deterministic result order, so they never change a run's output.
    llm_max_concurrent: int = 8
    cluster_workers: int = 4
    allow_heuristic_reports: bool = False
    min_text_chars: int = 24
    jaccard_threshold: float = 0.25
    absorption_threshold: float = 0.75
    min_cluster_size: int = 3
    min_samples: int = 2
    cluster_epsilon: float = 0.08
    merge_similarity_threshold: float = 0.92
    merge_max_time_gap_seconds: int = 6 * 60 * 60
    merge_min_context_overlap: int = 2
    min_severity: Severity = Severity.MEDIUM
    max_evaluation_rounds: int = 3
    rl_similarity_gate: float = 0.60
    rl_max_steps: int = 10
    rl_action_top_k: int = 5
    trend_window_hours: int = 168
    trend_similarity_threshold: float = 0.80
    volume_sigma_k: float = 1.5
    trend_min_similar_snapshots: int = 4
    trend_volume_min_posts: int = 4
    early_warning_threshold: float = 0.75
    trend_embedding_backend: str = "hashing"
    max_queries_per_perspective: int = 5
    enable_pivot_iocs: bool = True
    enable_real_tools: bool = True
    tool_timeout_seconds: int = 20
    max_tool_results: int = 5
    # Read cap for a tool's JSON response body. NVD returns >200 KB for a
    # widely-referenced CVE, and a cap below the real payload truncates the
    # body mid-string, so the tool fails with `invalid JSON response` rather
    # than returning the record. 2 MiB clears every provider this runtime
    # calls while still bounding memory and time. WEB_CRAWL keeps its own,
    # much smaller cap: it fetches HTML, not a structured record.
    tool_max_bytes: int = 2 * 1024 * 1024
    web_crawl_max_bytes: int = 200_000
    web_crawl_render_js: bool = True
    web_crawl_wait_ms: int = 1000
    search_engine_provider: str = "searxng"
    search_engine_max_results: int = 10
    searxng_base_url: str = "http://searxng:8080"
    nvd_api_key: str = ""
    github_token: str = ""
    x_bearer_token: str = ""
    x_bearer_token_file: str = ""
    x_bearer_fallback_token_file: str = ""
    reddit_accounts_file: str = ""
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_extra_accounts: str = ""
    reddit_user_agent: str = "tiblender/0.1"
    embedding_backend: str = "hashing"
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_features: int = 384
    embedding_fallback: bool = True
    embedding_local_files_only: bool = True
    embedding_device: str = ""
    hdbscan_backend: str = "sklearn"
    judge_a_provider: str = ""
    judge_a_model: str = ""
    judge_a_base_url: str = ""
    judge_a_api_key: str = ""
    judge_a_azure_endpoint: str = ""
    judge_a_azure_deployment: str = ""
    judge_a_azure_api_version: str = ""
    judge_b_provider: str = ""
    judge_b_model: str = ""
    judge_b_base_url: str = ""
    judge_b_api_key: str = ""
    judge_b_azure_endpoint: str = ""
    judge_b_azure_deployment: str = ""
    judge_b_azure_api_version: str = ""
    rl_policy_backend: str = "multiaction_dqn"
    reference_rl_hidden_dim: int = 256
    reference_rl_dropout: float = 0.10
    reference_rl_bc_epochs: int = 20
    reference_rl_bc_batch_size: int = 32
    reference_rl_bc_learning_rate: float = 0.0003
    reference_rl_bc_patience: int = 3
    reference_rl_bc_validation_split: float = 0.10
    reference_rl_dqn_episodes: int = 50
    reference_rl_dqn_learning_rate: float = 0.0001
    reference_rl_replay_capacity: int = 10_000
    reference_rl_training_min_transitions: int = 1_000
    reference_rl_epsilon_init: float = 1.0
    reference_rl_epsilon_min: float = 0.01
    reference_rl_epsilon_decay: float = 0.9
    reference_rl_discount: float = 0.99

    @classmethod
    def from_env(cls) -> "Settings":
        # Both generic provider variables and TIBLENDER_* names are accepted so
        # Docker Compose, local shells, and CI can use the same settings class.
        env_file = Path(os.getenv("TIBLENDER_ENV_FILE", ".env"))
        _load_dotenv(env_file)
        llm_provider = os.getenv("TIBLENDER_LLM_PROVIDER", "azure_openai")
        provider_lower = llm_provider.strip().lower()
        if provider_lower in {"openai_compatible", "report"}:
            llm_model = _first_env(
                "TIBLENDER_REPORT_DEPLOYMENT",
                "TIBLENDER_LLM_MODEL",
            )
            openai_base_url = _first_env(
                "TIBLENDER_REPORT_ENDPOINT",
                "TIBLENDER_OPENAI_BASE_URL",
                default="https://api.openai.com/v1",
            )
            openai_api_key = _first_env(
                "TIBLENDER_REPORT_API_KEY",
                "OPENAI_API_KEY",
                "TIBLENDER_OPENAI_API_KEY",
                "AZURE_OPENAI_API_KEY",
                "TIBLENDER_AZURE_OPENAI_API_KEY",
            )
        elif provider_lower in {"azure", "azure_openai", "foundry", "microsoft_foundry"}:
            llm_model = _first_env(
                "TIBLENDER_LLM_MODEL",
                "AZURE_OPENAI_DEPLOYMENT",
                "TIBLENDER_AZURE_OPENAI_DEPLOYMENT",
            )
            openai_base_url = _first_env(
                "TIBLENDER_OPENAI_BASE_URL",
                default="https://api.openai.com/v1",
            )
            openai_api_key = _first_env(
                "AZURE_OPENAI_API_KEY",
                "TIBLENDER_AZURE_OPENAI_API_KEY",
                "OPENAI_API_KEY",
                "TIBLENDER_OPENAI_API_KEY",
            )
        else:
            llm_model = _first_env(
                "TIBLENDER_LLM_MODEL",
                "TIBLENDER_REPORT_DEPLOYMENT",
            )
            openai_base_url = _first_env(
                "TIBLENDER_OPENAI_BASE_URL",
                "TIBLENDER_REPORT_ENDPOINT",
                default="https://api.openai.com/v1",
            )
            openai_api_key = _first_env(
                "OPENAI_API_KEY",
                "TIBLENDER_OPENAI_API_KEY",
                "TIBLENDER_REPORT_API_KEY",
                "AZURE_OPENAI_API_KEY",
                "TIBLENDER_AZURE_OPENAI_API_KEY",
            )
        return cls(
            input_path=Path(os.getenv("TIBLENDER_INPUT_PATH", "sample_runs/collector_experiment_sample/sample_run_posts.jsonl")),
            output_dir=Path(os.getenv("TIBLENDER_OUTPUT_DIR", "output")),
            mode=_choice_env("TIBLENDER_MODE", "auto", MODES),
            seed=int(os.getenv("TIBLENDER_SEED", "20260909")),
            clock_mode=_choice_env("TIBLENDER_CLOCK", "auto", CLOCK_MODES),
            fixtures_dir=Path(os.getenv("TIBLENDER_FIXTURES_DIR", "fixtures")),
            config_dir=Path(os.getenv("TIBLENDER_CONFIG_DIR", "config")),
            history_dir=(Path(os.environ["TIBLENDER_HISTORY_DIR"]) if os.getenv("TIBLENDER_HISTORY_DIR") else None),
            cycle_interval_seconds=int(os.getenv("TIBLENDER_CYCLE_INTERVAL_SECONDS", "3600")),
            collect_window_hours=float(os.getenv("TIBLENDER_COLLECT_WINDOW_HOURS", "1.0")),
            max_dynamic_queries_per_platform=int(os.getenv("TIBLENDER_MAX_DYNAMIC_QUERIES_PER_PLATFORM", "30")),
            max_posts_per_query=int(os.getenv("TIBLENDER_MAX_POSTS_PER_QUERY", "100")),
            telegram_api_id=os.getenv("TIBLENDER_TELEGRAM_API_ID", ""),
            telegram_api_hash=os.getenv("TIBLENDER_TELEGRAM_API_HASH", ""),
            telegram_session=os.getenv("TIBLENDER_TELEGRAM_SESSION", ""),
            telegram_max_channels=int(os.getenv("TIBLENDER_TELEGRAM_MAX_CHANNELS", "0")),
            discord_bot_token=_first_env("TIBLENDER_DISCORD_BOT_TOKEN", "DISCORD_BOT_TOKEN"),
            discord_user_token=_first_env("TIBLENDER_DISCORD_USER_TOKEN", "DISCORD_USER_TOKEN"),
            discord_max_guilds=int(os.getenv("TIBLENDER_DISCORD_MAX_GUILDS", "0")),
            discord_max_channels_per_guild=int(os.getenv("TIBLENDER_DISCORD_MAX_CHANNELS_PER_GUILD", "0")),
            virustotal_api_key=_first_env("VIRUSTOTAL_API_KEY", "TIBLENDER_VIRUSTOTAL_API_KEY"),
            ocr_enabled=_bool_env("TIBLENDER_OCR_ENABLED", True),
            pivot_depth=int(os.getenv("TIBLENDER_PIVOT_DEPTH", "1")),
            pivot_max_resolutions_per_ip=max_resolutions_per_ip(),
            passive_dns_limit=_positive_int_env("TIBLENDER_PASSIVE_DNS_LIMIT", DEFAULT_PASSIVE_DNS_LIMIT),
            perspective_workers=int(os.getenv("TIBLENDER_PERSPECTIVE_WORKERS", "4")),
            llm_provider=llm_provider,
            llm_model=llm_model,
            openai_base_url=openai_base_url,
            openai_api_key=openai_api_key,
            azure_openai_endpoint=_first_env("AZURE_OPENAI_ENDPOINT", "TIBLENDER_AZURE_OPENAI_ENDPOINT"),
            azure_openai_deployment=_first_env("AZURE_OPENAI_DEPLOYMENT", "TIBLENDER_AZURE_OPENAI_DEPLOYMENT"),
            azure_openai_api_version=_first_env("AZURE_OPENAI_API_VERSION", "TIBLENDER_AZURE_OPENAI_API_VERSION"),
            llm_temperature=float(os.getenv("TIBLENDER_LLM_TEMPERATURE", "0.0")),
            llm_max_output_tokens=int(os.getenv("TIBLENDER_LLM_MAX_OUTPUT_TOKENS", "4096")),
            llm_timeout_seconds=int(os.getenv("TIBLENDER_LLM_TIMEOUT_SECONDS", "60")),
            llm_batch_size=_positive_int_env("TIBLENDER_LLM_BATCH_SIZE", 20),
            llm_cluster_batch_size=_positive_int_env("TIBLENDER_LLM_CLUSTER_BATCH_SIZE", 10),
            llm_max_concurrent=_positive_int_env("TIBLENDER_LLM_MAX_CONCURRENT", 8),
            cluster_workers=_positive_int_env("TIBLENDER_CLUSTER_WORKERS", 4),
            allow_heuristic_reports=_bool_env("TIBLENDER_ALLOW_HEURISTIC_REPORTS", False),
            min_text_chars=int(os.getenv("TIBLENDER_MIN_TEXT_CHARS", "24")),
            jaccard_threshold=float(os.getenv("TIBLENDER_JACCARD_THRESHOLD", "0.25")),
            absorption_threshold=float(os.getenv("TIBLENDER_ABSORPTION_THRESHOLD", "0.75")),
            min_cluster_size=int(os.getenv("TIBLENDER_MIN_CLUSTER_SIZE", "3")),
            min_samples=int(os.getenv("TIBLENDER_MIN_SAMPLES", "2")),
            cluster_epsilon=float(os.getenv("TIBLENDER_CLUSTER_EPSILON", "0.08")),
            merge_similarity_threshold=float(os.getenv("TIBLENDER_MERGE_SIMILARITY_THRESHOLD", "0.92")),
            merge_max_time_gap_seconds=int(os.getenv("TIBLENDER_MERGE_MAX_TIME_GAP_SECONDS", str(6 * 60 * 60))),
            merge_min_context_overlap=int(os.getenv("TIBLENDER_MERGE_MIN_CONTEXT_OVERLAP", "2")),
            min_severity=Severity(os.getenv("TIBLENDER_MIN_SEVERITY", "MEDIUM").upper()),
            max_evaluation_rounds=int(os.getenv("TIBLENDER_MAX_EVAL_ROUNDS", "3")),
            rl_similarity_gate=float(os.getenv("TIBLENDER_RL_SIMILARITY_GATE", "0.60")),
            rl_max_steps=int(os.getenv("TIBLENDER_RL_MAX_STEPS", "10")),
            rl_action_top_k=int(os.getenv("TIBLENDER_RL_ACTION_TOP_K", "5")),
            trend_window_hours=int(os.getenv("TIBLENDER_TREND_WINDOW_HOURS", "168")),
            trend_similarity_threshold=float(os.getenv("TIBLENDER_TREND_SIM_THRESHOLD", "0.80")),
            volume_sigma_k=float(os.getenv("TIBLENDER_VOLUME_SIGMA_K", "1.5")),
            trend_min_similar_snapshots=max(1, int(os.getenv("TIBLENDER_TREND_MIN_SIMILAR_SNAPSHOTS", "4"))),
            trend_volume_min_posts=max(1, int(os.getenv("TIBLENDER_TREND_VOLUME_MIN_POSTS", "4"))),
            early_warning_threshold=float(os.getenv("TIBLENDER_EARLY_WARNING_THRESHOLD", "0.75")),
            trend_embedding_backend=os.getenv("TIBLENDER_TREND_EMBEDDING_BACKEND", "hashing").strip().lower() or "hashing",
            max_queries_per_perspective=int(os.getenv("TIBLENDER_MAX_QUERIES_PER_PERSPECTIVE", "5")),
            enable_pivot_iocs=_bool_env("TIBLENDER_ENABLE_PIVOT_IOCS", True),
            enable_real_tools=_bool_env("TIBLENDER_ENABLE_REAL_TOOLS", True),
            tool_timeout_seconds=int(os.getenv("TIBLENDER_TOOL_TIMEOUT_SECONDS", "20")),
            max_tool_results=int(os.getenv("TIBLENDER_MAX_TOOL_RESULTS", "5")),
            tool_max_bytes=int(os.getenv("TIBLENDER_TOOL_MAX_BYTES", str(2 * 1024 * 1024))),
            web_crawl_max_bytes=int(os.getenv("TIBLENDER_WEB_CRAWL_MAX_BYTES", "200000")),
            web_crawl_render_js=_bool_env("TIBLENDER_WEB_CRAWL_RENDER_JS", True),
            web_crawl_wait_ms=int(os.getenv("TIBLENDER_WEB_CRAWL_WAIT_MS", "1000")),
            search_engine_provider=os.getenv("SEARCH_ENGINE_PROVIDER", "searxng").strip().lower() or "searxng",
            search_engine_max_results=int(os.getenv("SEARCH_ENGINE_MAX_RESULTS", "10")),
            searxng_base_url=os.getenv("SEARXNG_BASE_URL", "http://searxng:8080"),
            nvd_api_key=_first_env("NVD_API_KEY", "TIBLENDER_NVD_API_KEY"),
            github_token=_first_env("GITHUB_TOKEN", "GITHUB_API_KEY", "TIBLENDER_GITHUB_TOKEN"),
            x_bearer_token=_first_env(
                "X_BEARER_TOKEN",
                "TWITTER_BEARER_TOKEN",
                "TIBLENDER_X_BEARER_TOKEN",
                "TIBLENDER_TWITTER_BEARER_TOKEN",
            ),
            x_bearer_token_file=_path_env(
                "TIBLENDER_X_BEARER_TOKEN_FILE",
                "X_BEARER_TOKEN_FILE",
                "TWITTER_BEARER_TOKEN_FILE",
                "API_KEYS_FILE",
                "API_KEYS_PAYPERUSE_FILE",
            ),
            x_bearer_fallback_token_file=_path_env(
                "TIBLENDER_X_BEARER_FALLBACK_TOKEN_FILE",
                "X_BEARER_FALLBACK_TOKEN_FILE",
                "API_KEYS_FALLBACK_FILE",
            ),
            reddit_accounts_file=_path_env("TIBLENDER_REDDIT_ACCOUNTS_FILE", "REDDIT_ACCOUNTS_FILE"),
            reddit_client_id=_first_env("TIBLENDER_REDDIT_CLIENT_ID", "REDDIT_CLIENT_ID"),
            reddit_client_secret=_first_env("TIBLENDER_REDDIT_CLIENT_SECRET", "REDDIT_CLIENT_SECRET"),
            reddit_extra_accounts=_first_env("TIBLENDER_REDDIT_EXTRA_ACCOUNTS", "REDDIT_EXTRA_ACCOUNTS"),
            reddit_user_agent=os.getenv("TIBLENDER_REDDIT_USER_AGENT", "tiblender/0.1"),
            embedding_backend=os.getenv("TIBLENDER_EMBEDDING_BACKEND", "hashing").strip().lower() or "hashing",
            embedding_model=os.getenv("TIBLENDER_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"),
            embedding_features=int(os.getenv("TIBLENDER_EMBEDDING_FEATURES", "384")),
            embedding_fallback=_bool_env("TIBLENDER_EMBEDDING_FALLBACK", True),
            embedding_local_files_only=_bool_env("TIBLENDER_EMBEDDING_LOCAL_FILES_ONLY", True),
            embedding_device=os.getenv("TIBLENDER_EMBEDDING_DEVICE", ""),
            hdbscan_backend=os.getenv("TIBLENDER_HDBSCAN_BACKEND", "sklearn"),
            judge_a_provider=os.getenv("TIBLENDER_JUDGE_A_PROVIDER", ""),
            judge_a_model=os.getenv("TIBLENDER_JUDGE_A_MODEL", ""),
            judge_a_base_url=os.getenv("TIBLENDER_JUDGE_A_OPENAI_BASE_URL", ""),
            judge_a_api_key=_first_env("TIBLENDER_JUDGE_A_OPENAI_API_KEY", "TIBLENDER_JUDGE_A_AZURE_OPENAI_API_KEY"),
            judge_a_azure_endpoint=os.getenv("TIBLENDER_JUDGE_A_AZURE_OPENAI_ENDPOINT", ""),
            judge_a_azure_deployment=os.getenv("TIBLENDER_JUDGE_A_AZURE_OPENAI_DEPLOYMENT", ""),
            judge_a_azure_api_version=os.getenv("TIBLENDER_JUDGE_A_AZURE_OPENAI_API_VERSION", ""),
            judge_b_provider=os.getenv("TIBLENDER_JUDGE_B_PROVIDER", ""),
            judge_b_model=os.getenv("TIBLENDER_JUDGE_B_MODEL", ""),
            judge_b_base_url=os.getenv("TIBLENDER_JUDGE_B_OPENAI_BASE_URL", ""),
            judge_b_api_key=_first_env("TIBLENDER_JUDGE_B_OPENAI_API_KEY", "TIBLENDER_JUDGE_B_AZURE_OPENAI_API_KEY"),
            judge_b_azure_endpoint=os.getenv("TIBLENDER_JUDGE_B_AZURE_OPENAI_ENDPOINT", ""),
            judge_b_azure_deployment=os.getenv("TIBLENDER_JUDGE_B_AZURE_OPENAI_DEPLOYMENT", ""),
            judge_b_azure_api_version=os.getenv("TIBLENDER_JUDGE_B_AZURE_OPENAI_API_VERSION", ""),
            rl_policy_backend=_choice_env("TIBLENDER_RL_POLICY_BACKEND", "multiaction_dqn", RL_POLICY_BACKENDS),
            reference_rl_hidden_dim=int(os.getenv("TIBLENDER_REFERENCE_RL_HIDDEN_DIM", "256")),
            reference_rl_dropout=float(os.getenv("TIBLENDER_REFERENCE_RL_DROPOUT", "0.10")),
            reference_rl_bc_epochs=int(os.getenv("TIBLENDER_REFERENCE_RL_BC_EPOCHS", "20")),
            reference_rl_bc_batch_size=int(os.getenv("TIBLENDER_REFERENCE_RL_BC_BATCH_SIZE", "32")),
            reference_rl_bc_learning_rate=float(os.getenv("TIBLENDER_REFERENCE_RL_BC_LEARNING_RATE", "0.0003")),
            reference_rl_bc_patience=int(os.getenv("TIBLENDER_REFERENCE_RL_BC_PATIENCE", "3")),
            reference_rl_bc_validation_split=float(os.getenv("TIBLENDER_REFERENCE_RL_BC_VALIDATION_SPLIT", "0.10")),
            reference_rl_dqn_episodes=int(os.getenv("TIBLENDER_REFERENCE_RL_DQN_EPISODES", "50")),
            reference_rl_dqn_learning_rate=float(os.getenv("TIBLENDER_REFERENCE_RL_DQN_LEARNING_RATE", "0.0001")),
            reference_rl_replay_capacity=int(os.getenv("TIBLENDER_REFERENCE_RL_REPLAY_CAPACITY", "10000")),
            reference_rl_training_min_transitions=int(os.getenv("TIBLENDER_REFERENCE_RL_TRAINING_MIN_TRANSITIONS", "1000")),
            reference_rl_epsilon_init=float(os.getenv("TIBLENDER_REFERENCE_RL_EPSILON_INIT", "1.0")),
            reference_rl_epsilon_min=float(os.getenv("TIBLENDER_REFERENCE_RL_EPSILON_MIN", "0.01")),
            reference_rl_epsilon_decay=float(os.getenv("TIBLENDER_REFERENCE_RL_EPSILON_DECAY", "0.9")),
            reference_rl_discount=float(os.getenv("TIBLENDER_REFERENCE_RL_DISCOUNT", "0.99")),
        )

    def for_judge(self, label: str) -> "Settings":
        # Judges may have independent deployments, but inherit the main LLM
        # configuration when judge-specific values are omitted.
        prefix = label.lower()
        if prefix not in {"a", "b"}:
            raise ValueError("judge label must be 'a' or 'b'")
        provider = getattr(self, f"judge_{prefix}_provider") or self.llm_provider
        model = getattr(self, f"judge_{prefix}_model") or self.llm_model
        base_url = getattr(self, f"judge_{prefix}_base_url") or self.openai_base_url
        api_key = getattr(self, f"judge_{prefix}_api_key") or self.openai_api_key
        azure_endpoint = getattr(self, f"judge_{prefix}_azure_endpoint") or self.azure_openai_endpoint
        azure_deployment = getattr(self, f"judge_{prefix}_azure_deployment") or self.azure_openai_deployment
        azure_api_version = getattr(self, f"judge_{prefix}_azure_api_version") or self.azure_openai_api_version
        return replace(
            self,
            llm_provider=provider,
            llm_model=model,
            openai_base_url=base_url,
            openai_api_key=api_key,
            azure_openai_endpoint=azure_endpoint,
            azure_openai_deployment=azure_deployment,
            azure_openai_api_version=azure_api_version,
        )

    def resolved_history_dir(self) -> Path:
        """Effective directory for cross-cycle trend history, policy state,
        and collector state: everything that must carry across
        cycles lives here.

        ``history_dir`` is ``None`` unless a caller explicitly set it
        (directly, via ``dataclasses.replace``, or via
        ``TIBLENDER_HISTORY_DIR``). When unset, it resolves relative to
        ``output_dir`` -- so a single-cycle run pointed at a tmp_path
        ``output_dir`` still gets its own isolated history tree. When set,
        it is always used verbatim.

        This used to compare ``history_dir`` to the module's own default
        *by value* rather than tracking explicitness, which meant an
        explicitly-set ``history_dir`` that happened to equal the literal
        default string ("output/history") was silently treated as unset and
        re-derived from ``output_dir`` instead. That broke the multi-cycle
        scheduler specifically: it explicitly sets
        ``history_dir=output_root/"history"`` for every cycle so state
        carries across cycles, but whenever ``output_root`` was named
        "output" (the common/documented case), that explicit value equalled
        the old default and got silently discarded in favor of each cycle's
        own ``output_dir`` (``output/cycle_00N``) -- isolating history,
        policy state, and trend detection per cycle instead of sharing them.
        """
        if self.history_dir is not None:
            return Path(self.history_dir)
        return Path(self.output_dir) / "history"

    def is_offline(self) -> bool:
        return self.mode == "offline"

    def is_live(self) -> bool:
        return self.mode == "live"

    def uses_fixed_clock(self) -> bool:
        """Whether this run pins the clock instead of reading wall time.

        ``TIBLENDER_CLOCK=auto`` (the default) means "fixed when the run is
        fully offline". Every writer of a timestamp that must be reproducible
        -- the scheduler's per-cycle clock and the viewer bundle's build stamp
        -- asks this one predicate rather than re-deriving the rule.
        """
        return self.clock_mode == "fixed" or (self.clock_mode == "auto" and self.is_offline())

    def llm_configured(self) -> bool:
        provider = self.llm_provider.strip().lower()
        if provider in {"", "heuristic", "offline", "none"}:
            return False
        if provider in {"openai", "openai_compatible"}:
            return bool(self.openai_api_key and self.llm_model and self.openai_base_url)
        if provider in {"azure", "azure_openai", "foundry", "microsoft_foundry"}:
            return bool(
                self.openai_api_key
                and self.azure_openai_endpoint
                and self.azure_openai_deployment
                and self.azure_openai_api_version
            )
        return bool(self.openai_api_key and self.llm_model)

    def validate_report_generation(self) -> None:
        # Normal runs are LLM-backed. Heuristic reports must be opted into so
        # smoke tests cannot be mistaken for full report generation, except
        # non-live runs (offline/auto) which may fall back to heuristics
        # automatically since no live LLM is expected to be reachable there.
        if self.allow_heuristic_reports:
            return
        if self.llm_configured():
            return
        if self.mode != "live":
            self.allow_heuristic_reports = True
            return
        raise MissingLLMConfigurationError(
            "LLM configuration is required for full LLM-backed report generation. "
            "Set TIBLENDER_LLM_PROVIDER plus the matching model/deployment and key in .env. "
            "For local smoke tests only, set TIBLENDER_ALLOW_HEURISTIC_REPORTS=true."
        )

    def prepare_output_dirs(self) -> None:
        for relative in ("reports", "markdown", "stix", "pdf", "intermediate"):
            (self.output_dir / relative).mkdir(parents=True, exist_ok=True)

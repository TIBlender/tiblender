from pathlib import Path

import pytest

from tiblender_runtime.models import Severity
from tiblender_runtime.settings import MissingLLMConfigurationError, Settings


def test_settings_default_input_uses_collector_sample(monkeypatch):
    monkeypatch.delenv("TIBLENDER_INPUT_PATH", raising=False)

    settings = Settings.from_env()

    assert settings.input_path == Path("sample_runs/collector_experiment_sample/sample_run_posts.jsonl")


def test_settings_reads_runtime_hyperparameters_from_env(monkeypatch):
    monkeypatch.setenv("TIBLENDER_INPUT_PATH", "data/input.jsonl")
    monkeypatch.setenv("TIBLENDER_OUTPUT_DIR", "data/out")
    monkeypatch.setenv("TIBLENDER_MIN_TEXT_CHARS", "42")
    monkeypatch.setenv("TIBLENDER_MIN_CLUSTER_SIZE", "5")
    monkeypatch.setenv("TIBLENDER_MIN_SAMPLES", "3")
    monkeypatch.setenv("TIBLENDER_CLUSTER_EPSILON", "0.11")
    monkeypatch.setenv("TIBLENDER_JACCARD_THRESHOLD", "0.31")
    monkeypatch.setenv("TIBLENDER_ABSORPTION_THRESHOLD", "0.77")
    monkeypatch.setenv("TIBLENDER_MERGE_SIMILARITY_THRESHOLD", "0.93")
    monkeypatch.setenv("TIBLENDER_MERGE_MAX_TIME_GAP_SECONDS", "7200")
    monkeypatch.setenv("TIBLENDER_MERGE_MIN_CONTEXT_OVERLAP", "4")
    monkeypatch.setenv("TIBLENDER_MIN_SEVERITY", "HIGH")
    monkeypatch.setenv("TIBLENDER_MAX_EVAL_ROUNDS", "4")
    monkeypatch.setenv("TIBLENDER_RL_SIMILARITY_GATE", "0.61")
    monkeypatch.setenv("TIBLENDER_RL_MAX_STEPS", "12")
    monkeypatch.setenv("TIBLENDER_RL_ACTION_TOP_K", "4")
    monkeypatch.setenv("TIBLENDER_TREND_WINDOW_HOURS", "72")
    monkeypatch.setenv("TIBLENDER_TREND_SIM_THRESHOLD", "0.82")
    monkeypatch.setenv("TIBLENDER_VOLUME_SIGMA_K", "1.7")
    monkeypatch.setenv("TIBLENDER_TREND_MIN_SIMILAR_SNAPSHOTS", "6")
    monkeypatch.setenv("TIBLENDER_TREND_VOLUME_MIN_POSTS", "7")
    monkeypatch.setenv("TIBLENDER_EARLY_WARNING_THRESHOLD", "0.81")
    monkeypatch.setenv("TIBLENDER_MAX_QUERIES_PER_PERSPECTIVE", "6")
    monkeypatch.setenv("TIBLENDER_ENABLE_PIVOT_IOCS", "false")
    monkeypatch.setenv("TIBLENDER_ALLOW_HEURISTIC_REPORTS", "true")
    monkeypatch.setenv("TIBLENDER_ENABLE_REAL_TOOLS", "false")
    monkeypatch.setenv("TIBLENDER_TOOL_TIMEOUT_SECONDS", "7")
    monkeypatch.setenv("TIBLENDER_MAX_TOOL_RESULTS", "9")
    monkeypatch.setenv("TIBLENDER_WEB_CRAWL_MAX_BYTES", "12345")
    monkeypatch.setenv("TIBLENDER_EMBEDDING_BACKEND", "bge_large")
    monkeypatch.setenv("TIBLENDER_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    monkeypatch.setenv("TIBLENDER_EMBEDDING_FEATURES", "1024")
    monkeypatch.setenv("TIBLENDER_EMBEDDING_FALLBACK", "false")
    monkeypatch.setenv("TIBLENDER_EMBEDDING_LOCAL_FILES_ONLY", "false")
    monkeypatch.setenv("TIBLENDER_RL_POLICY_BACKEND", "disabled")
    monkeypatch.setenv("TIBLENDER_REFERENCE_RL_HIDDEN_DIM", "256")
    monkeypatch.setenv("TIBLENDER_REFERENCE_RL_BC_EPOCHS", "20")
    monkeypatch.setenv("TIBLENDER_REFERENCE_RL_BC_BATCH_SIZE", "32")
    monkeypatch.setenv("TIBLENDER_REFERENCE_RL_BC_LEARNING_RATE", "0.0003")
    monkeypatch.setenv("TIBLENDER_REFERENCE_RL_DQN_EPISODES", "50")
    monkeypatch.setenv("TIBLENDER_REFERENCE_RL_DQN_LEARNING_RATE", "0.0001")
    monkeypatch.setenv("TIBLENDER_REFERENCE_RL_REPLAY_CAPACITY", "10000")
    monkeypatch.setenv("TIBLENDER_REFERENCE_RL_TRAINING_MIN_TRANSITIONS", "1000")

    settings = Settings.from_env()

    assert settings.input_path == Path("data/input.jsonl")
    assert settings.output_dir == Path("data/out")
    assert settings.min_text_chars == 42
    assert settings.min_cluster_size == 5
    assert settings.min_samples == 3
    assert settings.cluster_epsilon == 0.11
    assert settings.jaccard_threshold == 0.31
    assert settings.absorption_threshold == 0.77
    assert settings.merge_similarity_threshold == 0.93
    assert settings.merge_max_time_gap_seconds == 7200
    assert settings.merge_min_context_overlap == 4
    assert settings.min_severity == Severity.HIGH
    assert settings.max_evaluation_rounds == 4
    assert settings.rl_similarity_gate == 0.61
    assert settings.rl_max_steps == 12
    assert settings.rl_action_top_k == 4
    assert settings.trend_window_hours == 72
    assert settings.trend_similarity_threshold == 0.82
    assert settings.volume_sigma_k == 1.7
    assert settings.trend_min_similar_snapshots == 6
    assert settings.trend_volume_min_posts == 7
    assert settings.early_warning_threshold == 0.81
    assert settings.max_queries_per_perspective == 6
    assert settings.enable_pivot_iocs is False
    assert settings.allow_heuristic_reports is True
    assert settings.enable_real_tools is False
    assert settings.tool_timeout_seconds == 7
    assert settings.max_tool_results == 9
    assert settings.web_crawl_max_bytes == 12345
    assert settings.embedding_backend == "bge_large"
    assert settings.embedding_model == "BAAI/bge-large-en-v1.5"
    assert settings.embedding_features == 1024
    assert settings.embedding_fallback is False
    assert settings.embedding_local_files_only is False
    assert settings.rl_policy_backend == "disabled"
    assert settings.reference_rl_hidden_dim == 256
    assert settings.reference_rl_bc_epochs == 20
    assert settings.reference_rl_bc_batch_size == 32
    assert settings.reference_rl_bc_learning_rate == 0.0003
    assert settings.reference_rl_dqn_episodes == 50
    assert settings.reference_rl_dqn_learning_rate == 0.0001
    assert settings.reference_rl_replay_capacity == 10000
    assert settings.reference_rl_training_min_transitions == 1000


def test_settings_identifies_missing_llm_configuration():
    settings = Settings(llm_provider="heuristic", openai_api_key="")

    assert settings.llm_configured() is False


def test_settings_defaults_to_azure_openai_provider(monkeypatch):
    for name in (
        "TIBLENDER_LLM_PROVIDER",
        "TIBLENDER_ENV_FILE",
        "OPENAI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "TIBLENDER_AZURE_OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TIBLENDER_ENV_FILE", "/tmp/tiblender-missing-env")

    settings = Settings.from_env()

    assert settings.llm_provider == "azure_openai"
    assert settings.llm_configured() is False


def test_settings_accepts_azure_openai_api_key_fallback(monkeypatch):
    monkeypatch.setenv("TIBLENDER_LLM_PROVIDER", "azure_openai")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "deployment")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2026-01-01")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "dummy-azure-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    settings = Settings.from_env()

    assert settings.openai_api_key == "dummy-azure-key"
    assert settings.llm_configured() is True


def test_settings_accepts_tiblender_prefixed_provider_keys(monkeypatch):
    for name in (
        "OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_API_KEY",
        "X_BEARER_TOKEN",
        "TWITTER_BEARER_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TIBLENDER_LLM_PROVIDER", "azure_openai")
    monkeypatch.setenv("TIBLENDER_AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("TIBLENDER_AZURE_OPENAI_DEPLOYMENT", "deployment")
    monkeypatch.setenv("TIBLENDER_AZURE_OPENAI_API_VERSION", "2026-01-01")
    monkeypatch.setenv("TIBLENDER_AZURE_OPENAI_API_KEY", "dummy-azure-key")
    monkeypatch.setenv("TIBLENDER_X_BEARER_TOKEN", "dummy-x-token")

    settings = Settings.from_env()

    assert settings.openai_api_key == "dummy-azure-key"
    assert settings.azure_openai_endpoint == "https://example.openai.azure.com"
    assert settings.azure_openai_deployment == "deployment"
    assert settings.azure_openai_api_version == "2026-01-01"
    assert settings.x_bearer_token == "dummy-x-token"
    assert settings.llm_configured() is True


def test_settings_maps_custom_report_provider_to_openai_compatible_fields(monkeypatch):
    monkeypatch.setenv("TIBLENDER_LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("TIBLENDER_OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("TIBLENDER_REPORT_ENDPOINT", "https://example.provider.test/v1")
    monkeypatch.setenv("TIBLENDER_REPORT_DEPLOYMENT", "custom-model")
    monkeypatch.setenv("TIBLENDER_REPORT_API_KEY", "dummy-report-key")

    settings = Settings.from_env()

    assert settings.llm_provider == "openai_compatible"
    assert settings.openai_base_url == "https://example.provider.test/v1"
    assert settings.llm_model == "custom-model"
    assert settings.openai_api_key == "dummy-report-key"
    assert settings.llm_configured() is True


def test_settings_reads_production_platform_credential_paths(monkeypatch):
    monkeypatch.setenv("TIBLENDER_X_BEARER_TOKEN_FILE", "/app/collectors/x_crawler/config/api_keys.txt")
    monkeypatch.setenv("TIBLENDER_X_BEARER_FALLBACK_TOKEN_FILE", "/app/collectors/x_crawler/config/fallback_keys.txt")
    monkeypatch.setenv("TIBLENDER_REDDIT_ACCOUNTS_FILE", "/app/collectors/reddit_crawler/config/reddit_accounts.txt")
    monkeypatch.setenv("TIBLENDER_REDDIT_CLIENT_ID", "reddit-client")
    monkeypatch.setenv("TIBLENDER_REDDIT_CLIENT_SECRET", "reddit-secret")
    monkeypatch.setenv("TIBLENDER_REDDIT_EXTRA_ACCOUNTS", "extra-client:extra-secret")

    settings = Settings.from_env()

    assert settings.x_bearer_token_file == "/app/collectors/x_crawler/config/api_keys.txt"
    assert settings.x_bearer_fallback_token_file == "/app/collectors/x_crawler/config/fallback_keys.txt"
    assert settings.reddit_accounts_file == "/app/collectors/reddit_crawler/config/reddit_accounts.txt"
    assert settings.reddit_client_id == "reddit-client"
    assert settings.reddit_client_secret == "reddit-secret"
    assert settings.reddit_extra_accounts == "extra-client:extra-secret"


def test_settings_reads_independent_judge_configs(monkeypatch):
    monkeypatch.setenv("TIBLENDER_JUDGE_A_PROVIDER", "azure_openai")
    monkeypatch.setenv("TIBLENDER_JUDGE_A_AZURE_OPENAI_ENDPOINT", "https://judge-a.example")
    monkeypatch.setenv("TIBLENDER_JUDGE_A_AZURE_OPENAI_DEPLOYMENT", "judge-a")
    monkeypatch.setenv("TIBLENDER_JUDGE_A_AZURE_OPENAI_API_VERSION", "2026-01-01")
    monkeypatch.setenv("TIBLENDER_JUDGE_A_AZURE_OPENAI_API_KEY", "dummy-judge-a-key")
    monkeypatch.setenv("TIBLENDER_JUDGE_B_PROVIDER", "openai")
    monkeypatch.setenv("TIBLENDER_JUDGE_B_MODEL", "judge-b-model")
    monkeypatch.setenv("TIBLENDER_JUDGE_B_OPENAI_API_KEY", "dummy-judge-b-key")

    settings = Settings.from_env()
    judge_a = settings.for_judge("a")
    judge_b = settings.for_judge("b")

    assert judge_a.llm_provider == "azure_openai"
    assert judge_a.azure_openai_endpoint == "https://judge-a.example"
    assert judge_a.azure_openai_deployment == "judge-a"
    assert judge_a.openai_api_key == "dummy-judge-a-key"
    assert judge_a.llm_configured() is True
    assert judge_b.llm_provider == "openai"
    assert judge_b.llm_model == "judge-b-model"
    assert judge_b.openai_api_key == "dummy-judge-b-key"
    assert judge_b.llm_configured() is True


def test_offline_mode_allows_heuristic_reports_without_llm(monkeypatch):
    settings = Settings(mode="offline", llm_provider="heuristic")
    settings.validate_report_generation()
    assert settings.allow_heuristic_reports is True


def test_live_mode_requires_llm():
    settings = Settings(mode="live", llm_provider="heuristic")
    with pytest.raises(MissingLLMConfigurationError):
        settings.validate_report_generation()


def test_env_reads_new_collector_settings(monkeypatch):
    monkeypatch.setenv("TIBLENDER_MODE", "offline")
    monkeypatch.setenv("TIBLENDER_DISCORD_BOT_TOKEN", "abc")
    monkeypatch.setenv("TIBLENDER_TELEGRAM_API_ID", "123")
    monkeypatch.setenv("TIBLENDER_VIRUSTOTAL_API_KEY", "vt")
    settings = Settings.from_env()
    assert settings.mode == "offline" and settings.discord_bot_token == "abc"
    assert settings.telegram_api_id == "123" and settings.virustotal_api_key == "vt"


def test_env_reads_the_discord_user_token_and_the_collection_caps(monkeypatch):
    monkeypatch.setenv("TIBLENDER_DISCORD_USER_TOKEN", "usr")
    monkeypatch.setenv("TIBLENDER_DISCORD_MAX_GUILDS", "10")
    monkeypatch.setenv("TIBLENDER_DISCORD_MAX_CHANNELS_PER_GUILD", "5")
    monkeypatch.setenv("TIBLENDER_TELEGRAM_MAX_CHANNELS", "20")
    settings = Settings.from_env()
    assert settings.discord_user_token == "usr"
    assert settings.discord_max_guilds == 10
    assert settings.discord_max_channels_per_guild == 5
    assert settings.telegram_max_channels == 20


def test_collection_caps_default_to_no_cap(monkeypatch):
    for name in (
        "TIBLENDER_DISCORD_MAX_GUILDS",
        "TIBLENDER_DISCORD_MAX_CHANNELS_PER_GUILD",
        "TIBLENDER_TELEGRAM_MAX_CHANNELS",
        "TIBLENDER_DISCORD_USER_TOKEN",
        "DISCORD_USER_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = Settings.from_env()
    assert settings.discord_max_guilds == 0
    assert settings.discord_max_channels_per_guild == 0
    assert settings.telegram_max_channels == 0
    assert settings.discord_user_token == ""


def test_retired_collector_settings_are_gone():
    settings = Settings()
    for retired in ("discord_directories", "telegram_snowball_depth"):
        assert not hasattr(settings, retired)


def test_rl_env_defaults_match_the_dataclass_defaults(monkeypatch):
    monkeypatch.delenv("TIBLENDER_RL_POLICY_BACKEND", raising=False)
    monkeypatch.delenv("TIBLENDER_REFERENCE_RL_EPSILON_DECAY", raising=False)

    settings = Settings.from_env()

    assert settings.rl_policy_backend == "multiaction_dqn" == Settings().rl_policy_backend
    assert settings.reference_rl_epsilon_decay == 0.9 == Settings().reference_rl_epsilon_decay


def test_rl_env_overrides_are_read(monkeypatch):
    monkeypatch.setenv("TIBLENDER_RL_POLICY_BACKEND", "disabled")
    monkeypatch.setenv("TIBLENDER_REFERENCE_RL_EPSILON_DECAY", "0.75")

    settings = Settings.from_env()

    assert settings.rl_policy_backend == "disabled"
    assert settings.reference_rl_epsilon_decay == 0.75


def test_removed_rl_settings_are_gone():
    settings = Settings()

    for name in ("rl_entropy_gate", "dqn_hidden_dim", "dqn_learning_rate"):
        assert not hasattr(settings, name)


def test_resolved_history_dir_defaults_under_an_overridden_output_dir():
    settings = Settings(output_dir=Path("/tmp/some-cycle-output"))

    assert settings.resolved_history_dir() == Path("/tmp/some-cycle-output/history")


def test_resolved_history_dir_is_unchanged_when_both_are_at_their_defaults():
    settings = Settings()

    assert settings.resolved_history_dir() == Path("output/history")


def test_resolved_history_dir_respects_an_explicit_history_dir_override():
    settings = Settings(output_dir=Path("/tmp/some-cycle-output"), history_dir=Path("/tmp/explicit-history"))

    assert settings.resolved_history_dir() == Path("/tmp/explicit-history")


def test_resolved_history_dir_is_not_discarded_when_it_equals_the_single_cycle_default_string():
    # Regression: resolved_history_dir() used to compare
    # history_dir to the module's own default *by value*, so an explicitly
    # set history_dir that happened to equal the literal string
    # "output/history" was silently treated as unset and re-derived from
    # output_dir instead. That is exactly what the multi-cycle scheduler
    # hits whenever its output_root is named "output": it explicitly sets
    # history_dir=output_root/"history" ("output/history") on every cycle so
    # state carries across cycles, while each cycle's own output_dir is
    # output/cycle_00N -- the old value-comparison heuristic silently
    # re-derived "output/cycle_00N/history" instead, isolating history per
    # cycle. history_dir is now Optional and tracked by explicitness
    # (None vs. set), not by value, so this must return the explicit value.
    settings = Settings(output_dir=Path("output/cycle_002"), history_dir=Path("output/history"))

    assert settings.resolved_history_dir() == Path("output/history")


def test_history_dir_defaults_to_none():
    assert Settings().history_dir is None


def test_history_dir_from_env_is_none_when_unset(monkeypatch):
    monkeypatch.delenv("TIBLENDER_HISTORY_DIR", raising=False)

    settings = Settings.from_env()

    assert settings.history_dir is None
    assert settings.resolved_history_dir() == settings.output_dir / "history"


def test_history_dir_from_env_is_read_when_set(monkeypatch, tmp_path):
    monkeypatch.setenv("TIBLENDER_HISTORY_DIR", str(tmp_path / "shared-history"))

    settings = Settings.from_env()

    assert settings.history_dir == tmp_path / "shared-history"
    assert settings.resolved_history_dir() == tmp_path / "shared-history"


# ---------------------------------------------------------------------------
# Enumerated settings: a typo must fail loudly rather than silently degrading
# (TIBLENDER_MODE=ofline used to fall through to the `auto` path, which makes
# live network calls in what the operator believed was an offline run).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["offline", "OFFLINE", " live ", "auto"])
def test_mode_accepts_the_documented_values_case_insensitively(monkeypatch, value):
    monkeypatch.setenv("TIBLENDER_MODE", value)

    assert Settings.from_env().mode == value.strip().lower()


def test_invalid_mode_raises_value_error_naming_value_and_allowed_set(monkeypatch):
    monkeypatch.setenv("TIBLENDER_MODE", "ofline")

    with pytest.raises(ValueError) as excinfo:
        Settings.from_env()

    message = str(excinfo.value)
    assert "ofline" in message
    assert "TIBLENDER_MODE" in message
    for allowed in ("auto", "live", "offline"):
        assert allowed in message


def test_invalid_clock_mode_raises_value_error(monkeypatch):
    monkeypatch.setenv("TIBLENDER_CLOCK", "frozen")

    with pytest.raises(ValueError) as excinfo:
        Settings.from_env()

    message = str(excinfo.value)
    assert "frozen" in message
    assert "fixed" in message and "wall" in message and "auto" in message


@pytest.mark.parametrize("value", ["auto", "fixed", "WALL"])
def test_clock_mode_accepts_the_documented_values(monkeypatch, value):
    monkeypatch.setenv("TIBLENDER_CLOCK", value)

    assert Settings.from_env().clock_mode == value.lower()


def test_invalid_rl_policy_backend_raises_value_error(monkeypatch):
    monkeypatch.setenv("TIBLENDER_RL_POLICY_BACKEND", "torch_dqn")

    with pytest.raises(ValueError) as excinfo:
        Settings.from_env()

    message = str(excinfo.value)
    assert "torch_dqn" in message
    assert "multiaction_dqn" in message and "disabled" in message


@pytest.mark.parametrize("value", ["multiaction_dqn", "disabled"])
def test_rl_policy_backend_accepts_the_documented_values(monkeypatch, value):
    monkeypatch.setenv("TIBLENDER_RL_POLICY_BACKEND", value)

    assert Settings.from_env().rl_policy_backend == value


@pytest.mark.parametrize(
    "name,default",
    [("TIBLENDER_MODE", "auto"), ("TIBLENDER_CLOCK", "auto"), ("TIBLENDER_RL_POLICY_BACKEND", "multiaction_dqn")],
)
def test_empty_enumerated_value_falls_back_to_the_default(monkeypatch, name, default):
    monkeypatch.setenv(name, "   ")

    settings = Settings.from_env()

    assert getattr(settings, {"TIBLENDER_MODE": "mode", "TIBLENDER_CLOCK": "clock_mode", "TIBLENDER_RL_POLICY_BACKEND": "rl_policy_backend"}[name]) == default




def test_trend_volume_gates_default_to_the_deployed_values(monkeypatch):
    """The deployed Layer 4 detector requires at least four similar historical
    snapshots and at least four posts in the current cluster before it will
    call a volume anomaly; the artifact keeps both gates."""
    monkeypatch.delenv("TIBLENDER_TREND_MIN_SIMILAR_SNAPSHOTS", raising=False)
    monkeypatch.delenv("TIBLENDER_TREND_VOLUME_MIN_POSTS", raising=False)

    settings = Settings.from_env()

    assert settings.trend_min_similar_snapshots == 4
    assert settings.trend_volume_min_posts == 4
    # k stays at the paper's 1.5 (the deployed env default is 1.2).
    assert settings.volume_sigma_k == 1.5


def test_tool_max_bytes_defaults_to_two_mib_and_is_env_overridable(monkeypatch):
    """The JSON read cap is a setting, not a literal.

    A fixed 200 KB cap truncated NVD's response for a widely-referenced CVE
    mid-string, so the lookup failed with `invalid JSON response` and the
    recorded response for that CVE shipped as an `error`.
    """
    monkeypatch.delenv("TIBLENDER_TOOL_MAX_BYTES", raising=False)
    assert Settings.from_env().tool_max_bytes == 2 * 1024 * 1024

    monkeypatch.setenv("TIBLENDER_TOOL_MAX_BYTES", "4194304")
    assert Settings.from_env().tool_max_bytes == 4194304


def test_tool_max_bytes_is_the_cap_the_json_helper_actually_reads(monkeypatch):
    """The setting must reach the socket read, not just the dataclass."""
    from dataclasses import replace as dataclass_replace

    from tiblender_runtime import tools as tools_module
    from tiblender_runtime.models import ToolQuery

    seen: dict[str, int] = {}

    def fake_http_json(url, *, headers=None, timeout=30, max_bytes=None):
        seen["max_bytes"] = max_bytes
        return {"vulnerabilities": [{"cve": {"id": "CVE-2026-33186", "descriptions": []}}]}, url

    settings = dataclass_replace(
        Settings(allow_heuristic_reports=True, enable_real_tools=True, mode="live"),
        tool_max_bytes=1_234_567,
    )
    monkeypatch.setattr(tools_module, "_http_json", fake_http_json)
    runner = tools_module.ToolRunner(settings, services={})
    runner._nvd_cve_lookup(ToolQuery("Vulnerability", "NVD_CVE_LOOKUP", {"cve": "CVE-2026-33186"}))
    assert seen["max_bytes"] == 1_234_567


def test_pivot_max_resolutions_per_ip_default_and_env_override(monkeypatch):
    """The shared-hosting threshold is configurable and defaults to 5.

    Above it a resolution answer describes co-tenancy, not the campaign, so
    the co-resident names are dropped instead of shipped as indicators.
    """
    from tiblender_runtime.settings import Settings

    monkeypatch.delenv("TIBLENDER_PIVOT_MAX_RESOLUTIONS_PER_IP", raising=False)
    assert Settings.from_env().pivot_max_resolutions_per_ip == 5
    monkeypatch.setenv("TIBLENDER_PIVOT_MAX_RESOLUTIONS_PER_IP", "12")
    assert Settings.from_env().pivot_max_resolutions_per_ip == 12

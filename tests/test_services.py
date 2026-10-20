from pathlib import Path

from tiblender_runtime.services import resolve_services, doctor_report
from tiblender_runtime.settings import Settings


def _fixtures(tmp_path: Path) -> Path:
    for platform in ("x", "reddit", "telegram", "discord"):
        (tmp_path / "collectors" / platform).mkdir(parents=True)
        (tmp_path / "collectors" / platform / "cycle_001.json").write_text("{}", encoding="utf-8")
    return tmp_path


def test_offline_mode_uses_fixtures_and_heuristics(tmp_path):
    settings = Settings(mode="offline", fixtures_dir=_fixtures(tmp_path), x_bearer_token="secret")
    services = resolve_services(settings)
    assert services["x"].mode == "fixture"          # credentials ignored offline
    assert services["llm"].mode == "heuristic"
    assert services["passive_dns"].mode == "fixture"


def test_live_mode_never_falls_back_to_fixture(tmp_path):
    settings = Settings(mode="live", fixtures_dir=_fixtures(tmp_path), llm_provider="heuristic")
    services = resolve_services(settings)
    assert services["x"].mode == "unavailable"
    assert services["reddit"].mode == "unavailable"
    assert services["llm"].mode == "unavailable"
    assert services["whois"].mode == "live"


def test_auto_mode_prefers_live_when_credentials_present(tmp_path):
    settings = Settings(mode="auto", fixtures_dir=_fixtures(tmp_path), discord_bot_token="tok", reddit_client_id="a", reddit_client_secret="b")
    services = resolve_services(settings)
    assert services["discord"].mode == "live"
    assert services["reddit"].mode == "live"
    assert services["x"].mode == "fixture"
    assert services["telegram"].mode in {"fixture", "live"}


def test_doctor_report_masks_nothing_sensitive_and_lists_fixtures(tmp_path):
    settings = Settings(mode="offline", fixtures_dir=_fixtures(tmp_path), openai_api_key="sk-secret")
    report = doctor_report(settings)
    assert "sk-secret" not in str(report)
    assert report["fixtures_present"]["x"] == ["cycle_001.json"]
    assert {item["name"] for item in report["services"]} >= {"llm", "x", "reddit", "telegram", "discord", "passive_dns"}


def test_offline_mode_forces_deterministic_embedding_and_disables_live_ocr(tmp_path):
    # Even with the optional dependencies importable and a reference backend
    # requested, an offline run must take the deterministic path: model
    # weights, library versions, and OCR binaries would otherwise leak into
    # supposedly reproducible output.
    settings = Settings(
        mode="offline",
        fixtures_dir=_fixtures(tmp_path),
        embedding_backend="bge_large",
        ocr_enabled=True,
    )

    services = resolve_services(settings)

    assert services["embedding"].mode == "heuristic"
    assert "offline" in services["embedding"].reason
    assert services["ocr"].mode == "unavailable"
    assert "offline" in services["ocr"].reason


def test_non_offline_mode_still_resolves_ocr_and_embedding_from_the_environment(tmp_path):
    settings = Settings(mode="auto", fixtures_dir=_fixtures(tmp_path), ocr_enabled=False)

    services = resolve_services(settings)

    assert services["embedding"].mode == "heuristic"  # default hashing backend
    assert services["ocr"].mode == "unavailable"
    assert "TIBLENDER_OCR_ENABLED" in services["ocr"].reason


def test_resolved_embedding_backend_pins_hashing_offline_only():
    from tiblender_runtime.services import resolved_embedding_backend

    assert resolved_embedding_backend(Settings(mode="offline"), "bge_large") == "hashing"
    assert resolved_embedding_backend(Settings(mode="auto"), "bge_large") == "bge_large"
    assert resolved_embedding_backend(Settings(mode="live"), "minilm") == "minilm"


def test_passive_dns_is_live_only_with_a_virustotal_key():
    with_key = resolve_services(Settings(mode="live", virustotal_api_key="k"))["passive_dns"]
    without_key = resolve_services(Settings(mode="live"))["passive_dns"]

    assert with_key.mode == "live" and "virustotal" in with_key.reason
    assert without_key.mode == "unavailable" and "VIRUSTOTAL_API_KEY" in without_key.reason


def test_a_discord_user_token_alone_unlocks_live_collection(tmp_path):
    settings = Settings(mode="auto", fixtures_dir=_fixtures(tmp_path), discord_user_token="usr")
    assert resolve_services(settings)["discord"].mode == "live"


def test_no_discord_credential_falls_back_to_fixtures(tmp_path):
    settings = Settings(mode="auto", fixtures_dir=_fixtures(tmp_path))
    assert resolve_services(settings)["discord"].mode == "fixture"


# ---------------------------------------------------------------------------
# `doctor` is the command the docs point readers at to see "what actually
# got wired up", so it must not report the search engine as live purely
# because SEARXNG_BASE_URL has a (default) value: the compose alias
# http://searxng:8080 does not resolve unless `--profile live` started the
# service. The reachability probe lives in doctor_report only -- resolve_services
# stays a pure configuration resolution with no network access.
# ---------------------------------------------------------------------------


def _search_engine_row(report: dict) -> dict:
    return next(row for row in report["services"] if row["name"] == "search_engine")


def test_doctor_reports_search_engine_unavailable_when_searxng_is_unreachable():
    settings = Settings(mode="live", searxng_base_url="http://searxng:8080")

    row = _search_engine_row(doctor_report(settings, probe=lambda url: False))

    assert row["mode"] == "unavailable"
    assert "not reachable" in row["reason"]
    assert "http://searxng:8080" in row["reason"]


def test_doctor_confirms_a_reachable_search_engine():
    settings = Settings(mode="live", searxng_base_url="http://searxng:8080")

    row = _search_engine_row(doctor_report(settings, probe=lambda url: True))

    assert row["mode"] == "live"
    assert "reachable" in row["reason"]


def test_doctor_never_probes_the_network_in_offline_mode(tmp_path):
    probed: list[str] = []

    def _probe(url: str) -> bool:
        probed.append(url)
        return True

    report = doctor_report(Settings(mode="offline", fixtures_dir=_fixtures(tmp_path)), probe=_probe)

    assert probed == [], "an offline doctor run must make no outbound connection"
    assert _search_engine_row(report)["mode"] == "fixture"


def test_resolve_services_itself_never_probes_the_network():
    # run_one_cycle/_write_run_summary call resolve_services on every cycle;
    # a probe there would put a network call (and its latency) in the pipeline.
    services = resolve_services(Settings(mode="live", searxng_base_url="http://searxng:8080"))

    assert services["search_engine"].mode == "live"

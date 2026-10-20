"""Report/post selection, cycle windowing, and noise assignment for
`tools/build_fixtures.py`.

Kept separate from the CLI/payload-assembly code in `build_fixtures.py` so
each module stays focused: this one turns released-dataset records into
`PreparedPost` objects tagged with a cycle index; `build_fixtures.py` turns
those into platform payloads.

The single source of fixture material is the released dataset's
`raw_dataset/reports.jsonl` (8,288 reports). Every fixture post is an
anonymized excerpt of a `source_posts.items[]` entry of one of those reports
and keeps that entry's `source_ref`, so any fixture post can be looked up in
the dataset.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

from tiblender_runtime.iocs import VALID_TLDS, refang
from tiblender_runtime.legitimate_hosts import REFERENCE_HOSTS

from tools.anonymize import anonymize_text, pseudonym
from tools.fixture_shapes import PreparedPost

ALL_PLATFORMS = ("x", "reddit", "telegram", "discord")

# The four paper categories (`paper_category` in the released schema). The
# bundle must show all four so the demo exercises every category-specific
# path (keyword vocabulary, severity rules, report templates).
ALL_CATEGORIES = ("APT/Ransomware", "Malware/C2", "Phishing/Fraud", "Vulnerability")

# Selection shape. The demo replays three *consecutive* collection windows of
# the deployment, so the builder first picks one contiguous band of
# `cycles * CYCLE_DAYS` days out of the 31-day deployment and then splits it
# into equal, back-to-back cycle windows. Fixed-width windows (rather than
# chunking whatever dates happen to be present) keep the three windows
# contiguous and increasing by construction, which is what the collectors'
# incremental watermarks assume.
CYCLE_DAYS = 4

MIN_POSTS_PER_REPORT = 3
# Every platform must contribute at least this many posts to every cycle. Two
# rather than one: the Discord fixture transport is read incrementally with an
# `after` watermark, and a single-message cycle cannot demonstrate that the
# watermark actually advances (tests/test_fixture_bundle.py pins this).
MIN_POSTS_PER_PLATFORM = 2
# Cap per report so a single very large cluster cannot dominate a cycle; the
# earliest `MAX_POSTS_PER_REPORT` in-window posts are kept.
MAX_POSTS_PER_REPORT = 8

# Per-cycle primary-post targets. The ramp keeps the bundle's volume growing
# across the three cycles (31 / 32 / 42 posts including noise on the current
# dataset) so the quoted per-cycle numbers, the trend statistics that depend
# on volume growth, and the >=90-post bundle invariant all stay in the same
# range.
TARGET_PRIMARY_POSTS = (11, 26, 34)

MAX_NOISE_PER_CYCLE = 6
PSEUDONYM_SALT = "tiblender-fixture"

# Same ASCII-only boundary rationale as `tools/anonymize.py`: Python's `\b`
# treats CJK characters as word characters, so a plain `\b` would silently
# fail to match one of these tokens when it directly abuts Japanese text
# (common in this corpus) -- which would wrongly let a post with an IoC
# through this noise-eligibility check.
_LEFT_BOUND = r"(?<![A-Za-z0-9_])"
_RIGHT_BOUND = r"(?![A-Za-z0-9_])"

_CVE_RE = re.compile(_LEFT_BOUND + r"CVE-\d{4}-\d{4,8}" + _RIGHT_BOUND, re.IGNORECASE)
_HASH_RE = re.compile(
    _LEFT_BOUND + r"[a-fA-F0-9]{64}" + _RIGHT_BOUND
    + "|" + _LEFT_BOUND + r"[a-fA-F0-9]{40}" + _RIGHT_BOUND
    + "|" + _LEFT_BOUND + r"[a-fA-F0-9]{32}" + _RIGHT_BOUND
)
_IP_RE = re.compile(_LEFT_BOUND + r"(?:\d{1,3}\.){3}\d{1,3}" + _RIGHT_BOUND)
_DOMAIN_RE = re.compile(_LEFT_BOUND + r"(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,24}" + _RIGHT_BOUND)

# Named zones that appear in the released timestamps but that
# `datetime.fromisoformat` cannot parse. Everything else in the dataset is
# either offset-aware ISO-8601 or naive (treated as UTC).
_NAMED_UTC_OFFSETS = {"JST": timedelta(hours=9), "UTC": timedelta(0), "GMT": timedelta(0)}
_NAMED_ZONE_RE = re.compile(r"\s([A-Z]{3})$")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def parse_dt(value: str) -> datetime:
    """Parse a released `created_at` / `source_posts[].timestamp` into UTC.

    The released dataset carries several shapes: offset-aware ISO-8601 (with
    or without fractional seconds), naive ISO-8601, `YYYY-MM-DD HH:MM`, and a
    handful with a trailing named zone (`... JST`). Naive values are read as
    UTC; a named zone is applied as its fixed offset.
    """
    text = str(value).strip().replace("Z", "+00:00")
    offset = timedelta(0)
    match = _NAMED_ZONE_RE.search(text)
    if match and match.group(1) in _NAMED_UTC_OFFSETS:
        offset = _NAMED_UTC_OFFSETS[match.group(1)]
        text = text[: match.start()].strip()
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = (parsed - offset).replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def text_has_ioc(text: str) -> bool:
    """Conservative, urlparse-free IoC check used to keep noise posts clean.

    Mirrors `tiblender_runtime.iocs.extract_iocs`'s CVE/hash/IP/domain rules
    without going through its URL parsing path, which raises on some
    defanged/malformed URL text found in the raw corpus.
    """
    refanged = refang(text)
    if _CVE_RE.search(refanged) or _HASH_RE.search(refanged) or _IP_RE.search(refanged):
        return True
    for match in _DOMAIN_RE.finditer(refanged):
        domain = match.group(0).lower().strip(".")
        if domain.rsplit(".", 1)[-1] not in VALID_TLDS:
            continue
        # REFERENCE_HOSTS, not the runtime's wider LEGITIMATE_DOMAINS: post
        # selection defines the committed fixture bundle and must not move
        # when the IoC allowlist grows.
        if domain in REFERENCE_HOSTS or any(domain.endswith(f".{legit}") for legit in REFERENCE_HOSTS):
            continue
        return True
    return False


@dataclass(frozen=True, slots=True)
class DatasetPost:
    """One dated `source_posts.items[]` entry of a released report."""

    source_ref: str
    platform: str
    posted_at: datetime
    text: str


@dataclass(slots=True)
class DatasetReport:
    """One released report, reduced to what fixture selection needs."""

    report_id: str
    created_at: datetime | None
    category: str
    severity: str
    indicators: tuple[str, ...]
    posts: tuple[DatasetPost, ...]

    @property
    def sort_created_at(self) -> datetime:
        return self.created_at or datetime.max.replace(tzinfo=UTC)


# The released corpus is multilingual: a quarter of the released reports carry
# at least one non-English source post (Japanese, Korean, Russian, Chinese,
# Arabic). Those posts are eligible fixture material, because the demo should
# exercise the multilingual path the paper describes: the runtime's language
# guard composes every report_text section in English whatever language the
# posts were written in (tiblender_runtime/language.py), and the post body is
# published as source evidence in the language it was posted in -- which is
# what the released dataset does with `source_posts.items[].text`.


def _post_text(item: dict) -> str:
    """Fixture text for a released source post.

    The release carries the post body in its original language under `text`.
    """
    return str(item.get("text") or "").strip()


def is_fixture_eligible_text(text: str) -> bool:
    """Whether a released post's text may become fixture material.

    Any non-empty post body, in any language -- see the note above.
    """
    return bool(text)


def load_dataset_reports(path: Path) -> list[DatasetReport]:
    """Load `raw_dataset/reports.jsonl` into `DatasetReport`s.

    Only *dated* posts on the four collected platforms with non-empty text
    are kept: an undated post cannot be placed in a collection window, and a
    post from an unknown platform has no fixture transport. Language is not a
    criterion -- see `is_fixture_eligible_text`.
    """
    reports: list[DatasetReport] = []
    for record in iter_jsonl(path):
        posts: list[DatasetPost] = []
        for item in ((record.get("source_posts") or {}).get("items") or []):
            timestamp = item.get("timestamp")
            platform = item.get("platform")
            text = _post_text(item)
            source_ref = str(item.get("source_ref") or "")
            if not timestamp or platform not in ALL_PLATFORMS or not source_ref:
                continue
            if not is_fixture_eligible_text(text):
                continue
            try:
                posted_at = parse_dt(timestamp)
            except ValueError:
                continue
            posts.append(
                DatasetPost(source_ref=source_ref, platform=platform, posted_at=posted_at, text=text)
            )
        if not posts:
            continue
        indicators = [str(entry.get("value", "")) for entry in ((record.get("iocs") or {}).get("items") or [])]
        indicators += [str(cve) for cve in ((record.get("report_text") or {}).get("cves") or [])]
        created_at = None
        if record.get("created_at"):
            try:
                created_at = parse_dt(record["created_at"])
            except ValueError:
                created_at = None
        reports.append(
            DatasetReport(
                report_id=str(record["report_id"]),
                created_at=created_at,
                category=str(record.get("paper_category", "")),
                severity=str(record.get("severity", "")),
                indicators=tuple(sorted({value for value in indicators if value})),
                posts=tuple(sorted(posts, key=lambda post: (post.posted_at, post.source_ref))),
            )
        )
    return reports


# ---------------------------------------------------------------------------
# Collection band and cycle windows
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Window:
    start: datetime
    end: datetime
    cycle_index: int
    label: str

    def contains(self, moment: datetime) -> bool:
        return self.start <= moment <= self.end


def _has_indicator(indicators: tuple[str, ...], texts: list[str]) -> bool:
    if not indicators:
        return False
    concatenated = refang(" ".join(texts))
    for value in indicators:
        refanged = refang(value)
        if refanged and refanged in concatenated:
            return True
    return False


def choose_band_start(reports: list[DatasetReport], cycles: int) -> date:
    """The first day of the contiguous collection band the demo replays.

    The band is `cycles * CYCLE_DAYS` days long. Its start is the day that
    maximizes how many released reports are *bandable* -- have at least
    `MIN_POSTS_PER_REPORT` dated posts inside the band and carry an indicator
    (IoC value or CVE) verbatim in their post text -- with ties broken by the
    earliest day, so the choice is a deterministic function of the dataset
    file alone. Picking the densest band is what lets the bundle show four
    platforms, all four categories, and cross-platform reports in every one of
    the three windows, drawn from the released reports themselves.
    """
    band_days = cycles * CYCLE_DAYS
    ioc_bearing = [report for report in reports if _has_indicator(report.indicators, [post.text for post in report.posts])]
    days = sorted({post.posted_at.date() for report in ioc_bearing for post in report.posts})
    if not days:
        raise ValueError("no dated posts in the dataset; cannot choose a collection band")

    best_start = days[0]
    best_count = -1
    for start in days:
        end = start + timedelta(days=band_days - 1)
        count = 0
        for report in ioc_bearing:
            in_band = sum(1 for post in report.posts if start <= post.posted_at.date() <= end)
            if in_band >= MIN_POSTS_PER_REPORT:
                count += 1
        if count > best_count:
            best_count = count
            best_start = start
    return best_start


def compute_windows(band_start: date, cycles: int) -> list[Window]:
    """`cycles` equal, back-to-back `CYCLE_DAYS`-day windows from `band_start`."""
    windows: list[Window] = []
    for index in range(cycles):
        first_day = band_start + timedelta(days=CYCLE_DAYS * index)
        last_day = first_day + timedelta(days=CYCLE_DAYS - 1)
        windows.append(
            Window(
                start=datetime.combine(first_day, time.min, tzinfo=UTC),
                end=datetime.combine(last_day, time(23, 59, 59), tzinfo=UTC),
                cycle_index=index + 1,
                label=f"cycle_{index + 1:03d}",
            )
        )
    return windows


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Candidate:
    report_id: str
    created_at: datetime
    category: str
    severity: str
    cycle_index: int
    platforms: frozenset[str]
    posts: tuple[DatasetPost, ...]

    @property
    def sort_key(self) -> tuple:
        # Prefer cross-platform reports (they are what makes the demo show
        # cross-platform clustering at all), then chronological, then by id
        # so the order never depends on dict/file ordering.
        return (-len(self.platforms), self.created_at, self.report_id)


def select_candidates(reports: list[DatasetReport], windows: list[Window]) -> list[Candidate]:
    """Every released report that can supply one cycle's worth of fixture posts.

    A report is attached to exactly one cycle -- the window holding most of
    its dated posts (earliest window on a tie) -- and contributes only the
    posts inside that window, capped at `MAX_POSTS_PER_REPORT`. It qualifies
    when at least `MIN_POSTS_PER_REPORT` posts remain and at least one of the
    report's indicators (an `iocs.items[].value` or a `report_text.cves`
    entry) appears verbatim, after refanging, in the retained post text -- so
    every selected report really does carry an IoC or CVE into the cycle it
    is placed in.
    """
    candidates: list[Candidate] = []
    for report in reports:
        by_cycle: dict[int, list[DatasetPost]] = {}
        for post in report.posts:
            for window in windows:
                if window.contains(post.posted_at):
                    by_cycle.setdefault(window.cycle_index, []).append(post)
                    break
        if not by_cycle:
            continue
        cycle_index = max(by_cycle, key=lambda index: (len(by_cycle[index]), -index))
        posts = sorted(by_cycle[cycle_index], key=lambda post: (post.posted_at, post.source_ref))
        posts = posts[:MAX_POSTS_PER_REPORT]
        if len(posts) < MIN_POSTS_PER_REPORT:
            continue
        if not _has_indicator(report.indicators, [post.text for post in posts]):
            continue
        candidates.append(
            Candidate(
                report_id=report.report_id,
                created_at=report.sort_created_at,
                category=report.category,
                severity=report.severity,
                cycle_index=cycle_index,
                platforms=frozenset(post.platform for post in posts),
                posts=tuple(posts),
            )
        )
    candidates.sort(key=lambda candidate: (candidate.cycle_index, *candidate.sort_key))
    return candidates


def grow_selection(candidates: list[Candidate], windows: list[Window]) -> list[Candidate]:
    """Pick the reports each cycle ships, deterministically.

    Per cycle, in the candidate sort order (cross-platform first, then
    chronological, then by report id):

    1. **Platform coverage.** For each of `x`, `reddit`, `telegram`,
       `discord` in turn, keep taking the first candidate that covers it
       until the cycle holds at least `MIN_POSTS_PER_PLATFORM` posts on that
       platform. This is what guarantees every platform appears in every
       cycle from the selected reports, with no out-of-window filler.
    2. **Volume.** Keep adding candidates in order until the cycle holds at
       least `TARGET_PRIMARY_POSTS[cycle - 1]` posts.

    Finally, bundle-wide **category coverage**: for any of the four paper
    categories still missing, the first candidate of that category (in cycle
    order, then candidate order) is added to its own cycle.
    """
    selected: list[Candidate] = []
    selected_ids: set[str] = set()

    for window in windows:
        cycle_candidates = [c for c in candidates if c.cycle_index == window.cycle_index]
        picked: list[Candidate] = []

        def platform_count(platform: str) -> int:
            return sum(
                1 for candidate in picked for post in candidate.posts if post.platform == platform
            )

        for platform in ALL_PLATFORMS:
            while platform_count(platform) < MIN_POSTS_PER_PLATFORM:
                for candidate in cycle_candidates:
                    if candidate.report_id in selected_ids or platform not in candidate.platforms:
                        continue
                    picked.append(candidate)
                    selected_ids.add(candidate.report_id)
                    break
                else:
                    # No candidate left for this platform in this cycle; the
                    # bundle checks will flag it rather than let it pass.
                    break

        target = TARGET_PRIMARY_POSTS[window.cycle_index - 1] if window.cycle_index - 1 < len(TARGET_PRIMARY_POSTS) else TARGET_PRIMARY_POSTS[-1]
        total = sum(len(candidate.posts) for candidate in picked)
        for candidate in cycle_candidates:
            if total >= target:
                break
            if candidate.report_id in selected_ids:
                continue
            picked.append(candidate)
            selected_ids.add(candidate.report_id)
            total += len(candidate.posts)

        selected.extend(picked)

    covered_categories = {candidate.category for candidate in selected}
    for category in ALL_CATEGORIES:
        if category in covered_categories:
            continue
        for candidate in candidates:
            if candidate.report_id in selected_ids or candidate.category != category:
                continue
            selected.append(candidate)
            selected_ids.add(candidate.report_id)
            covered_categories.add(category)
            break

    selected.sort(key=lambda candidate: (candidate.cycle_index, *candidate.sort_key))
    return selected


# ---------------------------------------------------------------------------
# Preparing posts (anonymize + assign ids/pseudonyms)
# ---------------------------------------------------------------------------


def prepare_primary_posts(selected: list[Candidate]) -> list[PreparedPost]:
    prepared: list[PreparedPost] = []
    for candidate in selected:
        for post in candidate.posts:
            prepared.append(
                PreparedPost(
                    platform=post.platform,
                    post_ref=post.source_ref,
                    report_id=candidate.report_id,
                    category=candidate.category,
                    kind="primary",
                    cycle_index=candidate.cycle_index,
                    text=anonymize_text(post.text),
                    posted_at=post.posted_at,
                    author=pseudonym(post.source_ref, salt=PSEUDONYM_SALT),
                )
            )
    return prepared


def counts_by_cycle_platform(posts: list[PreparedPost]) -> dict[tuple[int, str], int]:
    counts: dict[tuple[int, str], int] = {}
    for post in posts:
        key = (post.cycle_index, post.platform)
        counts[key] = counts.get(key, 0) + 1
    return counts


def build_noise_pool(
    reports: list[DatasetReport],
    selected_report_ids: set[str],
    windows: list[Window],
) -> dict[int, list[tuple[DatasetReport, DatasetPost, int]]]:
    """Low-signal, indicator-free posts of *other* released reports per cycle.

    Noise makes each cycle look like a real collection window rather than a
    curated cluster set. It is drawn from reports that were not selected, and
    only from posts that carry no IoC at all, so noise can never add an
    indicator (and therefore never a tool query or a recording) of its own.
    """
    pool: dict[int, list[tuple[DatasetReport, DatasetPost, int]]] = {
        window.cycle_index: [] for window in windows
    }
    for report in reports:
        if report.report_id in selected_report_ids:
            continue
        for post in report.posts:
            for window in windows:
                if not window.contains(post.posted_at):
                    continue
                if text_has_ioc(post.text):
                    break
                pool[window.cycle_index].append((report, post, window.cycle_index))
                break
    for cycle_posts in pool.values():
        cycle_posts.sort(key=lambda entry: entry[1].source_ref)
    return pool


def assign_noise(
    noise_pool: dict[int, list[tuple[DatasetReport, DatasetPost, int]]],
    primary_counts: dict[tuple[int, str], int],
    windows: list[Window],
    rng: random.Random,
) -> list[PreparedPost]:
    prepared: list[PreparedPost] = []
    for window in windows:
        cycle = window.cycle_index
        pool = list(noise_pool.get(cycle, []))
        chosen: list[tuple[DatasetReport, DatasetPost, int]] = []
        chosen_refs: set[str] = set()

        # Platform top-up first, so it survives the MAX_NOISE_PER_CYCLE cap.
        # Selection covers every platform from real IoC-bearing reports where
        # it can, but a window can simply not hold two Discord reports that
        # also carry an indicator; noise -- real, indicator-free posts from
        # the same window -- makes up the difference so every platform reaches
        # MIN_POSTS_PER_PLATFORM in every cycle.
        for platform in ALL_PLATFORMS:
            platform_pool = [entry for entry in pool if entry[1].platform == platform]
            while (
                primary_counts.get((cycle, platform), 0)
                + sum(1 for entry in chosen if entry[1].platform == platform)
                < MIN_POSTS_PER_PLATFORM
            ):
                remaining = [entry for entry in platform_pool if entry[1].source_ref not in chosen_refs]
                if not remaining:
                    break
                pick = rng.choice(remaining)
                chosen.append(pick)
                chosen_refs.add(pick[1].source_ref)

        remaining_pool = [entry for entry in pool if entry[1].source_ref not in chosen_refs]
        rng.shuffle(remaining_pool)
        for entry in remaining_pool:
            if len(chosen) >= MAX_NOISE_PER_CYCLE:
                break
            chosen.append(entry)

        for report, post, cycle_index in chosen[:MAX_NOISE_PER_CYCLE]:
            prepared.append(
                PreparedPost(
                    platform=post.platform,
                    post_ref=post.source_ref,
                    report_id=report.report_id,
                    category=report.category,
                    kind="noise",
                    cycle_index=cycle_index,
                    text=anonymize_text(post.text),
                    posted_at=post.posted_at,
                    author=pseudonym(post.source_ref, salt=PSEUDONYM_SALT),
                )
            )
    return prepared

"""Per-cycle accounting of LLM stage fallbacks.

Every LLM-backed stage may fail for a single unit (one chunk of posts, one
batch of candidate clusters, one cluster, one perspective, one judge round,
one report, one trend signal) without aborting the cycle: the stage retries
once and then uses its deterministic path. Those fallbacks must stay visible,
so each one is recorded here and surfaces three ways:

* aggregated per-stage counters in ``cycle_summary.json`` under
  ``llm_fallbacks``;
* a histogram of *why* they happened in ``cycle_summary.json`` under
  ``llm_fallback_reasons``, bucketed by a category this code attaches where
  the failure is raised -- never by pattern-matching the failure text, which
  carries provider output and model summaries of scraped posts;
* per-report stage names in ``metadata.llm_stage_fallbacks``, which also
  downgrades that report's ``metadata.generation_mode`` to ``llm_partial``.

**Attribution is per report, never per cycle.** A fallback is recorded with
the scope it actually damaged -- the cluster ids and/or the post ids of the
failing unit -- and a report is ``llm_partial`` only when one of those scopes
intersects its own cluster or posts. Batched stages (post normalization,
cluster refinement) send several units per request, so they record every
cluster/post in the failing batch and leave the batches that succeeded alone.
Cluster refinement used to be treated as cycle-global, which marked every
report of a cycle ``llm_partial`` after a single failing batch.

Some LLM answers are only *partially* valid rather than a failure -- for
example, the perspective focus-selection agent returning one valid focus
type instead of the two required. Padding that up with the deterministic
ranking is not a fallback: it is still counted (``perspective_focus_padded``
alongside the ``llm_fallbacks`` counters) but never attributed to a report's
``llm_stage_fallbacks`` / ``generation_mode`` (see ``record_padding``).

Every fallback also logs exactly one WARNING, built by
:func:`format_fallback_line`, and that line is content-free by construction:
the stage, the category, the counts, the affected cluster ids and a bounded
phrase this code wrote. The provider's and the model's own words go to DEBUG
instead, so a WARNING-level log can be read and shared without leaking
collected post text. Call sites that want to say more about a fallback build
their line with the same formatter and send the raw failure to
``logger.debug``; no WARNING or ERROR anywhere in the package formats an
``LLMGenerationError`` (see
``tests/test_llm_fallback_reasons.py``'s poisoned-draft test).

The tracker is a process-level singleton reset at the start of every cycle
(``reset_fallbacks``). Perspective agents run in a thread pool, so all
mutation is guarded by a lock; every accessor returns a fresh copy so callers
can never mutate the recorded state.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable

logger = logging.getLogger(__name__)

# Counter keys are the public shape of cycle_summary.json's llm_fallbacks
# block; they are always present, even when zero, so a consumer can tell
# "no fallback happened" from "this runtime did not record fallbacks".
FALLBACK_KEYS: tuple[str, ...] = (
    "normalization_posts",
    "cluster_refinement_batches",
    "planner_clusters",
    "perspective_units",
    "judge_rounds",
    "report_fields",
    "dynamic_query_signals",
    "strategic_value_clusters",
)

# Stage name recorded on a report whose cluster was affected by that key.
STAGE_BY_KEY: dict[str, str] = {
    "normalization_posts": "post_normalization",
    "cluster_refinement_batches": "cluster_refinement",
    "planner_clusters": "investigation_planning",
    "perspective_units": "perspective_queries",
    "judge_rounds": "evaluation_judges",
    "report_fields": "report_fields",
    "dynamic_query_signals": "dynamic_queries",
    "strategic_value_clusters": "strategic_value",
}

# Reason categories for cycle_summary.json's ``llm_fallback_reasons``
# histogram. Like FALLBACK_KEYS these are always present, even when zero.
#
# A category is **authored by this code at the point the failure is raised**
# (``LLMGenerationError.category``), never inferred by scanning the failure
# text. That text routinely ends with material this code did not write -- up
# to 500 characters of the model's own draft, up to 2000 characters of an HTTP
# error body, or a whole provider response -- and the model's draft is a
# summary of scraped, adversary-authored posts. Matching markers against it
# would let a collected post move a counter in a histogram this artifact
# publishes (a post mentioning "timed out" inside a JSON-parse failure would
# be counted as a timeout).
REASON_KEYS: tuple[str, ...] = (
    "truncated",
    "timeout",
    "http_status",
    "network",
    "empty_content",
    "json_parse",
    "invalid_response",
    # The model answered in the language of the source posts instead of
    # English. Authored by the Step 5 caller after the language guard
    # (language.failing_report_fields) rejects a draft, never by scanning the
    # draft for words -- the draft is a summary of scraped posts.
    "non_english_output",
    "unknown",
)

# Bookkeeping counters that live in the same ``llm_fallbacks`` block as the
# per-stage counters but are NOT stage counters, so they are excluded from the
# "reasons total equals fallbacks total" invariant:
#
# * ``perspective_focus_padded`` -- a padding event, not a fallback, so it has
#   no failure reason to bucket (see ``record_padding``);
# * ``unattributed_fallbacks`` -- how many of the fallbacks counted above
#   carried no cluster and no post scope, so they could not reach any report's
#   ``generation_mode``. It double-counts by design: those events are already
#   in their stage counter and in the reason histogram; this counter only says
#   how many of them are invisible per report.
PAD_KEYS: tuple[str, ...] = ("perspective_focus_padded",)
UNATTRIBUTED_KEY = "unattributed_fallbacks"

# Longest code-authored summary the WARNING line will print. The summary is
# always written by this module's callers; the provider's/model's own text
# goes to DEBUG instead (see ``record``).
SUMMARY_MAX_CHARS = 120

# Stages whose fallbacks are recorded before Step 5 builds the reports, so an
# unattributed fallback there is a bug: nothing will ever surface it on a
# report. The two feedback-stage keys run after report generation and
# legitimately never reach one.
_PRE_REPORT_KEYS = frozenset(FALLBACK_KEYS) - {"dynamic_query_signals", "strategic_value_clusters"}


def category_of(error: BaseException | None) -> str:
    """The category this code attached where ``error`` was raised.

    Anything without one -- a third-party exception, or a raise site that
    forgot -- is ``"unknown"`` rather than guessed from its message.
    """
    value = getattr(error, "category", "")
    return value if value in REASON_KEYS else "unknown"


def _clean_category(category: str) -> str:
    return category if category in REASON_KEYS else "unknown"


def format_fallback_line(
    stage: str,
    *,
    category: str,
    count: int = 1,
    cluster_ids: Iterable[str] = (),
    post_count: int = 0,
    summary: str = "",
    attributed: bool | None = None,
) -> str:
    """The one content-free sentence a fallback may be logged with.

    Every field is authored by this code or by the pipeline: the stage name,
    the category (:data:`REASON_KEYS`), the counts, the affected cluster ids
    (SHA-1 digests) and ``summary``, a code-authored phrase truncated to
    :data:`SUMMARY_MAX_CHARS`. Nothing the provider or the model said can
    reach it, so any call site that wants to announce a fallback must build
    its WARNING here and send the raw failure to ``logger.debug`` instead.
    """
    clusters = sorted({str(value) for value in cluster_ids if str(value)})
    if attributed is None:
        attributed = bool(clusters or post_count)
    scope = ",".join(clusters) or ("<unattributed>" if not attributed else "-")
    return (
        f"LLM stage {_clean_stage(stage)} fell back to its deterministic path "
        f"(count={int(count)}, reason={_clean_category(category)}, "
        f"clusters={scope}, posts={int(post_count)}): "
        f"{(summary or 'no summary recorded')[:SUMMARY_MAX_CHARS]}"
    )


def _clean_stage(stage: str) -> str:
    # Stage names are code-authored (STAGE_BY_KEY); anything else is a caller
    # bug and must not put free text on a WARNING line.
    return stage if stage in set(STAGE_BY_KEY.values()) else "unknown_stage"


class LLMFallbackTracker:
    """Thread-safe counters plus the cluster/post scope of each fallback."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters = dict.fromkeys(FALLBACK_KEYS, 0)
        self._pad_counters = dict.fromkeys(PAD_KEYS, 0)
        self._reason_counters = dict.fromkeys(REASON_KEYS, 0)
        self._unattributed = 0
        # Scope of every recorded fallback, per stage. A report is partial
        # when its own cluster id is in the stage's cluster set or one of its
        # posts is in the stage's post set -- nothing wider.
        self._cluster_ids_by_stage: dict[str, set[str]] = {}
        self._post_ids_by_stage: dict[str, set[str]] = {}

    def reset(self) -> None:
        with self._lock:
            self._counters = dict.fromkeys(FALLBACK_KEYS, 0)
            self._pad_counters = dict.fromkeys(PAD_KEYS, 0)
            self._reason_counters = dict.fromkeys(REASON_KEYS, 0)
            self._unattributed = 0
            self._cluster_ids_by_stage = {}
            self._post_ids_by_stage = {}

    def record(
        self,
        key: str,
        *,
        count: int = 1,
        cluster_id: str = "",
        cluster_ids: Iterable[str] = (),
        post_ids: Iterable[str] = (),
        category: str = "unknown",
        summary: str = "",
        detail: str = "",
    ) -> None:
        """Record ``count`` fallbacks for ``key`` and log why they happened.

        ``cluster_id``/``cluster_ids`` and ``post_ids`` are the scope of the
        failing unit: exactly the clusters and posts whose reports may now
        carry a deterministic answer. A batched stage passes every member of
        the failing batch; the batches that succeeded are never named, so a
        single bad batch cannot mark the whole cycle ``llm_partial``.

        ``category`` is authored by the caller (usually
        :func:`category_of` over the exception that this code raised), never
        derived from the failure text. ``summary`` is a short, code-authored
        phrase safe to log; ``detail`` may carry the provider's or the model's
        own words and therefore only ever reaches DEBUG.
        """
        if key not in self._counters:
            raise KeyError(f"unknown LLM fallback key: {key}")
        stage = STAGE_BY_KEY[key]
        scoped_clusters = {str(value) for value in cluster_ids if str(value)}
        if cluster_id:
            scoped_clusters.add(str(cluster_id))
        scoped_posts = {str(post_id) for post_id in post_ids if str(post_id)}
        category = _clean_category(category)
        attributed = bool(scoped_clusters or scoped_posts)
        with self._lock:
            self._counters[key] += max(0, int(count))
            self._reason_counters[category] += max(0, int(count))
            if not attributed:
                self._unattributed += max(0, int(count))
            if scoped_clusters:
                self._cluster_ids_by_stage.setdefault(stage, set()).update(scoped_clusters)
            if scoped_posts:
                self._post_ids_by_stage.setdefault(stage, set()).update(scoped_posts)
        # Content-free by construction: stage, category, counts and cluster
        # ids are all authored here or by the pipeline, and ``summary`` is a
        # bounded phrase the caller wrote. Nothing the provider or the model
        # said appears on this line.
        logger.warning(
            "%s",
            format_fallback_line(
                stage,
                category=category,
                count=count,
                cluster_ids=scoped_clusters,
                post_count=len(scoped_posts),
                summary=summary,
                attributed=attributed,
            ),
        )
        if not attributed and key in _PRE_REPORT_KEYS:
            # Counted and logged, never silently dropped: without a scope this
            # fallback cannot reach any report's generation_mode, which is the
            # same invisibility the old cycle-global attribution had.
            logger.error(
                "LLM stage %s recorded a fallback with no cluster or post scope; "
                "it cannot reach any report's generation_mode",
                stage,
            )
        if detail:
            logger.debug("LLM stage %s fallback detail: %s", stage, detail)

    def record_padding(
        self,
        key: str,
        *,
        count: int = 1,
        cluster_id: str = "",
        summary: str = "",
    ) -> None:
        """Record ``count`` non-fallback padding events for ``key``.

        Unlike :meth:`record`, this never touches the per-cluster/per-post
        stage bookkeeping or the reason histogram, so it can never flip a
        report's ``generation_mode`` or appear in its ``llm_stage_fallbacks``.
        """
        if key not in self._pad_counters:
            raise KeyError(f"unknown LLM padding key: {key}")
        with self._lock:
            self._pad_counters[key] += max(0, int(count))
        logger.info(
            "LLM stage %s padded a partially valid answer (count=%d, cluster=%s): %s",
            key,
            count,
            cluster_id or "-",
            (summary or "no summary recorded")[:SUMMARY_MAX_CHARS],
        )

    def counters(self) -> dict[str, int]:
        with self._lock:
            return {**self._counters, **self._pad_counters, UNATTRIBUTED_KEY: self._unattributed}

    def reason_counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._reason_counters)

    def total(self) -> int:
        with self._lock:
            return sum(self._counters.values())

    def stages_for_cluster(
        self,
        cluster_id: str,
        post_ids: Iterable[str] = (),
    ) -> list[str]:
        """Stage names that fell back for *this* cluster, in pipeline order.

        A stage is named only when its recorded scope actually covers this
        report: the cluster id was in the failing unit, or one of the
        cluster's own posts was. Stages whose fallbacks touched other
        clusters or other batches are not named here.
        """
        wanted_posts = {str(post_id) for post_id in post_ids}
        cluster_key = str(cluster_id)
        matched: set[str] = set()
        with self._lock:
            for stage, cluster_scope in self._cluster_ids_by_stage.items():
                if cluster_key and cluster_key in cluster_scope:
                    matched.add(stage)
            for stage, post_scope in self._post_ids_by_stage.items():
                if wanted_posts and post_scope & wanted_posts:
                    matched.add(stage)
        # Pipeline order (FALLBACK_KEYS order) rather than record order, so
        # the list a report carries is stable regardless of thread timing.
        return [STAGE_BY_KEY[key] for key in FALLBACK_KEYS if STAGE_BY_KEY[key] in matched]


_TRACKER = LLMFallbackTracker()


def tracker() -> LLMFallbackTracker:
    return _TRACKER


def reset_fallbacks() -> None:
    """Start a new cycle's accounting (called once per cycle)."""
    _TRACKER.reset()


def record_fallback(
    key: str,
    *,
    count: int = 1,
    cluster_id: str = "",
    cluster_ids: Iterable[str] = (),
    post_ids: Iterable[str] = (),
    category: str = "unknown",
    summary: str = "",
    detail: str = "",
) -> None:
    _TRACKER.record(
        key,
        count=count,
        cluster_id=cluster_id,
        cluster_ids=cluster_ids,
        post_ids=post_ids,
        category=category,
        summary=summary,
        detail=detail,
    )


def record_padding(key: str, *, count: int = 1, cluster_id: str = "", summary: str = "") -> None:
    _TRACKER.record_padding(key, count=count, cluster_id=cluster_id, summary=summary)


def fallback_counters() -> dict[str, int]:
    return _TRACKER.counters()


def fallback_reason_counters() -> dict[str, int]:
    """Histogram of *why* this cycle's fallbacks happened, by category."""
    return _TRACKER.reason_counters()


def stages_for_cluster(
    cluster_id: str,
    post_ids: Iterable[str] = (),
) -> list[str]:
    return _TRACKER.stages_for_cluster(cluster_id, post_ids)

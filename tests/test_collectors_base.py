from datetime import UTC, datetime
from tiblender_runtime.collectors.base import CollectionQuery, CollectionWindow, QueryPlan, RawPost


def test_window_ending_at_and_contains():
    window = CollectionWindow.ending_at(datetime(2026, 3, 20, 10, tzinfo=UTC), hours=1.0, cycle_index=3)
    assert window.start == datetime(2026, 3, 20, 9, tzinfo=UTC) and window.label == "cycle_003"
    assert window.contains("2026-03-20T09:30:00Z") and not window.contains("2026-03-20T11:00:00Z")


def test_query_plan_orders_by_priority_then_query():
    plan = QueryPlan([CollectionQuery("x", "b", "static", priority=5), CollectionQuery("x", "a", "dynamic", priority=9), CollectionQuery("reddit", "c", "subreddit")])
    assert [q.query for q in plan.for_platform("x")] == ["a", "b"]
    assert [q.query for q in plan.dynamic()] == ["a"]


def test_raw_post_id_is_platform_scoped():
    assert RawPost("x", "1", "hi", "2026-01-01T00:00:00Z").post_id == "x-1"

from nightly_rca.logic import (
    classify_fix_lineage,
    coverage_summary,
    discover_clusters,
    match_clusters_to_profiles,
    parse_dates,
)


def test_parse_dates_is_sorted_deduplicated_and_validated():
    response = {"dates": ["2026-07-19", {"date": "2026-07-21"}, "bad", "2026-07-21"]}
    assert parse_dates(response) == ["2026-07-21", "2026-07-19"]


def test_cluster_discovery_combines_alert_and_anomaly_evidence():
    clusters = discover_clusters(
        {"stability": {"buckets": [{"key": "PMGR_UNEXPECTED_EXIT", "count": 5000}]}},
        {"anomalies": [{"alert_name": "PMGR_UNEXPECTED_EXIT", "score": 97, "receiver_ids": ["R1"]}]},
        ["2026-07-21"],
        history_counts={"pmgr_unexpected_exit": 3},
    )
    assert len(clusters) == 1
    assert clusters[0]["receivers_seen"] == ["R1"]
    assert set(clusters[0]["signal_families"]) == {"stability", "ml_anomaly"}
    assert clusters[0]["recurrence_nights"] == 3
    assert clusters[0]["confidence"] == "high"


def test_profile_matching_respects_direct_patterns():
    clusters = [{
        "candidate_id": "c1", "key_patterns": ["PMGR_UNEXPECTED_EXIT"],
        "signal_families": ["stability"], "receivers_seen": ["R1"],
    }]
    catalog = {"pmgr": {"strong_patterns": ["PMGR_UNEXPECTED_EXIT"]}}
    result = match_clusters_to_profiles(clusters, catalog)
    assert result[0]["match_quality"] == "MATCHES_EXISTING_PROFILE"
    assert result[0]["best_profile_match"] == "pmgr"


def test_fix_lineage_classification_is_conservative():
    assert classify_fix_lineage({"jira_fix_ids": ["A"], "commit_sha": "abc"}) == "NEWLY_RESOLVED"
    assert classify_fix_lineage({"jira_fix_ids": ["A"]}) == "PARTIALLY_ADDRESSED"
    assert classify_fix_lineage({"jira_issue_ids": ["A"]}) == "FIX_LINEAGE_UPDATED"
    assert classify_fix_lineage({}) == "STILL_OPEN"


def test_profile_duplicate_requires_same_profile_receiver_and_date():
    clusters = [{
        "candidate_id": "c1", "key_patterns": ["PMGR_UNEXPECTED_EXIT"],
        "signal_families": ["stability"], "receivers_seen": [], "dates": ["2026-07-21"],
    }]
    catalog = {"pmgr": {"strong_patterns": ["PMGR_UNEXPECTED_EXIT"]}}
    unrelated_case = [{
        "case_id": "old", "issue_profile": "pmgr", "receiver_id": "R1",
        "event_date": "2026-07-20",
    }]
    result = match_clusters_to_profiles(clusters, catalog, unrelated_case)
    assert result[0]["match_quality"] == "MATCHES_EXISTING_PROFILE"

    clusters[0]["receivers_seen"] = ["R1"]
    clusters[0]["dates"] = ["2026-07-20"]
    result = match_clusters_to_profiles(clusters, catalog, unrelated_case)
    assert result[0]["match_quality"] == "DUPLICATE_OF_KNOWN_CASE"


def test_explicit_missing_required_logs_never_becomes_validation_pass():
    response = {"coverage": [{
        "receiver_id": "R1", "status": "complete",
        "available_log_types": ["stbc_main"], "required_logs_present": False,
    }]}
    summary = coverage_summary(response)
    assert summary["covered_receivers"] == 1
    assert summary["required_logs_present"] is False


def test_profile_matching_preserves_anchored_token_patterns():
    clusters = [{
        "candidate_id": "c1031", "key_patterns": ["1031"],
        "signal_families": ["guide_epg_popup"], "receivers_seen": [], "dates": ["2026-07-21"],
    }]
    catalog = {"guide_1031": {"strong_patterns": [r"(^|_)1031($|_)|^1031$"]}}
    result = match_clusters_to_profiles(clusters, catalog)
    assert result[0]["match_quality"] == "MATCHES_EXISTING_PROFILE"
    assert result[0]["best_profile_match"] == "guide_1031"


def test_inactive_status_does_not_pass_a_canary():
    from nightly_rca.phases import _call_pass

    assert _call_pass({"active": False, "status": "INACTIVE"}) is False
    assert _call_pass({"status": "PARTIAL"}) is False
    assert _call_pass({"status": "READY"}) is True


def test_history_learning_uses_only_completed_runs(tmp_path):
    import json
    from nightly_rca.history import load_history_signals

    completed = tmp_path / "runs" / "20260720T020000Z"
    incomplete = tmp_path / "runs" / "20260719T020000Z"
    completed.mkdir(parents=True)
    incomplete.mkdir(parents=True)
    completed.joinpath("state.json").write_text(json.dumps({
        "status": "COMPLETE",
        "data": {
            "candidate_clusters": [{"suspected_profile": "mystery"}],
            "profile_validations": [{
                "suspected_profile": "mystery",
                "classification": "PROFILE_VALIDATION_FAIL",
            }],
        },
    }))
    incomplete.joinpath("state.json").write_text(json.dumps({
        "status": "ABORTED",
        "data": {
            "candidate_clusters": [{"suspected_profile": "mystery"}],
            "profile_validations": [{
                "suspected_profile": "mystery",
                "classification": "PROFILE_VALIDATION_PASS",
            }],
        },
    }))

    signals = load_history_signals(tmp_path, "current", 14)
    assert signals["recurrence"] == {"mystery": 1}
    assert signals["validation_failures"] == {"mystery": 1}
    assert signals["validation_passes"] == {}


# ---------------------------------------------------------------------------
# Tests for select_validation_targets (fairness-aware selector)
# ---------------------------------------------------------------------------

from nightly_rca.logic import select_validation_targets


def _fresh(profile_id: str, priority: float, severity: str = "medium") -> dict:
    """Build a minimal fresh (non-resumed) validation target."""
    return {
        "profile_id": profile_id,
        "candidate_id": f"cand_{profile_id}",
        "cluster": {
            "priority_score": priority,
            "severity_guess": severity,
            "confidence": "high",
            "receivers_seen": [f"R_{profile_id}"],
        },
    }


def _resumed(profile_id: str, priority: float, created_at: str = "2026-07-27T02:00:00Z") -> dict:
    """Build a minimal resumed-tracker validation target."""
    t = _fresh(profile_id, priority)
    t["resumed_tracker"] = {
        "profile_id": profile_id,
        "created_at": created_at,
        "workflow_status": "upload_requested_pending_receipt",
    }
    return t


def test_selector_respects_max_total():
    """Output list never exceeds max_total regardless of input length."""
    targets = [_fresh(str(i), float(i)) for i in range(20)]
    result = select_validation_targets(targets, max_total=5)
    assert len(result) == 5


def test_selector_empty_input():
    assert select_validation_targets([], max_total=10) == []


def test_selector_zero_max():
    targets = [_fresh("a", 50.0)]
    assert select_validation_targets(targets, max_total=0) == []


def test_resumed_targets_are_not_starved_by_fresh_wave():
    """Resumed trackers must get at least one slot even when outnumbered."""
    fresh_wave = [_fresh(str(i), float(100 - i)) for i in range(9)]
    pending = [_resumed("old_profile", 10.0)]
    targets = fresh_wave + pending

    result = select_validation_targets(targets, max_total=5)
    resumed_ids = {t["profile_id"] for t in result if t.get("resumed_tracker")}
    assert "old_profile" in resumed_ids, (
        "Resumed tracker was starved — should have been given at least one slot"
    )


def test_floor_is_respected_when_resumed_slots_available():
    """With default floor=max//3, at most max//3 resumed trackers are included."""
    resumed = [_resumed(f"r{i}", 5.0) for i in range(6)]
    fresh = [_fresh(f"f{i}", float(50 - i)) for i in range(6)]
    targets = fresh + resumed

    result = select_validation_targets(targets, max_total=6)
    resumed_count = sum(1 for t in result if t.get("resumed_tracker"))
    # floor = min(6, max(1, 6//3)) = 2
    assert resumed_count >= 2, "Floor not satisfied"
    assert len(result) <= 6


def test_fresh_fills_unused_resumed_slots():
    """When fewer resumed trackers exist than the floor, fresh targets fill the gap."""
    resumed = [_resumed("r1", 5.0)]
    fresh = [_fresh(f"f{i}", float(50 - i)) for i in range(10)]
    targets = fresh + resumed

    result = select_validation_targets(targets, max_total=6)
    # Should have 1 resumed + 5 fresh (no wasted slots)
    assert len(result) == 6
    resumed_count = sum(1 for t in result if t.get("resumed_tracker"))
    assert resumed_count == 1


def test_explicit_zero_floor_disables_resume_guarantee():
    """Passing resumed_floor=0 lets fresh targets take all slots."""
    fresh = [_fresh(f"f{i}", float(100 - i)) for i in range(5)]
    resumed = [_resumed("r1", 5.0)]
    targets = fresh + resumed

    result = select_validation_targets(targets, max_total=5, resumed_floor=0)
    resumed_in_result = [t for t in result if t.get("resumed_tracker")]
    assert len(resumed_in_result) == 0


def test_higher_priority_fresh_targets_rank_above_lower():
    """Within fresh bucket, higher priority_score must come first."""
    targets = [_fresh("low", 10.0), _fresh("high", 90.0), _fresh("mid", 50.0)]
    result = select_validation_targets(targets, max_total=3)
    ids = [t["profile_id"] for t in result]
    assert ids == ["high", "mid", "low"]


def test_resumed_targets_rank_above_fresh_by_default():
    """Resumed targets should score above any fresh target with equivalent cluster signal."""
    fresh = _fresh("f1", 200.0, severity="high")   # very high fresh priority
    pending = _resumed("r1", 1.0)                  # low cluster score but resumed
    result = select_validation_targets([fresh, pending], max_total=2)
    # Both fit; resumed should be first (higher _target_priority)
    assert result[0]["profile_id"] == "r1", (
        "Resumed target should outrank fresh target in sorted output"
    )


def test_deduplication_keeps_first_occurrence():
    """Duplicate (profile_id, receiver) pairs are dropped, keeping first occurrence."""
    t1 = _fresh("pmgr", 50.0)
    t2 = _fresh("pmgr", 80.0)  # same profile_id AND same receiver R_pmgr → duplicate
    result = select_validation_targets([t1, t2], max_total=5)
    assert len(result) == 1


def test_selector_handles_targets_without_cluster():
    """Targets missing cluster key should not raise; they sort to the back."""
    t_no_cluster = {"profile_id": "mystery", "candidate_id": "cand_mystery"}
    t_with_cluster = _fresh("known", 80.0)
    result = select_validation_targets([t_no_cluster, t_with_cluster], max_total=2)
    assert len(result) == 2
    # known has a real score; it should rank higher
    assert result[0]["profile_id"] == "known"

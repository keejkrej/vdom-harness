"""I_sku catalog rebind + I_weight TrainJob stub tests. No API key. No LoRA."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from tau2_vdom.improve import (
    CATALOG_JUMP_MODEL,
    EVAL_DIR,
    FORBIDDEN_HANG_SOURCES,
    I_SKU_NOTE,
    I_WEIGHT_NOTE,
    INCOMPLETE_FIXTURE_ID,
    LIVE_HANG_OBS_ISKU_FILE,
    LIVE_HANG_OBS_ISKU_R6_FILE,
    SERVING_MODEL,
    _sidecar_catalog_jump,
    assert_live_hang_obs_isku_cell,
    build_live_hang_obs_isku_report,
    incomplete_fixture_traces,
    incomplete_reason,
    incomplete_train_traces,
    is_incomplete_episode,
    pending_live_hang_obs_isku_report,
    run_isku_mount_cell,
    run_hybrid_state_s_dump,
    run_hybrid_state_serving_step_dump,
    run_weight_fixture_improve,
)
from tau2_vdom.runner import control_batch
from tau2_vdom.sidecar import default_sidecar


def c_topology(graph: dict) -> list[dict]:
    out: list[dict] = []

    def walk(n: dict) -> None:
        if not n:
            return
        out.append(
            {
                "key": n.get("key"),
                "objective": n.get("objective") or "",
                "prompt": n.get("prompt") or "",
            }
        )
        for child in n.get("children") or []:
            walk(child)

    walk((graph or {}).get("root") or {})
    return out


def _reset_sidecar():
    sidecar = default_sidecar()
    sidecar.request({"op": "set_technique", "technique": "one-shot"})
    return sidecar


def test_incomplete_picker() -> None:
    hung = SimpleNamespace(
        task_id="41",
        trial=0,
        hung=True,
        termination_reason="timeout",
        reward_info=None,
    )
    transfer = SimpleNamespace(
        task_id="39",
        trial=0,
        hung=False,
        termination_reason="transfer_to_human",
        reward_info=SimpleNamespace(reward=0.0),
    )
    hit = SimpleNamespace(
        task_id="41",
        trial=1,
        hung=False,
        termination_reason="user_stop",
        reward_info=SimpleNamespace(reward=1.0),
    )
    assert is_incomplete_episode(hung) is True
    assert is_incomplete_episode(transfer) is True
    assert is_incomplete_episode(hit) is False
    assert incomplete_reason(hung) == "hung"
    assert incomplete_reason(transfer) == "reward0-early-transfer"

    traces = incomplete_train_traces([hung, transfer, hit])
    assert {t["taskId"] for t in traces} == {"41", "39"}
    assert all(t["reason"] in {"hung", "reward0-early-transfer"} for t in traces)


def test_incomplete_fixture_shape() -> None:
    traces = incomplete_fixture_traces()
    assert traces[0]["taskId"] == INCOMPLETE_FIXTURE_ID
    assert traces[0]["reward"] == 0.0
    assert traces[0]["reason"] == "reward0-early-transfer"
    assert "transfer" in traces[0]["termination"]


def test_catalog_jump_mounts_0813() -> None:
    sidecar = _reset_sidecar()
    before_state = sidecar.request({"op": "get_state"})
    graph_before = before_state.get("graph") or {}
    topo_before = c_topology(graph_before)

    missing = _sidecar_catalog_jump(sidecar, before=0.0)
    assert missing["arm"] == "I_sku"
    assert missing["jumped"] is False
    assert missing["rejected"] is True
    assert missing["servingPaused"] is False
    assert missing["serving"]["sku"] == SERVING_MODEL
    assert "0813 existing is not a gate" in str(missing.get("catalog") or missing.get("gate") or missing)

    reject = _sidecar_catalog_jump(sidecar, before=1.0, after=0.0)
    assert reject["arm"] == "I_sku"
    assert reject["kind"] == "catalog-rebind"
    assert reject["trained"] is False
    assert reject["jumped"] is False
    assert reject["rejected"] is True
    assert reject["servingPaused"] is False
    assert reject["servingModel"] == SERVING_MODEL
    assert reject["serving"]["sku"] == SERVING_MODEL
    assert reject["proposed"] == CATALOG_JUMP_MODEL
    assert "catalog rebind" in reject["honestNote"]
    assert "not fine-tuning" in reject["honestNote"]
    assert I_SKU_NOTE in reject["honestNote"]

    mount = _sidecar_catalog_jump(sidecar, before=0.0, after=1.0)
    assert mount["arm"] == "I_sku"
    assert mount["kind"] == "catalog-rebind"
    assert mount["trained"] is False
    assert mount["jumped"] is True
    assert mount["mounted"] is True
    assert mount["servingPaused"] is False
    assert mount["servingModel"] == CATALOG_JUMP_MODEL
    assert mount["serving"]["sku"] == CATALOG_JUMP_MODEL
    graph = mount.get("graph") or (mount.get("catalog") or {}).get("graph") or {}
    solve = (graph or {}).get("root") or {}
    assert solve.get("model") in {None, SERVING_MODEL}
    assert c_topology(graph) == topo_before
    assert "not fine-tuning" in (mount.get("honestNote") or "")
    later = sidecar.request({"op": "ping"})
    # Process default is S0. The I_sku *cell* wrote 0813; ping is not HybridState.S.
    assert later.get("servingModel") == SERVING_MODEL
    assert later.get("serving", {}).get("sku") == SERVING_MODEL
    assert later.get("servingSku", {}).get("sku") == SERVING_MODEL
    assert later.get("servingPaused") is False


def test_isku_writes_s_not_c() -> None:
    """MUST-HOLD: I_sku mount writes S only. C topology stays. Omit after keeps 0731."""
    sidecar = _reset_sidecar()
    start = sidecar.request({"op": "get_state"})
    graph_before = start.get("graph") or {}
    topo_before = c_topology(graph_before)
    assert start.get("servingSku", {}).get("sku") == SERVING_MODEL

    omit = _sidecar_catalog_jump(sidecar, before=0.0)
    assert omit["rejected"] is True
    assert omit["jumped"] is False
    assert omit["servingPaused"] is False
    assert omit["serving"]["sku"] == SERVING_MODEL
    assert "0813 existing is not a gate" in str(omit.get("catalog") or omit.get("gate") or omit)

    reject = _sidecar_catalog_jump(sidecar, before=1.0, after=0.0)
    assert reject["serving"]["sku"] == SERVING_MODEL
    assert reject["servingModel"] == SERVING_MODEL
    assert reject["servingPaused"] is False

    mount = _sidecar_catalog_jump(sidecar, before=0.0, after=1.0)
    graph = mount.get("graph") or {}
    assert c_topology(graph) == topo_before
    solve = (graph.get("root") or {})
    assert solve.get("model") in {None, SERVING_MODEL}
    assert mount["serving"]["sku"] == CATALOG_JUMP_MODEL
    assert mount["servingModel"] == CATALOG_JUMP_MODEL
    assert mount["servingPaused"] is False
    assert mount["trained"] is False
    assert mount["kind"] == "catalog-rebind"
    assert "not fine-tuning" in (mount.get("honestNote") or "")
    ping = sidecar.request({"op": "ping"})
    # Standalone catalog jump has no weighted episode; process servingSku is not truth.
    assert ping.get("servingModel") == SERVING_MODEL
    assert ping.get("servingSku", {}).get("sku") == SERVING_MODEL
    assert ping.get("servingPaused") is False


def test_isku_mount_cell_controller_no_live() -> None:
    """Mount-cell controller path: fixture after, no live. 44→0813, 39 stays 0731."""
    sidecar = _reset_sidecar()
    obs39 = {
        "taskId": "39",
        "nSteps": 4,
        "nSuccessProxy": 0,
        "lastActions": ["cancel_reservation"],
        "channels": ["env"],
        "critique": "",
        "toolFailures": 0,
        "repeatActions": 0,
        "arm": "I_loop",
        "refusedCancel": True,
        "inventedPolicy": True,
        "hung": False,
        "techniqueRecommendation": "policy-checklist",
        "missedActions": [{"name": "cancel_reservation"}],
    }
    obs44 = {
        "taskId": "44",
        "nSteps": 0,
        "nSuccessProxy": 0,
        "lastActions": [],
        "channels": [],
        "critique": "",
        "toolFailures": 0,
        "repeatActions": 0,
        "arm": "I_sku",
        "hung": True,
        "termination": "timeout",
    }
    ctrl = control_batch([obs39, obs44])
    assert ctrl["buckets"]["44"] == "I_sku"
    assert ctrl["buckets"]["39"] == "I_loop"
    assert ctrl["applyScope"]["waitKept"] == []
    assert ctrl["applyScope"]["weighted"] == ["44"]
    loop = sidecar.request({"op": "i_loop", "obs": [obs39, obs44], "model": "deterministic"})
    assert loop.get("applyScope", {}).get("weighted") == ["44"]
    assert loop.get("servingPaused") is False
    # Dedicated path: i_sku WITH after (live airline improveLoop omits after).
    mount = _sidecar_catalog_jump(sidecar, before=0.0, after=1.0)
    assert mount["mounted"] is True
    assert mount["servingPaused"] is False
    assert mount["trained"] is False
    assert mount["serving"]["sku"] == CATALOG_JUMP_MODEL
    turn44 = sidecar.request(
        {
            "op": "turn",
            "taskId": "44",
            "model": "deterministic",
            "policy": "",
            "tools": [],
            "messages": [{"role": "user", "content": "hello"}],
        }
    )
    turn39 = sidecar.request(
        {
            "op": "turn",
            "taskId": "39",
            "model": "deterministic",
            "policy": "",
            "tools": [],
            "messages": [{"role": "user", "content": "hello"}],
        }
    )
    assert turn44.get("servingModel") == CATALOG_JUMP_MODEL
    assert turn39.get("servingModel") == SERVING_MODEL
    assert turn44.get("servingPaused") is False
    assert turn44.get("X", {}).get("44", {}).get("S", {}).get("sku") == CATALOG_JUMP_MODEL
    assert turn39.get("X", {}).get("39", {}).get("S", {}).get("sku") == SERVING_MODEL
    x44 = turn44.get("X", {}).get("44") or {}
    assert x44.get("H"), "sidecar turn writes H onto existing X"
    assert x44.get("M"), "sidecar turn writes M onto existing X"
    assert any(m.get("content") == "hello" for m in x44.get("H", []) if m.get("role") == "user")
    assert any(m.get("role") == "assistant" for m in x44.get("H", []))
    dump = sidecar.request({"op": "dump_hybrid"})
    assert dump.get("X", {}).get("44", {}).get("S", {}).get("sku") == CATALOG_JUMP_MODEL
    assert dump.get("X", {}).get("39", {}).get("S", {}).get("sku") == SERVING_MODEL
    assert dump.get("servingByTaskIs") == "derived cache from X.S, not the lookup"
    assert dump.get("dumpIsNot") == "ping / get_state S0"
    ping = sidecar.request({"op": "ping"})
    assert ping.get("servingModel") == SERVING_MODEL
    assert ping.get("serving", {}).get("sku") == SERVING_MODEL
    assert callable(run_isku_mount_cell)
    assert callable(run_hybrid_state_s_dump)
    assert callable(run_hybrid_state_serving_step_dump)
    src = Path(__file__).with_name("improve.py").read_text()
    assert "Fixture after only at the controller; omit here so live airline" in src
    assert "sku_w = _sidecar_catalog_jump(sidecar, before=before)" in src
    assert "sku_w = _sidecar_catalog_jump(sidecar, before=before, after=" not in src


def test_live_hang_obs_isku_refuses_old_hung_replay_after_phit() -> None:
    hung = {
        "taskId": "44",
        "nSteps": 0,
        "nSuccessProxy": 0,
        "lastActions": [],
        "channels": [],
        "critique": "",
        "toolFailures": 0,
        "repeatActions": 0,
        "arm": "I_sku",
        "hung": True,
        "termination": "timeout",
    }
    for name in FORBIDDEN_HANG_SOURCES:
        try:
            build_live_hang_obs_isku_report(obs_list=[hung], source_eval=[name])
        except ValueError as exc:
            assert "sourceEval-of-old-hung" in str(exc)
        else:
            raise AssertionError(f"should refuse {name}")
    try:
        build_live_hang_obs_isku_report(obs_list=[hung], controller_replay=True)
    except ValueError as exc:
        assert "controllerReplay=true" in str(exc)
    else:
        raise AssertionError("should refuse controllerReplay")
    try:
        build_live_hang_obs_isku_report(obs_list=[hung], after=1.0)
    except ValueError as exc:
        assert "after=" in str(exc)
    else:
        raise AssertionError("should refuse after=")
    try:
        build_live_hang_obs_isku_report(obs_list=[hung], p_hit_0813=0.5)
    except ValueError as exc:
        assert "pHit0813" in str(exc)
    else:
        raise AssertionError("should refuse pHit0813")


def test_live_hang_obs_isku_this_episode_hung_omit_after() -> None:
    """THIS episode (HungSimulation), not old hung files. I_sku omits after=."""
    sidecar = _reset_sidecar()
    hung = {
        "taskId": "44",
        "nSteps": 0,
        "nSuccessProxy": 0,
        "lastActions": [],
        "channels": [],
        "critique": "",
        "toolFailures": 0,
        "repeatActions": 0,
        "arm": "I_sku",
        "hung": True,
        "termination": "timeout",
    }
    ctrl = control_batch([hung])
    assert ctrl["episodes"][0]["arm"] == "I_sku"
    assert ctrl["applyScope"]["weighted"] == ["44"]
    assert "44" not in (ctrl["applyScope"]["waitKept"] or [])
    sku_w = _sidecar_catalog_jump(sidecar, before=0.0)
    assert sku_w["jumped"] is False
    assert sku_w["rejected"] is True
    assert sku_w["servingPaused"] is False
    assert sku_w["servingModel"] == SERVING_MODEL
    report = build_live_hang_obs_isku_report(
        obs_list=[hung],
        source_eval=[LIVE_HANG_OBS_ISKU_FILE],
        sku_w=sku_w,
    )
    assert report["live"] is True
    assert report["controllerReplay"] is False
    assert report["freshHang"] is True
    assert report["hung"] is True
    assert report["obs"]["arm"] == "I_sku"
    assert report["omitAfter"] is True
    assert report["jumped"] is False
    assert report["servingPaused"] is False
    assert report["servingModelAfter"] == SERVING_MODEL
    assert report["pHit0813"] is None
    assert report["gate"]["after"] is None
    assert report["gate"]["action"] == "reject"
    assert report["iSkuRequest"] == {"op": "i_sku", "before": 0}
    assert "after" not in report["iSkuRequest"]
    assert report["trained"] is False
    assert "live Obs of this episode" in report["reading"]
    assert "not a controller replay of saved hung-44" in report["reading"]
    assert "not a score" in report["reading"]
    assert "not a dump" in report["reading"]
    assert "not live hung-44 then served as a mount" in report["reading"]
    assert "not a new timeout" not in report["reading"]
    for name in FORBIDDEN_HANG_SOURCES:
        assert name not in json.dumps(report)


def test_live_hang_obs_isku_no_hang_keeps_hole_open() -> None:
    miss = {
        "taskId": "44",
        "nSteps": 4,
        "nSuccessProxy": 0,
        "lastActions": ["update_reservation_flights"],
        "channels": ["env"],
        "critique": "",
        "toolFailures": 0,
        "repeatActions": 0,
        "arm": "I_loop",
        "hung": False,
        "termination": "user_stop",
    }
    report = build_live_hang_obs_isku_report(
        obs_list=[miss],
        source_eval=[LIVE_HANG_OBS_ISKU_FILE],
    )
    assert report["freshHang"] is False
    assert report["hung"] is False
    assert report["holeOpen"] is True
    assert report["obs"]["arm"] == "I_loop"
    assert report["iSkuRequest"] is None
    assert report["controllerReplay"] is False
    assert report["pHit0813"] is None
    assert "hole remains open" in report["reading"]
    assert "hung-first Obs chose I_sku" not in report["reading"]


def test_live_hang_obs_isku_pending_key_and_landed() -> None:
    pending = pending_live_hang_obs_isku_report()
    assert pending["pendingKey"] is True
    assert pending["freshHang"] is False
    assert pending["hung"] is False
    assert pending["holeOpen"] is True
    assert pending["controllerReplay"] is False
    assert pending["obs"]["arm"] is None
    assert pending["pHit0813"] is None
    assert "pending a key" in pending["reading"]
    landed = json.loads((EVAL_DIR / LIVE_HANG_OBS_ISKU_FILE).read_text())
    assert_live_hang_obs_isku_cell(landed)
    assert landed["controllerReplay"] is False
    assert landed["pHit0813"] is None
    assert landed["omitAfter"] is True
    assert landed["servingPaused"] is False
    assert landed["trained"] is False
    if landed.get("pendingKey"):
        assert landed["freshHang"] is False
        assert landed["hung"] is False
        assert landed["holeOpen"] is True
    else:
        assert landed["pendingKey"] is False
        assert landed["freshHang"] is False
        assert landed["hung"] is False
        assert landed["holeOpen"] is True
        assert landed["obs"]["arm"] == "I_loop"
        assert landed["obs"]["hung"] is False
        assert landed["obs"]["taskId"] == "44"
        assert landed["obs"]["termination"] == "user_stop"
        assert landed["obs"]["nSuccessProxy"] == 0
        assert landed["iSkuRequest"] is None
        assert landed["gate"]["action"] is None
        assert "hole remains open" in landed["reading"]
        assert "hung-first Obs chose I_sku" not in landed["reading"]
    blob = json.dumps(landed)
    for name in FORBIDDEN_HANG_SOURCES:
        assert name not in blob
    src = Path(__file__).with_name("improve.py").read_text()
    assert "def run_live_hang_obs_isku" in src
    assert 'sku_w = _sidecar_catalog_jump(sidecar, before=before)' in src
    assert "sku_w = _sidecar_catalog_jump(sidecar, before=before, after=" not in src


def test_reject_cell_12_stays_controller_replay() -> None:
    reject = json.loads((EVAL_DIR / "improve-live-0731-isku-44-reject.json").read_text())
    assert reject["controllerReplay"] is True
    assert reject["sourceEval"] == list(FORBIDDEN_HANG_SOURCES)
    assert "not a new timeout" in reject["reading"]
    landed = json.loads((EVAL_DIR / LIVE_HANG_OBS_ISKU_FILE).read_text())
    assert landed["controllerReplay"] is False
    assert landed["kind"] != reject["kind"]
    blob = json.dumps(landed)
    for name in FORBIDDEN_HANG_SOURCES:
        assert name not in blob


def test_r6_later_timeout_does_not_overwrite_no_hang_packet() -> None:
    first = json.loads((EVAL_DIR / LIVE_HANG_OBS_ISKU_FILE).read_text())
    assert first["hung"] is False
    assert first["freshHang"] is False
    assert first["holeOpen"] is True
    assert first["obs"]["arm"] == "I_loop"
    assert first["controllerReplay"] is False
    r6 = json.loads((EVAL_DIR / LIVE_HANG_OBS_ISKU_R6_FILE).read_text())
    assert_live_hang_obs_isku_cell(r6)
    assert r6["controllerReplay"] is False
    assert r6["freshHang"] is True
    assert r6["hung"] is True
    assert r6["obs"]["arm"] == "I_sku"
    assert r6["applyScope"]["weighted"] == ["44"]
    assert "44" not in (r6["applyScope"]["waitKept"] or [])
    assert r6["omitAfter"] is True
    assert r6["gate"]["after"] is None
    assert r6["jumped"] is False
    assert r6["servingPaused"] is False
    assert r6["servingModelAfter"] == SERVING_MODEL
    assert r6["pHit0813"] is None
    assert r6["iSkuRequest"] == {"op": "i_sku", "before": 0}
    assert "after" not in r6["iSkuRequest"]
    blob = json.dumps(r6)
    for name in FORBIDDEN_HANG_SOURCES:
        assert name not in blob
    assert "not a new timeout" not in r6["reading"]


def test_weight_fixture_writes_report() -> None:
    path = run_weight_fixture_improve()
    latest = EVAL_DIR / "latest-improve.json"
    payload = json.loads(latest.read_text())
    assert path.is_file()
    assert payload["interventions"] == ["I_weight"]
    assert payload["servingPaused"] is False
    assert payload["iWeight"]["spawned"] is True
    assert payload["iWeight"]["done"] is True
    assert payload["iWeight"]["servingPaused"] is False
    assert payload["iWeight"]["not0731Weights"] is True
    assert payload["iWeight"]["rejected"] is True
    assert payload["iWeight"]["mounted"] is False
    job = payload["iWeight"]["job"]
    assert job["status"] in {"done", "failed"}
    assert job["servingPaused"] is False
    assert job.get("surrogate") is True
    assert job.get("not0731Weights") is True
    assert job.get("tracesUsed")
    assert job.get("artifactPointer") or job.get("status") == "failed"
    assert payload["stopReason"] == "weight-rejected"
    assert I_WEIGHT_NOTE in (payload.get("honestNote") or "")
    rounds = payload["rounds"]
    assert rounds[-1]["intervention"] == "I_weight"
    assert rounds[-1]["iWeight"]["spawned"] is True


def main() -> int:
    tests = [
        test_incomplete_picker,
        test_incomplete_fixture_shape,
        test_catalog_jump_mounts_0813,
        test_isku_writes_s_not_c,
        test_isku_mount_cell_controller_no_live,
        test_live_hang_obs_isku_refuses_old_hung_replay_after_phit,
        test_live_hang_obs_isku_this_episode_hung_omit_after,
        test_live_hang_obs_isku_no_hang_keeps_hole_open,
        test_live_hang_obs_isku_pending_key_and_landed,
        test_reject_cell_12_stays_controller_replay,
        test_r6_later_timeout_does_not_overwrite_no_hang_packet,
        test_weight_fixture_writes_report,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"ok  {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"not ok  {fn.__name__}: {exc}")
    print(f"{len(tests) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

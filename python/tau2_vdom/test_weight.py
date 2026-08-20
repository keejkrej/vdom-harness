"""I_sku catalog rebind + I_weight TrainJob stub tests. No API key. No LoRA."""

from __future__ import annotations

import json
from types import SimpleNamespace

from tau2_vdom.improve import (
    CATALOG_JUMP_MODEL,
    EVAL_DIR,
    I_SKU_NOTE,
    I_WEIGHT_NOTE,
    INCOMPLETE_FIXTURE_ID,
    SERVING_MODEL,
    _sidecar_catalog_jump,
    incomplete_fixture_traces,
    incomplete_reason,
    incomplete_train_traces,
    is_incomplete_episode,
    run_weight_fixture_improve,
)
from tau2_vdom.sidecar import default_sidecar


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
    sidecar = default_sidecar()
    missing = _sidecar_catalog_jump(sidecar, before=0.0)
    assert missing["arm"] == "I_sku"
    assert missing["jumped"] is False
    assert missing["rejected"] is True
    assert missing["servingPaused"] is False
    assert "0813 existing is not a gate" in str(missing.get("catalog") or missing.get("gate") or missing)

    reject = _sidecar_catalog_jump(sidecar, before=1.0, after=0.0)
    assert reject["arm"] == "I_sku"
    assert reject["kind"] == "catalog-rebind"
    assert reject["trained"] is False
    assert reject["jumped"] is False
    assert reject["rejected"] is True
    assert reject["servingPaused"] is False
    assert reject["servingModel"] == SERVING_MODEL
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
    graph = mount.get("graph") or (mount.get("catalog") or {}).get("graph") or {}
    solve = (graph or {}).get("root") or {}
    assert solve.get("model") == CATALOG_JUMP_MODEL
    assert "not fine-tuning" in (mount.get("honestNote") or "")


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

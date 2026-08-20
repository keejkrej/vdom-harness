"""Unit tests for failure-aware Obs and skip handling. No API key."""

from __future__ import annotations

from tau2_vdom.improve import (
    CATALOG_JUMP_MODEL,
    SERVING_MODEL,
    _collect_obs,
    _sidecar_catalog_jump,
    _task_p_hit,
    apply_scope_from_obs,
    missed_tool_names_only,
    pass_hat_k_from_rewards,
)
from tau2_vdom.sidecar import default_sidecar
from tau2_vdom.kernel_tools import strip_kernel_self_tools
from tau2_vdom.runner import (
    HungSimulation,
    _obs,
    control_batch,
    recommend_intervention,
    recommend_slice_intervention,
    serialize_reward_info,
)


def test_obs_missed_cancel_recommends_policy() -> None:
    reward_info = {
        "reward": 0.0,
        "action_checks": [
            {
                "action": {"name": "cancel_reservation", "arguments": {"reservation_id": "MSJ4OA"}},
                "action_match": False,
            },
            {
                "action": {"name": "cancel_reservation", "arguments": {"reservation_id": "8C8K4E"}},
                "action_match": True,
            },
        ],
        "communicate_checks": [],
        "nl_assertions": [
            {"nl_assertion": "Agent cancels reservation MSJ4OA.", "met": False},
        ],
    }
    actions = [
        {
            "kind": "tool",
            "text": "cancel_reservation",
            "toolName": "cancel_reservation",
            "toolArgs": {"reservation_id": "8C8K4E"},
            "ok": True,
        },
        {
            "kind": "text",
            "text": "I am afraid there is no way for me to cancel; no-show is not possible.",
            "ok": True,
        },
    ]
    obs = _obs(actions, 0.0, [], reward_info=reward_info)
    assert obs["nSuccessProxy"] == 0
    assert obs["missedActions"][0]["name"] == "cancel_reservation"
    assert obs["missedActions"][0]["arguments"]["reservation_id"] == "MSJ4OA"
    assert obs["refusedCancel"] is True
    assert obs["inventedPolicy"] is True
    assert obs["techniqueRecommendation"] == "policy-checklist"
    assert obs["arm"] == "I_loop"
    assert "policy-checklist" in obs["critique"]

    compact = serialize_reward_info(reward_info)
    assert compact is not None
    assert compact["missedActions"][0]["name"] == "cancel_reservation"
    assert compact["nl_assertions"][0]["nl_assertion"].startswith("Agent cancels")


def test_obs_mock_update_does_not_select_policy() -> None:
    reward_info = {
        "reward": 0.0,
        "action_checks": [
            {
                "action": {"name": "update_task_status", "arguments": {"task_id": "task_1"}},
                "action_match": False,
            }
        ],
    }
    actions = [
        {
            "kind": "tool",
            "text": "create_task",
            "toolName": "create_task",
            "toolArgs": {"title": "Important Meeting"},
            "ok": True,
        }
    ]
    obs = _obs(actions, 0.0, [], reward_info=reward_info)
    assert obs["missedActions"][0]["name"] == "update_task_status"
    assert obs["techniqueRecommendation"] is None
    assert obs["refusedCancel"] is False
    assert obs["arm"] == "I_loop"


def test_obs_hung_keeps_task() -> None:
    obs = _obs([], None, [], hung=True)
    assert obs["hung"] is True
    assert obs["nSuccessProxy"] == 0
    assert obs["arm"] == "I_sku"
    assert "null reward" in obs["critique"]
    assert recommend_intervention(obs) == "I_sku"


def test_skipped_task_stays_in_task_phit() -> None:
    by_task = {"39": [0.0], "44": [0.0], "41": [None]}
    phit = _task_p_hit(by_task)
    assert phit["39"] == 0.0
    assert phit["44"] == 0.0
    assert phit["41"] is None
    assert set(phit) == {"39", "41", "44"}
    # skip is not a measured 0 in pass^k
    hat = pass_hat_k_from_rewards(by_task)
    assert hat["1"] == 0.0
    assert hat["1"] == (0.0 + 0.0) / 2


def test_obs_personal_reason_is_invented_policy() -> None:
    actions = [
        {
            "kind": "text",
            "text": "I cannot cancel this economy reservation; a personal reason is not covered.",
            "ok": True,
        }
    ]
    obs = _obs(actions, 0.0, [])
    assert obs["inventedPolicy"] is True
    assert obs["refusedCancel"] is True
    assert obs["techniqueRecommendation"] == "policy-checklist"


def test_missed_tool_names_only_drops_gold_ids() -> None:
    names = missed_tool_names_only(
        [
            {
                "missedActions": [
                    {"name": "cancel_reservation", "arguments": {"reservation_id": "MSJ4OA"}},
                    {"name": "update_reservation_flights", "arguments": {"reservation_id": "S61CZX"}},
                ]
            }
        ]
    )
    assert names == ["cancel_reservation", "update_reservation_flights"]
    blob = " ".join(names)
    assert "MSJ4OA" not in blob
    assert "S61CZX" not in blob


def test_strip_kernel_self_tools_never_reach_env() -> None:
    leaked = [
        {"name": "get_agent_graph", "arguments": {}},
        {"name": "set_agent_graph", "arguments": {"graphPatch": {"nodes": []}}},
        {"name": "create_task", "arguments": {"title": "Important Meeting"}},
    ]
    gym = strip_kernel_self_tools(leaked)
    assert [t["name"] for t in gym] == ["create_task"]
    executed: list[str] = []

    def fake_env(calls: list[dict]) -> None:
        for tc in calls:
            if tc["name"] in {"get_agent_graph", "set_agent_graph"}:
                raise AssertionError(f"leaked {tc['name']} to env")
            executed.append(tc["name"])

    fake_env(gym)
    assert executed == ["create_task"]


def test_obs_includes_task_id() -> None:
    obs = _obs(
        [{"kind": "tool", "text": "cancel_reservation", "toolName": "cancel_reservation", "ok": True}],
        1.0,
        [],
        task_id="44",
    )
    assert obs["taskId"] == "44"
    assert obs["arm"] == "wait"
    assert obs["nSuccessProxy"] == 1


def test_apply_scope_mixed_wait_hit() -> None:
    scope = apply_scope_from_obs(
        [
            {"taskId": "44", "arm": "wait", "nSuccessProxy": 1, "hung": False},
            {"taskId": "39", "arm": "I_loop", "nSuccessProxy": 0, "hung": False},
        ]
    )
    assert scope == {"waitKept": ["44"], "looped": ["39"], "weighted": []}
    record = {"selfObsPath": "self", "applyScope": scope}
    assert record["selfObsPath"] == "self"
    assert record["applyScope"]["waitKept"] == ["44"]
    assert record["applyScope"]["looped"] == ["39"]


def test_collect_obs_sets_task_id() -> None:
    from types import SimpleNamespace

    hit = SimpleNamespace(
        task_id="44",
        messages=[],
        hung=False,
        reward_info=SimpleNamespace(reward=1.0),
    )
    miss = SimpleNamespace(
        task_id="39",
        messages=[
            SimpleNamespace(
                role="assistant",
                content="I cannot cancel this economy reservation; a personal reason is not covered.",
                tool_calls=[],
            )
        ],
        hung=False,
        reward_info=SimpleNamespace(reward=0.0),
    )
    obs = _collect_obs([hit, miss])
    assert obs[0]["taskId"] == "44"
    assert obs[0]["arm"] == "wait"
    assert obs[1]["taskId"] == "39"
    assert obs[1]["arm"] == "I_loop"
    assert apply_scope_from_obs(obs) == {"waitKept": ["44"], "looped": ["39"], "weighted": []}


def test_typed_arms_hung_hit_policy() -> None:
    hung = _obs([], None, [], hung=True)
    assert hung["arm"] == "I_sku"
    hung_attractor = _obs(
        [
            {
                "kind": "text",
                "text": "I cannot cancel this economy reservation; a personal reason is not covered.",
                "ok": True,
            }
        ],
        0.0,
        [],
        hung=True,
    )
    assert hung_attractor["inventedPolicy"] is True
    assert hung_attractor["arm"] == "I_sku"
    hit = _obs(
        [{"kind": "tool", "text": "cancel_reservation", "toolName": "cancel_reservation", "ok": True}],
        1.0,
        [],
        reward_info={"reward": 1.0, "action_checks": []},
    )
    assert hit["arm"] == "wait"
    policy = _obs(
        [
            {
                "kind": "text",
                "text": "I cannot cancel this economy reservation; a personal reason is not covered.",
                "ok": True,
            }
        ],
        0.0,
        [],
    )
    assert policy["arm"] == "I_loop"
    extra = _obs(
        [
            {
                "kind": "tool",
                "text": "cancel_reservation",
                "toolName": "cancel_reservation",
                "toolArgs": {"reservation_id": "OTHER"},
                "ok": True,
            }
        ],
        0.0,
        [],
    )
    assert extra["arm"] == "I_loop"
    assert extra["lastActions"] == ["cancel_reservation"]
    assert "MSJ4OA" not in str(extra)
    assert recommend_slice_intervention([hit, hung]) == "I_sku"
    assert recommend_slice_intervention([hit, policy]) == "I_loop"
    scope = apply_scope_from_obs(
        [
            {**hit, "taskId": "44"},
            {**hung, "taskId": "41"},
            {**policy, "taskId": "39"},
        ]
    )
    assert scope["waitKept"] == ["44"]
    assert scope["weighted"] == ["41"]
    assert scope["looped"] == ["39"]


def test_post_gate_39_44_obs_batch() -> None:
    """ICLR critic required log: post-gate airline 39/44. Hung is first-class."""
    obs39 = _obs(
        [
            {
                "kind": "text",
                "text": "I cannot cancel this economy reservation; a personal reason is not covered.",
                "ok": True,
            }
        ],
        0.0,
        [],
        reward_info={"reward": 0.0, "action_checks": [{"action": {"name": "cancel_reservation"}, "action_match": False}]},
        hung=False,
        task_id="39",
        termination="user_stop",
    )
    obs44 = _obs([], None, [], hung=True, task_id="44", termination="timeout")
    assert obs39["taskId"] == "39"
    assert obs39["arm"] == "I_loop"
    assert obs39["hung"] is False
    assert obs39["inventedPolicy"] is True
    assert recommend_intervention(obs39, loop_exhausted=False) == "I_loop"

    assert obs44["taskId"] == "44"
    assert obs44["hung"] is True
    assert obs44["nSuccessProxy"] == 0
    assert obs44["nSteps"] == 0
    assert obs44["lastActions"] == []
    assert obs44["arm"] == "I_sku"
    assert recommend_intervention(obs44, loop_exhausted=False) == "I_sku"
    assert recommend_intervention(obs44, loop_exhausted=True) == "I_sku"
    assert obs44["arm"] != "I_weight"
    old_rule_44 = "wait" if obs44["nSuccessProxy"] == 1 else "I_loop"
    assert old_rule_44 == "I_loop"
    assert obs44["arm"] != old_rule_44
    assert recommend_intervention(obs44, loop_exhausted=False) != "I_loop"

    from types import SimpleNamespace

    sim39 = SimpleNamespace(
        task_id="39",
        messages=[
            SimpleNamespace(
                role="assistant",
                content="I cannot cancel this economy reservation; a personal reason is not covered.",
                tool_calls=[],
            )
        ],
        hung=False,
        reward_info=SimpleNamespace(reward=0.0),
        termination_reason="user_stop",
    )
    batch = _collect_obs([sim39, HungSimulation("44", 0, "timeout")])
    assert [o["taskId"] for o in batch] == ["39", "44"]
    assert batch[0]["arm"] == "I_loop"
    assert batch[1]["arm"] == "I_sku"
    assert batch[1]["hung"] is True
    assert batch[1]["nSteps"] == 0
    scope = apply_scope_from_obs(batch)
    assert scope["waitKept"] == []
    assert scope["looped"] == ["39"]
    assert scope["weighted"] == ["44"]
    assert recommend_slice_intervention(batch, loop_exhausted=False) == "I_sku"
    ctrl = control_batch(batch, loop_exhausted=False)
    assert ctrl["episodes"][0]["arm"] == "I_loop"
    assert ctrl["episodes"][1]["arm"] == "I_sku"
    assert ctrl["episodes"][1]["hung"] is True
    assert ctrl["episodes"][1]["license"] == "hung"
    assert ctrl["slice"] == "I_sku"
    assert ctrl["buckets"]["39"] == "I_loop"
    assert ctrl["buckets"]["44"] == "I_sku"
    assert ctrl["applied"] == ["I_loop", "I_sku"]
    assert not (
        ctrl["slice"] == "I_sku"
        and "I_loop" not in ctrl["applied"]
        and "39" in ctrl["applyScope"]["looped"]
    ), "if only slice is consumed, 39 I_loop is dropped"
    assert ctrl["applyScope"]["waitKept"] == []
    assert ctrl["servingPaused"] is False
    assert ctrl["trained"] is False
    assert ctrl["serving"]["sku"] == SERVING_MODEL
    assert ctrl["servingSku"]["sku"] == SERVING_MODEL
    assert ctrl["episodes"][0]["serving"]["sku"] == SERVING_MODEL
    assert ctrl["episodes"][1]["serving"]["sku"] == SERVING_MODEL
    blob = " ".join(str(o.get("lastActions")) for o in batch)
    assert "MSJ4OA" not in blob
    assert "S61CZX" not in blob
    from tau2_vdom.improve import I_SKU_NOTE

    assert "catalog rebind" in I_SKU_NOTE
    assert "not fine-tuning" in I_SKU_NOTE
    assert "not pick a pricier model" in I_SKU_NOTE
    assert "0813 existing is not a gate" in I_SKU_NOTE
    assert CATALOG_JUMP_MODEL == "deepseek/deepseek-v4-pro-0813"


def test_i_loop_does_not_change_s() -> None:
    sidecar = default_sidecar()
    sidecar.request({"op": "set_technique", "technique": "one-shot"})
    miss39 = {
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
    loop = sidecar.request({"op": "i_loop", "obs": [miss39], "model": "deterministic"})
    assert loop.get("applied") is True
    assert loop.get("servingPaused") is False
    assert loop.get("serving", {}).get("sku") == SERVING_MODEL
    assert loop.get("servingSku", {}).get("sku") == SERVING_MODEL
    assert loop.get("servingModel") == SERVING_MODEL


def test_mixed_3944_s_not_identified_with_c() -> None:
    sidecar = default_sidecar()
    sidecar.request({"op": "set_technique", "technique": "one-shot"})
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
    loop = sidecar.request({"op": "i_loop", "obs": [obs39, obs44], "model": "deterministic"})
    scope = loop.get("applyScope") or {}
    assert scope.get("waitKept") == []
    assert scope.get("looped") == ["39"]
    assert scope.get("weighted") == ["44"]
    assert loop.get("servingSku", {}).get("sku") == SERVING_MODEL

    mount = _sidecar_catalog_jump(sidecar, before=0.0, after=1e-6)
    assert mount["servingPaused"] is False
    assert mount["trained"] is False
    assert mount["serving"]["sku"] == CATALOG_JUMP_MODEL
    solve = (mount.get("graph") or {}).get("root") or {}
    assert solve.get("model") in {None, SERVING_MODEL}

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
    assert turn44.get("servingModel") == CATALOG_JUMP_MODEL
    assert turn44.get("servingPaused") is False
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
    assert turn39.get("servingModel") == SERVING_MODEL
    assert turn39.get("servingPaused") is False

    # HybridState.S falsifier: a FRESH 39-only I_loop must not inherit process 0813.
    fresh = sidecar.request({"op": "i_loop", "obs": [obs39], "model": "deterministic"})
    assert fresh.get("applied") is True
    assert fresh.get("servingPaused") is False
    assert fresh.get("serving", {}).get("sku") == SERVING_MODEL
    assert fresh.get("servingSku", {}).get("sku") == SERVING_MODEL
    assert fresh.get("servingModel") == SERVING_MODEL
    turn_fresh = sidecar.request(
        {
            "op": "turn",
            "taskId": "39",
            "model": "deterministic",
            "policy": "",
            "tools": [],
            "messages": [{"role": "user", "content": "hello"}],
        }
    )
    assert turn_fresh.get("servingModel") == SERVING_MODEL
    assert turn_fresh.get("servingPaused") is False
    ping = sidecar.request({"op": "ping"})
    assert ping.get("servingModel") == SERVING_MODEL
    assert ping.get("servingSku", {}).get("sku") == SERVING_MODEL


def test_hit_still_waits() -> None:
    obs = _obs(
        [{"kind": "tool", "text": "cancel_reservation", "toolName": "cancel_reservation", "ok": True}],
        1.0,
        [],
        reward_info={"reward": 1.0, "action_checks": []},
    )
    assert obs["arm"] == "wait"
    assert obs["techniqueRecommendation"] is None


def main() -> int:
    tests = [
        test_obs_missed_cancel_recommends_policy,
        test_obs_mock_update_does_not_select_policy,
        test_obs_hung_keeps_task,
        test_typed_arms_hung_hit_policy,
        test_skipped_task_stays_in_task_phit,
        test_obs_personal_reason_is_invented_policy,
        test_missed_tool_names_only_drops_gold_ids,
        test_strip_kernel_self_tools_never_reach_env,
        test_obs_includes_task_id,
        test_apply_scope_mixed_wait_hit,
        test_collect_obs_sets_task_id,
        test_post_gate_39_44_obs_batch,
        test_i_loop_does_not_change_s,
        test_mixed_3944_s_not_identified_with_c,
        test_hit_still_waits,
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

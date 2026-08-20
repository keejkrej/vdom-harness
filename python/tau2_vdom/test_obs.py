"""Unit tests for failure-aware Obs and skip handling. No API key."""

from __future__ import annotations

from tau2_vdom.improve import (
    _collect_obs,
    _task_p_hit,
    apply_scope_from_obs,
    missed_tool_names_only,
    pass_hat_k_from_rewards,
)
from tau2_vdom.kernel_tools import strip_kernel_self_tools
from tau2_vdom.runner import (
    _obs,
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
    assert obs["arm"] == "I_weight"
    assert "null reward" in obs["critique"]
    assert recommend_intervention(obs) == "I_weight"


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
    assert hung["arm"] == "I_weight"
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
    assert recommend_slice_intervention([hit, hung]) == "I_weight"
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

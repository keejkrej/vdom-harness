"""Unit tests for failure-aware Obs and skip handling. No API key."""

from __future__ import annotations

from tau2_vdom.improve import (
    _task_p_hit,
    pass_hat_k_from_rewards,
)
from tau2_vdom.runner import _obs, serialize_reward_info


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
    assert "null reward" in obs["critique"]


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
        test_skipped_task_stays_in_task_phit,
        test_obs_personal_reason_is_invented_policy,
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

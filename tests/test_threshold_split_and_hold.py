"""Split session/weekly thresholds and the all-over-threshold hold.

Covers ``autoswitch.sessionThreshold`` / ``autoswitch.weeklyThreshold``
(inheriting ``autoswitch.threshold``), the per-account ``--session`` /
``--weekly`` overrides, ``autoswitch.holdWhenAllOverThreshold`` in the auto
engine and in ``cswap wrap``, and the reason/figures on the switch log line.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_swap import oauth, poll_policy, wrap
from claude_swap.autoswitch import (
    HoldEndEvent,
    HoldEvent,
    NoSwitchEvent,
    PollEvent,
    SwitchEvent,
    TickOutcome,
)
from claude_swap.exceptions import ConfigError
from claude_swap.settings import (
    SETTING_SPECS,
    AutoSwitchSettings,
    account_thresholds,
    effective_settings,
    load_settings,
    merged_with_cli,
    parse_setting_value,
    settings_path,
)
from tests.test_autoswitch import EngineHarness, _iso_at

NOW = 1_000_000.0


def _u(five: float, seven: float = 0.0, *, five_reset=None, seven_reset=None) -> dict:
    """Usage with both windows; resets are absolute epochs (or None)."""
    usage = {"five_hour": {"pct": five}, "seven_day": {"pct": seven}}
    if five_reset is not None:
        usage["five_hour"]["resets_at"] = _iso_at(five_reset)
    if seven_reset is not None:
        usage["seven_day"]["resets_at"] = _iso_at(seven_reset)
    return usage


def _harness(temp_home: Path, accounts: int = 3, **settings) -> EngineHarness:
    h = EngineHarness(temp_home, **settings)
    for n in range(1, accounts + 1):
        h.seed(n, f"{chr(96 + n)}@example.com")
    h.make_live("a@example.com", 1)
    return h


def _of(h: EngineHarness, cls) -> list:
    return [e for e in h.events if isinstance(e, cls)]


# ---------------------------------------------------------------------------
# Settings: inheritance, validation, CLI overlay
# ---------------------------------------------------------------------------


class TestSettingsInheritance:
    def _write(self, root: Path, section: dict) -> None:
        root.mkdir(parents=True, exist_ok=True)
        settings_path(root).write_text(json.dumps({"autoswitch": section}))

    def test_unset_split_keys_inherit_threshold(self, tmp_path):
        self._write(tmp_path, {"threshold": 80})
        s = load_settings(tmp_path)
        assert s.session_threshold is None and s.weekly_threshold is None
        assert s.effective_session_threshold == 80.0
        assert s.effective_weekly_threshold == 80.0
        assert s.hold_when_all_over_threshold is True

    def test_defaults_when_nothing_is_set(self, tmp_path):
        s = load_settings(tmp_path)
        assert (s.effective_session_threshold, s.effective_weekly_threshold) == (
            90.0,
            90.0,
        )

    def test_each_split_key_overrides_only_its_windows(self, tmp_path):
        self._write(tmp_path, {"threshold": 80, "sessionThreshold": 70})
        s = load_settings(tmp_path)
        assert s.effective_session_threshold == 70.0
        assert s.effective_weekly_threshold == 80.0
        self._write(tmp_path, {"weeklyThreshold": 97, "holdWhenAllOverThreshold": False})
        s = load_settings(tmp_path)
        assert s.effective_session_threshold == 90.0
        assert s.effective_weekly_threshold == 97.0
        assert s.hold_when_all_over_threshold is False

    def test_split_keys_are_clamped_like_threshold(self, tmp_path):
        self._write(tmp_path, {"sessionThreshold": 150, "weeklyThreshold": "x"})
        s = load_settings(tmp_path)
        assert s.session_threshold == 99.99
        assert s.weekly_threshold is None  # garbage -> unset -> inherits

    def test_config_set_validates_with_the_threshold_range(self):
        spec = SETTING_SPECS["autoswitch.sessionThreshold"]
        assert (spec.lo, spec.hi) == (
            SETTING_SPECS["autoswitch.threshold"].lo,
            SETTING_SPECS["autoswitch.threshold"].hi,
        )
        assert parse_setting_value(spec, "85") == 85.0
        with pytest.raises(ConfigError):
            parse_setting_value(spec, "120")
        with pytest.raises(ConfigError):
            parse_setting_value(SETTING_SPECS["autoswitch.weeklyThreshold"], "0")

    def test_config_list_shows_the_inherited_value(self, tmp_path):
        self._write(tmp_path, {"threshold": 75})
        rows = {spec.dotted: (value, is_set) for spec, value, is_set in effective_settings(tmp_path)}
        assert rows["autoswitch.sessionThreshold"] == (75.0, False)
        assert rows["autoswitch.weeklyThreshold"] == (75.0, False)
        assert rows["autoswitch.holdWhenAllOverThreshold"] == (True, False)

    def test_cli_threshold_replaces_the_split_values_unless_pinned(self):
        base = AutoSwitchSettings(session_threshold=70.0, weekly_threshold=95.0)
        merged = merged_with_cli(base, Namespace(threshold=80.0))
        assert (merged.effective_session_threshold, merged.effective_weekly_threshold) == (80.0, 80.0)
        merged = merged_with_cli(
            base, Namespace(threshold=80.0, weekly_threshold=97.0)
        )
        assert (merged.effective_session_threshold, merged.effective_weekly_threshold) == (80.0, 97.0)
        merged = merged_with_cli(base, Namespace(hold_when_all_over_threshold=False))
        assert merged.hold_when_all_over_threshold is False
        assert merged.session_threshold == 70.0

    def test_account_thresholds_precedence(self):
        s = AutoSwitchSettings(threshold=90.0, session_threshold=80.0)
        assert account_thresholds(s) == (80.0, 90.0)
        assert account_thresholds(s, (None, 97.0)) == (80.0, 97.0)
        assert account_thresholds(s, (60.0, 60.0)) == (60.0, 60.0)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestThresholdGap:
    def test_equal_thresholds_reduce_to_binding_pct(self):
        usage = _u(40, 85)
        assert oauth.threshold_gap(usage, (), 90, 90) == pytest.approx(85 - 90)

    def test_each_window_against_its_own_threshold(self):
        # 5h 85 vs session 80 -> +5; 7d 50 vs weekly 95 -> -45.
        assert oauth.threshold_gap(_u(85, 50), (), 80, 95) == pytest.approx(5)
        # 5h 50 vs 80 -> -30; 7d 96 vs 95 -> +1.
        assert oauth.threshold_gap(_u(50, 96), (), 80, 95) == pytest.approx(1)

    def test_per_model_windows_use_the_weekly_threshold(self):
        usage = _u(10, 10)
        usage["scoped"] = [{"name": "Fable", "pct": 92.0}]
        assert oauth.threshold_gap(usage, ("Fable",), 50, 90) == pytest.approx(2)
        assert oauth.threshold_gap(usage, (), 50, 90) == pytest.approx(-40)

    def test_unknown_usage_is_none(self):
        assert oauth.threshold_gap(None, (), 90, 90) is None


class TestUnderThresholdAt:
    def test_already_under_is_now(self):
        assert poll_policy.under_threshold_at(_u(10, 10), (), 90, 90, NOW) == NOW

    def test_latest_reset_of_the_over_windows(self):
        usage = _u(95, 96, five_reset=NOW + 600, seven_reset=NOW + 9000)
        assert poll_policy.under_threshold_at(usage, (), 90, 90, NOW) == pytest.approx(NOW + 9000)
        # Weekly threshold high enough that only the 5h window is over.
        assert poll_policy.under_threshold_at(usage, (), 90, 97, NOW) == pytest.approx(NOW + 600)

    def test_unprovable_reset_is_none(self):
        assert poll_policy.under_threshold_at(_u(95), (), 90, 90, NOW) is None
        stale = _u(95, five_reset=NOW - 5)
        assert poll_policy.under_threshold_at(stale, (), 90, 90, NOW) is None


class TestUrgentPolling:
    def _plan(self, new_usage, **kw):
        _, interval = poll_policy.plan_after_fetch(
            prev_interval_s=180.0,
            prev_usage=_u(20, 20),
            new_usage=new_usage,
            is_active=True,
            models=(),
            recent_429=False,
            now=NOW,
            rng=lambda: 0.5,
            **kw,
        )
        return interval

    def test_urgent_mode_keys_on_the_window_nearest_its_own_threshold(self):
        # 5h at 70: 30 points from a 100 session threshold, but 7d at 80 is
        # within the escalation margin of a weekly 90.
        usage = _u(70, 80)
        assert self._plan(usage, threshold=99.0, weekly_threshold=90.0) == (
            poll_policy.URGENT_INTERVAL_S
        )
        assert self._plan(usage, threshold=99.0, weekly_threshold=99.0) != (
            poll_policy.URGENT_INTERVAL_S
        )


# ---------------------------------------------------------------------------
# Engine: over-threshold by session only / weekly only
# ---------------------------------------------------------------------------


class TestSplitThresholdDecisions:
    def test_over_only_by_weekly_switches(self, temp_home):
        h = _harness(temp_home, session_threshold=80.0, weekly_threshold=90.0)
        outcome = h.tick_with_usage({
            "1": _u(50, 92), "2": _u(10, 20), "3": _u(40, 40),
        })
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 2
        assert _of(h, SwitchEvent)[0].trigger == "proactive"

    def test_over_only_by_session_switches(self, temp_home):
        h = _harness(temp_home, session_threshold=80.0, weekly_threshold=95.0)
        outcome = h.tick_with_usage({
            "1": _u(85, 10), "2": _u(5, 60), "3": _u(40, 70),
        })
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 2

    def test_a_single_threshold_would_not_have_switched(self, temp_home):
        # Same usage as above under the old single 90: the binding window is
        # 85% < 90 -> stays.
        h = _harness(temp_home, threshold=90.0)
        outcome = h.tick_with_usage({
            "1": _u(85, 10), "2": _u(5, 60), "3": _u(40, 40),
        })
        assert outcome is TickOutcome.NO_ACTION

    def test_under_both_stays_and_names_each_window(self, temp_home):
        h = _harness(temp_home, session_threshold=80.0, weekly_threshold=95.0)
        outcome = h.tick_with_usage({
            "1": _u(79, 94), "2": _u(5, 5), "3": _u(5, 5),
        })
        assert outcome is TickOutcome.NO_ACTION
        (event,) = _of(h, NoSwitchEvent)
        assert event.reason == "below-threshold"
        assert event.detail == "5h 79% < 80%, 7d 94% < 95%"

    def test_landing_skips_a_candidate_over_its_session_threshold(self, temp_home):
        # Account 2 has the most headroom (45) but its 5h window (55) is over
        # the 50 session threshold; account 3 (30 headroom) is under both.
        h = _harness(
            temp_home, session_threshold=50.0, weekly_threshold=95.0,
            hysteresis_pct=0.0,
        )
        outcome = h.tick_with_usage({
            "1": _u(90, 0), "2": _u(55, 0), "3": _u(10, 70),
        })
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 3

    def test_poll_event_shows_both_thresholds(self, temp_home):
        h = _harness(temp_home, session_threshold=80.0, weekly_threshold=95.0)
        h.tick_with_usage({"1": _u(10, 10), "2": _u(5, 5), "3": _u(5, 5)})
        (poll,) = _of(h, PollEvent)
        assert "switch at 5h 80% / weekly 95%" in poll.human()
        payload = poll.to_json()
        assert payload["sessionThreshold"] == 80.0
        assert payload["weeklyThreshold"] == 95.0

    def test_poll_event_unchanged_without_split(self, temp_home):
        h = _harness(temp_home)
        h.tick_with_usage({"1": _u(10, 10), "2": _u(5, 5), "3": _u(5, 5)})
        (poll,) = _of(h, PollEvent)
        assert "switch at 90%" in poll.human()
        assert "sessionThreshold" not in poll.to_json()

    def test_poll_pin_carries_the_weekly_threshold(self, temp_home):
        h = _harness(temp_home, session_threshold=80.0, weekly_threshold=95.0)
        assert h.switcher._poll_inputs_override == (80.0, ())
        assert h.switcher._poll_policy_weekly_threshold() == 95.0


class TestPerAccountSplitOverrides:
    def test_session_only_override(self, temp_home):
        h = _harness(temp_home)
        h.switcher.set_account_threshold("2", 70.0, "session")
        assert h.switcher.account_threshold_overrides("2") == (70.0, None)
        assert h.engine._thresholds_for("2") == (70.0, 90.0)

    def test_window_specific_override_beats_the_shared_one(self, temp_home):
        h = _harness(temp_home)
        h.switcher.set_account_threshold("2", 95.0)
        h.switcher.set_account_threshold("2", 80.0, "weekly")
        assert h.engine._thresholds_for("2") == (95.0, 80.0)
        # Setting the shared one again applies to both: drops the specific.
        h.switcher.set_account_threshold("2", 85.0)
        assert h.engine._thresholds_for("2") == (85.0, 85.0)

    def test_unset_one_kind_keeps_the_other(self, temp_home):
        h = _harness(temp_home)
        h.switcher.set_account_threshold("2", 70.0, "session")
        h.switcher.set_account_threshold("2", 97.0, "weekly")
        h.switcher.unset_account_threshold("2", "session")
        assert h.switcher.account_threshold_overrides("2") == (None, 97.0)
        h.switcher.unset_account_threshold("2")
        assert h.switcher.account_threshold_overrides("2") == (None, None)

    def test_listing_reports_every_kind(self, temp_home):
        h = _harness(temp_home)
        h.switcher.set_account_threshold("3", 95.0)
        h.switcher.set_account_threshold("2", 70.0, "session")
        rows = h.switcher.list_account_threshold_overrides()
        assert rows == [
            ("2", None, 70.0, None, "b@example.com"),
            ("3", 95.0, None, None, "c@example.com"),
        ]
        # The pre-split listing keeps its shape (shared overrides only).
        assert h.switcher.list_account_thresholds() == [("3", 95.0, "c@example.com")]

    def test_shared_override_still_applies_to_both_windows(self, temp_home):
        # Account 1 overridden to 50: its 7d at 60 is over even though the
        # fleet threshold (90) is far away.
        h = _harness(temp_home)
        h.switcher.set_account_threshold("1", 50.0)
        outcome = h.tick_with_usage({
            "1": _u(10, 60), "2": _u(5, 5), "3": _u(30, 30),
        })
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 2


# ---------------------------------------------------------------------------
# Engine: the all-over-threshold hold
# ---------------------------------------------------------------------------


class TestAllOverThresholdHold:
    def _all_over(self, h, *, active=100.0):
        now = h.clock.now
        return {
            "1": _u(active, 40, five_reset=now + 3600),
            "2": _u(95, 40, five_reset=now + 1800),
            "3": _u(92, 40, five_reset=now + 7200),
        }

    def test_at_limit_with_every_candidate_over_does_not_switch(self, temp_home):
        h = _harness(temp_home)
        outcome = h.tick_with_usage(self._all_over(h))
        assert outcome is TickOutcome.BLOCKED
        assert h.active_number() == 1
        assert not _of(h, SwitchEvent)
        (hold,) = _of(h, HoldEvent)
        # The earliest account back under its threshold: #2 at +1800s.
        assert hold.until == _iso_at(h.clock.now + 1800)
        assert "holding on Account-1" in hold.human()
        reasons = [e.reason for e in _of(h, NoSwitchEvent)]
        assert reasons == ["holding-all-over-threshold"]
        # Slow cadence: sleeps toward that release (capped by MAX_SLEEP_S).
        assert h.engine._sleep_until_ts == pytest.approx(
            h.clock.now + 1800 + poll_policy.RESET_SLACK_S
        )
        assert h.engine._next_delay(outcome) > h.settings.interval_seconds

    def test_proactive_with_every_candidate_over_does_not_switch(self, temp_home):
        h = _harness(temp_home, cooldown_seconds=0.0)
        outcome = h.tick_with_usage(self._all_over(h, active=96.0))
        assert outcome is TickOutcome.BLOCKED
        assert h.active_number() == 1

    def test_hold_is_announced_once_not_every_tick(self, temp_home):
        h = _harness(temp_home)
        for _ in range(4):
            h.tick_with_usage(self._all_over(h))
            h.clock.advance(60)
        assert len(_of(h, HoldEvent)) == 1
        assert not _of(h, HoldEndEvent)
        assert h.active_number() == 1

    def test_hold_survives_a_new_engine_process(self, temp_home):
        # cron `--once`: each tick is a fresh engine; the persisted flag keeps
        # the announcement single.
        h = _harness(temp_home)
        h.tick_with_usage(self._all_over(h))
        assert h.state()["hold"]["until"] is not None
        h.engine = h._make_engine()
        h.tick_with_usage(self._all_over(h))
        assert len(_of(h, HoldEvent)) == 1

    def test_a_candidate_under_threshold_takes_the_at_limit_escape(self, temp_home):
        h = _harness(temp_home)
        now = h.clock.now
        outcome = h.tick_with_usage({
            "1": _u(100, 40, five_reset=now + 3600),
            "2": _u(95, 40),
            "3": _u(85, 40),
        })
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 3
        assert _of(h, SwitchEvent)[0].trigger == "at-limit"
        assert not _of(h, HoldEvent)

    def test_at_limit_prefers_under_threshold_over_more_headroom(self, temp_home):
        # Account 2 has more headroom (40) but its own threshold (50) puts it
        # over; account 3 (15 headroom) is under the fleet 90.
        h = _harness(temp_home)
        h.switcher.set_account_threshold("2", 50.0)
        outcome = h.tick_with_usage({
            "1": _u(100, 40), "2": _u(60, 40), "3": _u(85, 40),
        })
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 3

    def test_hold_off_restores_the_old_escape(self, temp_home):
        h = _harness(temp_home, hold_when_all_over_threshold=False)
        outcome = h.tick_with_usage(self._all_over(h))
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() != 1
        assert _of(h, SwitchEvent)[0].trigger == "at-limit"
        assert not _of(h, HoldEvent)

    def test_hold_off_at_limit_still_lands_over_threshold_by_headroom(self, temp_home):
        h = _harness(temp_home, hold_when_all_over_threshold=False)
        h.switcher.set_account_threshold("2", 50.0)
        outcome = h.tick_with_usage({
            "1": _u(100, 40), "2": _u(60, 40), "3": _u(85, 40),
        })
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 2  # most headroom, as before

    def test_leaves_the_hold_when_a_candidate_resets(self, temp_home):
        h = _harness(temp_home)
        h.tick_with_usage(self._all_over(h))
        assert h.active_number() == 1
        h.clock.advance(1800 + 60)
        now = h.clock.now
        outcome = h.tick_with_usage({
            "1": _u(100, 40, five_reset=now + 1700),
            "2": _u(0, 40, five_reset=now + 18000),   # its 5h rolled over
            "3": _u(92, 40, five_reset=now + 5300),
        })
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 2
        (end,) = _of(h, HoldEndEvent)
        assert "Account-2" in end.reason
        assert "hold" not in h.state()
        # Order: the hold ends before the switch is reported.
        kinds = h.kinds()
        assert kinds.index("hold-end") < kinds.index("switch")

    def test_leaves_the_hold_when_the_active_account_resets(self, temp_home):
        h = _harness(temp_home)
        h.tick_with_usage(self._all_over(h))
        h.clock.advance(3600 + 60)
        outcome = h.tick_with_usage({
            "1": _u(0, 40), "2": _u(95, 40), "3": _u(92, 40),
        })
        assert outcome is TickOutcome.NO_ACTION
        assert h.active_number() == 1
        (end,) = _of(h, HoldEndEvent)
        assert "Account-1 (active)" in end.reason
        assert [e.reason for e in _of(h, NoSwitchEvent)][-1] == "below-threshold"

    def test_every_candidate_at_its_limit_is_still_all_exhausted(self, temp_home):
        # Nothing to hold back from: the old all-exhausted path reports it.
        h = _harness(temp_home)
        now = h.clock.now
        outcome = h.tick_with_usage({
            "1": _u(100, 40, five_reset=now + 3600),
            "2": _u(100, 40, five_reset=now + 1800),
            "3": _u(100, 40, five_reset=now + 7200),
        })
        assert outcome is TickOutcome.BLOCKED
        assert "all-exhausted" in h.kinds()

    def test_hold_by_weekly_window_only(self, temp_home):
        # Session threshold out of reach; every account's 7d is over the
        # weekly 80 -> hold even though no 5h window is anywhere near.
        h = _harness(temp_home, session_threshold=99.0, weekly_threshold=80.0)
        now = h.clock.now
        outcome = h.tick_with_usage({
            "1": _u(10, 85, seven_reset=now + 86400),
            "2": _u(10, 82, seven_reset=now + 3 * 86400),
            "3": _u(10, 90, seven_reset=now + 2 * 86400),
        })
        assert outcome is TickOutcome.BLOCKED
        assert h.active_number() == 1
        (hold,) = _of(h, HoldEvent)
        assert hold.until is not None

    def test_dry_run_holds_without_writing_state(self, temp_home):
        h = _harness(temp_home)
        h.engine = h._make_engine(dry_run=True)
        h.tick_with_usage(self._all_over(h))
        h.tick_with_usage(self._all_over(h))
        assert len(_of(h, HoldEvent)) == 1
        assert "hold" not in h.state()


class TestSwitchLogReason:
    def test_switch_log_line_names_trigger_and_figures(self, temp_home):
        h = _harness(temp_home)
        messages: list[str] = []
        real_info = h.switcher._logger.info

        def capture(msg, *args, **kwargs):
            messages.append(str(msg))
            return real_info(msg, *args, **kwargs)

        with patch.object(h.switcher._logger, "info", side_effect=capture):
            outcome = h.tick_with_usage({
                "1": _u(100, 62), "2": _u(12, 40), "3": _u(95, 10),
            })
        assert outcome is TickOutcome.SWITCHED
        line = next(m for m in messages if m.startswith("Switched from account"))
        assert line == (
            "Switched from account 1 to 2 (auto at-limit; "
            "from #1 5h 100% · 7d 62% -> to #2 5h 12% · 7d 40%)"
        )
        # The menu bar's history parser still reads it.
        from claude_swap.menubar import _SWITCH_LOG_RE

        assert _SWITCH_LOG_RE.search(line).groups() == ("1", "2")
        sw = _of(h, SwitchEvent)[0]
        assert sw.detail.startswith("from #1 5h 100%")
        assert "at-limit; from #1" in sw.human()


# ---------------------------------------------------------------------------
# cswap wrap
# ---------------------------------------------------------------------------


class _Entry:
    def __init__(self, value):
        self._value = value

    def decision_value(self):
        return self._value


class _WrapSwitcher:
    def __init__(self, snapshots, switch_results):
        self._snapshots = list(snapshots)
        self._results = list(switch_results)
        self.switch_calls = 0

    def switch(self, strategy=None, json_output=False, models=()):
        self.switch_calls += 1
        return self._results.pop(0)

    def _build_accounts_info(self):
        return [
            (1, "a@example.com", "", "", True, "", ""),
            (2, "b@example.com", "", "", False, "", ""),
        ]

    def _collect_usage_entries(self, accounts_info):
        snapshot = self._snapshots[0]
        if len(self._snapshots) > 1:
            self._snapshots.pop(0)
        return {num: _Entry(value) for num, value in snapshot.items()}

    def switchable_account_numbers(self):
        return ["1", "2"]


_SWITCHED = {
    "switched": True,
    "from": {"number": 1, "email": "a@example.com"},
    "to": {"number": 2, "email": "b@example.com"},
    "reason": "switched",
}


def _wrap_session(switcher, *, hold=True, clock=lambda: NOW):
    lines: list[str] = []
    session = wrap.WrapSession(
        switcher,
        [],
        out=lines.append,
        clock=clock,
        thresholds_for=lambda num: (90.0, 90.0),
        hold_when_all_over=hold,
    )
    session.injected = []
    session._inject = session.injected.append
    session.lines = lines
    return session


class TestWrapHold:
    def test_does_not_rotate_onto_an_over_threshold_account(self, monkeypatch):
        monkeypatch.setattr(wrap, "CONTINUE_DELAY_S", 0.0)
        held = {
            "1": _u(100, 40, five_reset=NOW + 3600),
            "2": _u(95, 40, five_reset=NOW + 1800),
        }
        released = {
            "1": _u(100, 40, five_reset=NOW + 3600),
            "2": _u(0, 40),
        }
        switcher = _WrapSwitcher([held, released], [_SWITCHED])
        session = _wrap_session(switcher)
        sleeps: list[float] = []
        session._sleep = sleeps.append
        session._recover("You've hit your session limit")
        # One wait (toward #2's reset), then exactly one switch after it.
        assert sleeps == [pytest.approx(1800 + poll_policy.RESET_SLACK_S)]
        assert switcher.switch_calls == 1
        assert session.injected == [b"continue\r"]
        text = "\n".join(session.lines)
        assert "all accounts are over their threshold: holding on Account-1" in text
        assert text.count("holding on Account-1") == 1
        assert "hold over" in text

    def test_resumes_on_the_active_account_when_it_resets_first(self, monkeypatch):
        monkeypatch.setattr(wrap, "CONTINUE_DELAY_S", 0.0)
        held = {
            "1": _u(100, 40, five_reset=NOW + 600),
            "2": _u(95, 40, five_reset=NOW + 1800),
        }
        back = {"1": _u(0, 40), "2": _u(95, 40, five_reset=NOW + 1800)}
        switcher = _WrapSwitcher([held, back], [])
        session = _wrap_session(switcher)
        session._sleep = lambda s: None
        session._recover("You've hit your session limit")
        assert switcher.switch_calls == 0
        assert session.injected == [b"continue\r"]
        assert any("quota is back on Account-1" in ln for ln in session.lines)

    def test_active_with_quota_and_everyone_over_does_not_wait(self):
        snap = {"1": _u(95, 40), "2": _u(93, 40)}
        switcher = _WrapSwitcher([snap], [])
        session = _wrap_session(switcher)

        def no_sleep(seconds):
            pytest.fail("waited although the active account has quota")

        session._sleep = no_sleep
        session._recover("You've hit your Opus limit")
        assert switcher.switch_calls == 0
        assert session.injected == []

    def test_a_candidate_under_threshold_switches_immediately(self, monkeypatch):
        monkeypatch.setattr(wrap, "CONTINUE_DELAY_S", 0.0)
        snap = {"1": _u(100, 40), "2": _u(50, 40)}
        switcher = _WrapSwitcher([snap], [_SWITCHED])
        session = _wrap_session(switcher)
        session._sleep = lambda s: pytest.fail("held with a healthy candidate")
        session._recover("You've hit your session limit")
        assert switcher.switch_calls == 1
        assert session.injected == [b"continue\r"]

    def test_hold_off_rotates_as_before(self, monkeypatch):
        monkeypatch.setattr(wrap, "CONTINUE_DELAY_S", 0.0)
        snap = {"1": _u(100, 40), "2": _u(95, 40)}
        switcher = _WrapSwitcher([snap], [_SWITCHED])
        session = _wrap_session(switcher, hold=False)
        session._sleep = lambda s: pytest.fail("hold is off")
        session._recover("You've hit your session limit")
        assert switcher.switch_calls == 1


# ---------------------------------------------------------------------------
# `cswap threshold --session/--weekly`
# ---------------------------------------------------------------------------


class TestThresholdCommandSplit:
    def _seed(self):
        from claude_swap.switcher import ClaudeAccountSwitcher

        switcher = ClaudeAccountSwitcher()
        switcher._setup_directories()
        switcher._init_sequence_file()
        data = switcher._get_sequence_data()
        data["accounts"]["2"] = {
            "email": "work@co.com",
            "uuid": "u2",
            "organizationUuid": "",
            "organizationName": "",
            "added": "2024-01-01T00:00:00Z",
        }
        data["sequence"] = [2]
        switcher._write_json(switcher.sequence_file, data)

    def _record(self) -> dict:
        from claude_swap.switcher import ClaudeAccountSwitcher

        return ClaudeAccountSwitcher()._get_sequence_data()["accounts"]["2"]

    def test_session_and_weekly_flags(self, temp_home, capsys):
        from claude_swap import cli

        self._seed()
        with patch("os.geteuid", return_value=1000, create=True):
            cli._threshold_command(["2", "80", "--session"])
            cli._threshold_command(["2", "97", "--weekly"])
        record = self._record()
        assert record["sessionThresholdOverride"] == 80.0
        assert record["weeklyThresholdOverride"] == 97.0
        assert "thresholdOverride" not in record
        with patch("os.geteuid", return_value=1000, create=True):
            cli._threshold_command([])
        out = capsys.readouterr().out
        assert "5h 80%" in out and "weekly 97%" in out
        with patch("os.geteuid", return_value=1000, create=True):
            cli._threshold_command(["2", "--unset", "--weekly"])
        record = self._record()
        assert "weeklyThresholdOverride" not in record
        assert record["sessionThresholdOverride"] == 80.0

    def test_session_and_weekly_are_exclusive(self, temp_home):
        from claude_swap import cli

        self._seed()
        with patch("os.geteuid", return_value=1000, create=True):
            with pytest.raises(SystemExit):
                cli._threshold_command(["2", "80", "--session", "--weekly"])

"""Tests for the PTY wrapper (cswap wrap)."""

from __future__ import annotations

import logging
import os
import stat
import time

import pytest

from claude_swap import poll_policy, wrap
from claude_swap.wrap import WrapSession, earliest_recovery_ts, find_limit_message


# ---------------------------------------------------------------------------
# Limit-message detection
# ---------------------------------------------------------------------------


class TestLimitDetection:
    @pytest.mark.parametrize(
        "text",
        [
            "You've hit your session limit",
            "You've hit your weekly limit",
            "You've hit your Opus limit",
            "You've hit your Sonnet limit",
            "You've hit your Fable limit",
            "You've hit your fast limit",
            "You've hit your monthly limit — raise it below",
            "You've hit your monthly spend limit · your X resets 5pm",
            "You've hit your channel's monthly spend limit.",
            "You've hit your team’s shared budget. Run /usage-credits",
            "You've hit your team's shared budget.",
        ],
    )
    def test_known_variants(self, text):
        assert find_limit_message(text) is not None

    @pytest.mark.parametrize(
        "text",
        [
            "everything is fine",
            "you have hit the road",
            "approaching your session limit",  # warning, not exhaustion
            "",
        ],
    )
    def test_negatives(self, text):
        assert find_limit_message(text) is None

    def test_ansi_split_message(self):
        raw = b"\x1b[1mYou've hit your \x1b[0msession limit\x1b[2m"
        assert find_limit_message(wrap.strip_ansi(raw)) == (
            "You've hit your session limit"
        )

    def test_osc_sequence_stripped(self):
        raw = b"\x1b]0;title\x07You've hit your weekly limit"
        assert find_limit_message(wrap.strip_ansi(raw)) is not None


# ---------------------------------------------------------------------------
# earliest_recovery_ts
# ---------------------------------------------------------------------------


def _usage(five_hour=None, seven_day=None, scoped=None):
    """Build an internal (snake_case) usage dict."""
    usage = {}
    if five_hour is not None:
        usage["five_hour"] = five_hour
    if seven_day is not None:
        usage["seven_day"] = seven_day
    if scoped is not None:
        usage["scoped"] = scoped
    return usage


def _window(pct, resets_at):
    return {"pct": pct, "resets_at": resets_at}


class TestEarliestRecovery:
    NOW = 1_800_000_000.0

    def _iso(self, delta_s):
        from datetime import datetime, timezone

        return datetime.fromtimestamp(self.NOW + delta_s, tz=timezone.utc).isoformat()

    def test_single_exhausted_account(self):
        usage = {"1": _usage(five_hour=_window(100.0, self._iso(3600)))}
        assert earliest_recovery_ts(usage, (), self.NOW) == pytest.approx(
            self.NOW + 3600
        )

    def test_per_account_latest_reset_gates(self):
        """Blocked on 5h AND a scoped window: usable at the LATER reset."""
        usage = {
            "1": _usage(
                five_hour=_window(100.0, self._iso(3600)),
                scoped=[{"name": "Fable", "pct": 100.0, "resets_at": self._iso(7200)}],
            )
        }
        assert earliest_recovery_ts(usage, ("Fable",), self.NOW) == pytest.approx(
            self.NOW + 7200
        )

    def test_scoped_window_ignored_without_model_config(self):
        """A scoped limit doesn't count when models weren't configured."""
        usage = {
            "1": _usage(
                five_hour=_window(100.0, self._iso(3600)),
                scoped=[{"name": "Fable", "pct": 100.0, "resets_at": self._iso(7200)}],
            )
        }
        assert earliest_recovery_ts(usage, (), self.NOW) == pytest.approx(
            self.NOW + 3600
        )

    def test_minimum_across_accounts(self):
        usage = {
            "1": _usage(five_hour=_window(100.0, self._iso(7200))),
            "2": _usage(five_hour=_window(100.0, self._iso(1800))),
        }
        assert earliest_recovery_ts(usage, (), self.NOW) == pytest.approx(
            self.NOW + 1800
        )

    def test_unknown_usage_is_unprovable(self):
        usage = {
            "1": _usage(five_hour=_window(100.0, self._iso(3600))),
            "2": None,
        }
        assert earliest_recovery_ts(usage, (), self.NOW) is None

    def test_exhausted_without_reset_is_unprovable(self):
        usage = {"1": _usage(five_hour=_window(100.0, None))}
        assert earliest_recovery_ts(usage, (), self.NOW) is None

    def test_healthy_account_does_not_gate(self):
        usage = {
            "1": _usage(five_hour=_window(100.0, self._iso(3600))),
            "2": _usage(five_hour=_window(40.0, self._iso(100))),
        }
        assert earliest_recovery_ts(usage, (), self.NOW) == pytest.approx(
            self.NOW + 3600
        )

    def test_nobody_exhausted(self):
        usage = {"1": _usage(five_hour=_window(50.0, self._iso(3600)))}
        assert earliest_recovery_ts(usage, (), self.NOW) is None


# ---------------------------------------------------------------------------
# WrapSession recovery flow (fake switcher)
# ---------------------------------------------------------------------------


class _FakeEntry:
    def __init__(self, value):
        self._value = value

    def decision_value(self):
        return self._value


class FakeSwitcher:
    """Minimal stand-in for ClaudeAccountSwitcher (no real store access).

    Account 1 is the active one. ``usage_snapshots`` maps account number to
    usage, one snapshot per usage read; the last one repeats. With none
    given, usage is unknown.
    """

    def __init__(self, switch_results, usage_snapshots=()):
        self._switch_results = list(switch_results)
        self._usage_snapshots = list(usage_snapshots)
        self.switch_calls = []
        self.usage_fetches = []

    def switch(self, strategy=None, json_output=False, models=()):
        self.switch_calls.append(
            {"strategy": strategy, "json_output": json_output, "models": models}
        )
        return self._switch_results.pop(0)

    def _build_accounts_info(self):
        return [
            (1, "a@example.com", "", "", True, "", ""),
            (2, "b@example.com", "", "", False, "", ""),
        ]

    def _collect_usage_entries(self, accounts_info, fetch=None):
        self.usage_fetches.append(fetch)
        if not self._usage_snapshots:
            return {}
        snapshot = self._usage_snapshots[0]
        if len(self._usage_snapshots) > 1:
            self._usage_snapshots.pop(0)
        return {num: _FakeEntry(usage) for num, usage in snapshot.items()}

    def switchable_account_numbers(self):
        return ["1", "2"]


def _switched_result(number=2, email="b@example.com"):
    return {
        "switched": True,
        "from": {"number": 1, "email": "a@example.com"},
        "to": {"number": number, "email": email},
        "strategy": "best",
        "reason": "switched",
        "message": "switched",
        "warnings": [],
    }


def _noop_result(reason="no-better-candidate"):
    return {
        "switched": False,
        "from": {"number": 1, "email": "a@example.com"},
        "to": {"number": 1, "email": "a@example.com"},
        "strategy": "best",
        "reason": reason,
        "message": "stayed",
        "warnings": [],
    }


class _FakeSession(WrapSession):
    """Captures injections and status lines; skips the real PTY."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.injected = []
        self.lines = []

    def _inject(self, data):
        self.injected.append(data)

    def _out_default(self, line):
        self.lines.append(line)


def _make_session(switcher, **kwargs):
    lines = []
    notifications = []
    session = _FakeSession(
        switcher, [], out=lines.append, notify=notifications.append, **kwargs
    )
    session.status_lines = lines
    session.notifications = notifications
    return session


class TestRecovery:
    def test_switch_and_continue(self):
        switcher = FakeSwitcher([_switched_result()])
        session = _make_session(switcher)
        session._recover("You've hit your session limit")
        assert switcher.switch_calls == [
            {"strategy": "best", "json_output": True, "models": ()}
        ]
        assert session.injected == [b"continue\r"]
        assert any("switched to Account-2" in line for line in session.status_lines)

    def test_no_auto_continue(self):
        switcher = FakeSwitcher([_switched_result()])
        session = _make_session(switcher, auto_continue=False)
        session._recover("You've hit your session limit")
        assert session.injected == []

    def test_strategy_and_models_forwarded(self):
        switcher = FakeSwitcher([_switched_result()])
        session = _make_session(
            switcher, strategy="next-available", models=("Fable",)
        )
        session._recover("You've hit your Fable limit")
        assert switcher.switch_calls[0]["strategy"] == "next-available"
        assert switcher.switch_calls[0]["models"] == ("Fable",)

    def test_all_exhausted_waits_then_recovers(self):
        """First switch finds nothing; after the wait the retry switches."""
        switcher = FakeSwitcher([_noop_result(), _switched_result(number=3)])
        session = _make_session(switcher)
        session._compute_wait_s = lambda snapshot: 0.0  # don't really wait
        session._recover("You've hit your session limit")
        assert len(switcher.switch_calls) == 2
        assert session.injected == [b"continue\r"]

    def test_unknown_reset_uses_fallback_cadence(self):
        switcher = FakeSwitcher([_noop_result(), _switched_result()])
        session = _make_session(switcher)
        waits = []
        session._compute_wait_s = lambda snapshot: None
        orig_sleep = session._sleep
        session._sleep = lambda s: (waits.append(s), orig_sleep(0))[0]
        session._recover("You've hit your session limit")
        assert waits == [wrap.NO_RESET_FALLBACK_S]

    NOW = 1_800_000_000.0

    def _iso(self, delta_s):
        from datetime import datetime, timezone

        return datetime.fromtimestamp(self.NOW + delta_s, tz=timezone.utc).isoformat()

    def test_active_account_with_quota_gives_up_after_one_recheck(self):
        """The account in use has headroom, so there is no quota to wait for
        even though another account is exhausted. One re-check covers a stale
        reading; nothing proves the turn was aborted, so nothing is typed."""
        usage = {
            "1": _usage(five_hour=_window(7.0, None), seven_day=_window(40.0, None)),
            "2": _usage(seven_day=_window(100.0, self._iso(56_000))),
        }
        switcher = FakeSwitcher(
            [_noop_result("already-best"), _noop_result("already-best")], [usage]
        )
        session = _make_session(switcher, clock=lambda: self.NOW)
        sleeps = []

        def bounded_sleep(seconds):
            sleeps.append(seconds)
            if len(sleeps) > 1:
                pytest.fail("kept waiting for quota the active account already has")

        session._sleep = bounded_sleep
        session._recover("You've hit your session limit")
        assert len(sleeps) == 1
        assert sleeps[0] > poll_policy.SERVE_TTL_S
        assert len(switcher.switch_calls) == 2
        assert session.injected == []
        assert not any("exhausted" in line for line in session.status_lines)

    def test_recheck_catches_a_stale_reading(self, monkeypatch):
        """Right after a real limit the cached reading can still show
        headroom. Once it is old enough to be refetched, a fresh reading of
        the active account reveals the limit and the switch goes through."""
        monkeypatch.setattr(wrap, "CONTINUE_DELAY_S", 0.0)
        stale = {
            "1": _usage(five_hour=_window(95.0, None)),
            "2": _usage(five_hour=_window(10.0, None)),
        }
        fresh = {
            "1": _usage(five_hour=_window(100.0, self._iso(3600))),
            "2": _usage(five_hour=_window(10.0, None)),
        }
        switcher = FakeSwitcher(
            [_noop_result("already-best"), _switched_result()], [stale, fresh]
        )
        session = _make_session(switcher, clock=lambda: self.NOW)
        sleeps = []

        def bounded_sleep(seconds):
            sleeps.append(seconds)
            if len(sleeps) > 1:
                pytest.fail("re-checked more than once")

        session._sleep = bounded_sleep
        session._recover("You've hit your session limit")
        assert {"1"} in switcher.usage_fetches
        assert len(switcher.switch_calls) == 2
        assert session.injected == [b"continue\r"]

    def test_waits_then_resumes_on_the_active_account(self, monkeypatch):
        """Quota returns on the account already in use: there is nothing to
        switch to, but the aborted turn still has to be resumed."""
        monkeypatch.setattr(wrap, "CONTINUE_DELAY_S", 0.0)
        exhausted = {"1": _usage(five_hour=_window(100.0, self._iso(3600)))}
        recovered = {"1": _usage(five_hour=_window(0.0, None))}
        switcher = FakeSwitcher(
            [_noop_result("candidates-exhausted"), _noop_result("already-best")],
            [exhausted, recovered],
        )
        session = _make_session(switcher, clock=lambda: self.NOW)
        sleeps = []

        def one_sleep(seconds):
            sleeps.append(seconds)
            if len(sleeps) > 1:
                pytest.fail("kept waiting after quota returned")

        session._sleep = one_sleep
        session._recover("You've hit your session limit")
        assert len(switcher.switch_calls) == 2
        assert session.injected == [b"continue\r"]

    def test_debounce(self):
        switcher = FakeSwitcher([_switched_result()])
        session = _make_session(switcher)
        session.feed_output(b"You've hit your session limit")
        # The recovery thread needs a moment; wait for it.
        for _ in range(100):
            if session.injected:
                break
            time.sleep(0.05)
        assert session.injected == [b"continue\r"]
        # A reprint inside the debounce window must not trigger again.
        session.feed_output(b"You've hit your session limit")
        time.sleep(0.3)
        assert len(switcher.switch_calls) == 1

    def test_trigger_consumes_the_matched_output(self, monkeypatch):
        """A notice that already started a recovery must not start another
        one just because it is still inside the scan window."""
        monkeypatch.setattr(wrap, "CONTINUE_DELAY_S", 0.0)
        now = [self.NOW]
        switcher = FakeSwitcher([_switched_result(), _switched_result()])
        session = _make_session(switcher, clock=lambda: now[0])
        session.feed_output(b"You've hit your session limit")
        for _ in range(100):
            if session.injected and not session._recovery_lock.locked():
                break
            time.sleep(0.05)
        assert session.injected == [b"continue\r"]
        now[0] += wrap.TRIGGER_DEBOUNCE_S + 1
        session.feed_output(b"unrelated output")
        time.sleep(0.3)
        assert len(switcher.switch_calls) == 1


class TestStatusChannels:
    NOW = 1_800_000_000.0

    def _iso(self, delta_s):
        from datetime import datetime, timezone

        return datetime.fromtimestamp(self.NOW + delta_s, tz=timezone.utc).isoformat()

    def test_status_goes_to_the_log_and_never_to_the_terminal(
        self, capsys, caplog, monkeypatch
    ):
        """Anything written to the terminal lands inside claude's TUI and
        breaks its redraw, so status lines go to the log file instead."""
        logger = logging.getLogger("claude-swap")
        monkeypatch.setattr(logger, "handlers", [])
        session = WrapSession(FakeSwitcher([]), [])
        with caplog.at_level(logging.INFO, logger="claude-swap"):
            session._status("switch failed: boom")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""
        assert "wrap: switch failed: boom" in caplog.text

    def test_switch_and_resume_notifies_once(self, monkeypatch):
        monkeypatch.setattr(wrap, "CONTINUE_DELAY_S", 0.0)
        session = _make_session(FakeSwitcher([_switched_result()]))
        session._recover("You've hit your session limit")
        assert len(session.notifications) == 1
        assert "Account-2" in session.notifications[0]
        assert "resumed" in session.notifications[0]

    def test_waiting_for_quota_notifies_only_the_first_time(self, monkeypatch):
        """The wait is re-evaluated on a cadence; only its start is news."""
        monkeypatch.setattr(wrap, "CONTINUE_DELAY_S", 0.0)
        exhausted = {
            "1": _usage(five_hour=_window(100.0, self._iso(3600))),
            "2": _usage(five_hour=_window(100.0, self._iso(7200))),
        }
        switcher = FakeSwitcher(
            [
                _noop_result("candidates-exhausted"),
                _noop_result("candidates-exhausted"),
                _switched_result(),
            ],
            [exhausted],
        )
        session = _make_session(switcher, clock=lambda: self.NOW)
        sleeps = []

        def bounded_sleep(seconds):
            sleeps.append(seconds)
            if len(sleeps) > 2:
                pytest.fail("waited more often than the scripted switch results")

        session._sleep = bounded_sleep
        session._recover("You've hit your session limit")
        waiting = [n for n in session.notifications if "all accounts exhausted" in n]
        assert len(waiting) == 1
        assert "1h 1m" in waiting[0]
        logged = [s for s in session.status_lines if "all accounts exhausted" in s]
        assert len(logged) == 2

    def test_giving_up_on_a_healthy_account_notifies(self):
        """The turn may be sitting aborted; the user has to hear about it."""
        usage = {"1": _usage(five_hour=_window(7.0, None))}
        switcher = FakeSwitcher(
            [_noop_result("already-best"), _noop_result("already-best")], [usage]
        )
        session = _make_session(switcher, clock=lambda: self.NOW)
        session._sleep = lambda seconds: None
        session._recover("You've hit your session limit")
        assert len(session.notifications) == 1
        assert "type 'continue'" in session.notifications[0]

    def test_macos_notification_passes_text_as_arguments(self, monkeypatch):
        """Status text carries account emails and terminal output; it must
        never be spliced into the AppleScript source."""
        calls = []
        monkeypatch.setattr(
            wrap.subprocess, "run", lambda argv, **kwargs: calls.append(argv)
        )
        hostile = 'x" & do shell script "id'
        wrap.notify_macos(hostile, subtitle="my-project")
        (argv,) = calls
        assert argv[0] == "osascript"
        assert argv[-2:] == [hostile, "my-project"]
        assert not any("do shell script" in part for part in argv[:-2])

    def test_a_failing_notification_never_breaks_the_wrapper(self, monkeypatch):
        def boom(argv, **kwargs):
            raise OSError("osascript is gone")

        monkeypatch.setattr(wrap.subprocess, "run", boom)
        wrap.notify_macos("message", subtitle="my-project")

    def test_no_notifier_outside_macos(self, monkeypatch):
        def unexpected(argv, **kwargs):
            pytest.fail("spawned a notifier on a platform without one")

        monkeypatch.setattr(wrap.sys, "platform", "linux")
        monkeypatch.setattr(wrap.subprocess, "run", unexpected)
        session = WrapSession(FakeSwitcher([]), [], out=lambda line: None)
        session._status("quota is back", notify=True)


# ---------------------------------------------------------------------------
# End-to-end PTY: fake claude that hits a limit and reads the injected input
# ---------------------------------------------------------------------------


class TestPtyIntegration:
    def test_continue_is_typed_into_claude(self, tmp_path):
        transcript = tmp_path / "transcript.txt"
        fake_claude = tmp_path / "fake_claude.sh"
        fake_claude.write_text(
            "#!/bin/bash\n"
            'echo "You\'ve hit your session limit"\n'
            "read -r line\n"
            f"echo \"GOT:$line\" > {transcript}\n"
        )
        fake_claude.chmod(fake_claude.stat().st_mode | stat.S_IXUSR)

        switcher = FakeSwitcher([_switched_result()])
        session = WrapSession(
            switcher,
            [],
            claude_bin=str(fake_claude),
            out=lambda line: None, notify=lambda message: None,
        )
        # The continue injection happens CONTINUE_DELAY_S after the switch.
        exit_code = session.run()
        assert exit_code == 0
        assert transcript.read_text().strip() == "GOT:continue"
        assert switcher.switch_calls

    def test_exit_code_propagates(self, tmp_path):
        fake_claude = tmp_path / "fake_claude.sh"
        fake_claude.write_text("#!/bin/bash\nexit 42\n")
        fake_claude.chmod(fake_claude.stat().st_mode | stat.S_IXUSR)

        session = WrapSession(
            FakeSwitcher([]), [], claude_bin=str(fake_claude), out=lambda line: None, notify=lambda message: None,
        )
        assert session.run() == 42

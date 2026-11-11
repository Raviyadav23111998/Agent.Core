"""DeepSeek billed direct: its own list, time-of-day priced (2026-08-17).

The ledger is what budgets are enforced on, so a wrong price here is a wrong
limit: priced at a router's rate, a DeepSeek-direct month would read as a
fraction of what DeepSeek deducts.
"""
import datetime as dt

import pytest

from core import telemetry as T


def utc(y, mo, d, h, mi=0):
    return dt.datetime(y, mo, d, h, mi, tzinfo=dt.timezone.utc)


@pytest.fixture(autouse=True)
def no_config(monkeypatch):
    values = {}
    monkeypatch.setattr(T, "_config", lambda k: values.get(k, ""))
    return values


# 2026-10-12 is a Monday.
@pytest.mark.parametrize("at, peak", [
    (utc(2026, 10, 12, 0, 59), False),
    (utc(2026, 10, 12, 1, 0), True),     # 06:30 IST
    (utc(2026, 10, 12, 3, 59), True),
    (utc(2026, 10, 12, 4, 0), False),    # 09:30 IST
    (utc(2026, 10, 12, 5, 59), False),
    (utc(2026, 10, 12, 6, 0), True),     # 11:30 IST
    (utc(2026, 10, 12, 9, 59), True),
    (utc(2026, 10, 12, 10, 0), False),   # 15:30 IST
    (utc(2026, 10, 10, 2, 0), False),    # Saturday
    (utc(2026, 10, 11, 7, 0), False),    # Sunday
])
def test_peak_windows(at, peak):
    assert T.deepseek_is_peak(at) is peak


def test_a_configured_holiday_is_off_peak_by_its_beijing_date(no_config):
    no_config["llm.deepseek.offpeak_dates"] = "2026-10-12"
    assert T.deepseek_is_peak(utc(2026, 10, 12, 2, 0)) is False
    # 2026-10-11 20:00 UTC is already the 12th in Beijing — and a Monday
    # window cannot open on a Sunday UTC anyway, so the 13th is the check:
    assert T.deepseek_is_peak(utc(2026, 10, 13, 2, 0)) is True


def test_the_windows_can_be_moved_by_config(no_config):
    no_config["llm.deepseek.peak_utc"] = "12:00-13:00"
    assert T.deepseek_is_peak(utc(2026, 10, 12, 2, 0)) is False
    assert T.deepseek_is_peak(utc(2026, 10, 12, 12, 30)) is True


def test_peak_is_twice_off_peak():
    off = T.deepseek_direct_price("deepseek-flash", utc(2026, 10, 12, 5, 0))
    on = T.deepseek_direct_price("deepseek-flash", utc(2026, 10, 12, 7, 0))
    assert off == (0.15, 0.60, 0.003)
    assert on == pytest.approx((0.30, 1.20, 0.006))


def test_the_legacy_flash_name_bills_as_flash():
    at = utc(2026, 10, 12, 5, 0)
    assert T.deepseek_direct_price("deepseek-v4-flash", at) == T.deepseek_direct_price("deepseek-flash", at)


def test_direct_only_when_deepseek_bills_it():
    t = T.Telemetry()
    at = utc(2026, 10, 12, 5, 0)
    assert t._resolve_price("deepseek-flash", provider="deepseek", at=at) == (0.15, 0.60, 0.003)
    assert t._resolve_price("deepseek-flash", url="https://api.deepseek.com/chat/completions", at=at) == (0.15, 0.60, 0.003)
    # The same weights through a router keep the router's table.
    assert t._resolve_price("deepseek/deepseek-v4-flash", provider="openrouter", at=at) == T._price_for("deepseek/deepseek-v4-flash")


def test_deepseek_cache_hits_are_counted_as_cache_reads():
    assert T.cached_prompt_tokens({"prompt_tokens": 100, "prompt_cache_hit_tokens": 80}) == 80
    assert T.cached_prompt_tokens({"prompt_tokens_details": {"cached_tokens": 5}}) == 5
    assert T.cached_prompt_tokens({}) == 0


# ── capabilities: the answer evaluator's script reader sends photographs ──

from core.model_capabilities import capabilities_for


def test_deepseek_flash_reads_images_so_a_script_is_not_dropped():
    assert capabilities_for("deepseek-flash", "deepseek")["input"]["image"] is True
    assert capabilities_for("deepseek-v4-flash", "deepseek")["input"]["image"] is True


def test_the_router_text_model_of_the_same_name_still_does_not():
    assert capabilities_for("deepseek/deepseek-v4-flash", "openrouter")["input"]["image"] is False
    assert capabilities_for("deepseek-v4-pro", "deepseek")["input"]["image"] is False

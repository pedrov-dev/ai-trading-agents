from datetime import UTC, datetime

import pytest

from monitoring.drawdown import EquityPoint, build_drawdown_snapshot

_DEF_TIME = datetime(2026, 4, 3, 12, 0, tzinfo=UTC)


def test_build_drawdown_snapshot_tracks_high_water_mark_and_recovery() -> None:
    history = (
        EquityPoint(recorded_at=_DEF_TIME, equity=10_000.0),
        EquityPoint(recorded_at=_DEF_TIME.replace(hour=13), equity=11_000.0),
        EquityPoint(recorded_at=_DEF_TIME.replace(hour=14), equity=9_500.0),
        EquityPoint(recorded_at=_DEF_TIME.replace(hour=15), equity=9_800.0),
        EquityPoint(recorded_at=_DEF_TIME.replace(hour=16), equity=12_000.0),
    )

    snapshot = build_drawdown_snapshot(history)

    assert snapshot.high_water_mark_equity == 12_000.0
    assert snapshot.low_water_mark_equity == 9_500.0
    assert snapshot.current_drawdown_fraction == pytest.approx(0.0)
    assert snapshot.max_drawdown_fraction == pytest.approx(1_500.0 / 11_000.0)
    assert snapshot.observation_count == 5
    assert snapshot.is_recovered is True

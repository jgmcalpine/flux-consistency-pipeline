import pytest

from fluxpipe.config import estimate_cost
from fluxpipe.ledger import Ledger, SpendCapExceeded


def _reserve(ledger, cost, **kw):
    return ledger.reserve(stage="keyframe", model="flux-2-pro", prompt="p", est_cost_usd=cost, **kw)


def test_cap_blocks_the_call_that_would_cross_it(tmp_path):
    ledger = Ledger(tmp_path / "l.sqlite3", spend_cap_usd=1.00)
    _reserve(ledger, 0.60)
    with pytest.raises(SpendCapExceeded):
        _reserve(ledger, 0.50)
    assert ledger.total_spent() == 0.60


def test_failed_calls_still_count(tmp_path):
    ledger = Ledger(tmp_path / "l.sqlite3", spend_cap_usd=1.00)
    row = _reserve(ledger, 0.40)
    ledger.fail(row, "timeout")
    assert ledger.total_spent() == 0.40


def test_dry_runs_are_logged_but_free(tmp_path):
    ledger = Ledger(tmp_path / "l.sqlite3", spend_cap_usd=0.10)
    _reserve(ledger, 5.00, dry_run=True)
    assert ledger.total_spent() == 0


def test_video_cost_is_per_second():
    assert estimate_cost("flux-3-video-hd", seconds=5) == 0.85
    with pytest.raises(KeyError):
        estimate_cost("some-unpriced-model")

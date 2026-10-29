"""Быстрая проверка установки: скрипт запускается, модель работает, размеры на тестовом геле разумные.

  python tests/test_smoke.py      (или pytest tests/)
"""
import io
import contextlib
import os
import sys
import tempfile
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "claude_skill", "gel-electrophoresis-analyzer", "scripts"))
import analyze_gel as ag  # noqa: E402


def _args(image, out, **kw):
    base = dict(image=image, model=ag.DEFAULT_MODEL, out=out, ladder=None, ladder_sizes=None, ladder_lanes=None,
                negative=None, expected=None, bright_weight=0.08, ladder_ng=None, sample_ul=None, tolerance=0.10,
                invert="auto", norm="none", lane_centers=None, add_empty_left=0, add_empty_right=0,
                ladder_first_bp=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_pcr_gel():
    image = os.path.join(ROOT, "test_gels", "UVP02164June252019.jpg")
    with tempfile.TemporaryDirectory() as out:
        with contextlib.redirect_stdout(io.StringIO()):
            r = ag.analyze(_args(image, out, ladder="neb_1kb", ladder_lanes="2", add_empty_left=1, add_empty_right=1,
                                 expected="6=5000/1000,7=5000/1000"))
        assert r["lanes_detected"] == 8, r["lanes_detected"]
        cal = r["calibration"]["2"]
        assert cal["loo_median_err_pct"] < 5, cal["loo_median_err_pct"]
        assert 3001.0 in cal["locally_brightest_assigned_bp"], cal["locally_brightest_assigned_bp"]
        for lane in (6, 7):
            flags = r["lanes"][lane - 1]["flags"]
            assert "ожидаемый продукт найден" in flags, (lane, flags)
        for name in ("_overlay.png", "_report.md", "_results.json", "_calibration.png"):
            assert os.path.exists(os.path.join(out, "UVP02164June252019" + name)), name


if __name__ == "__main__":
    test_pcr_gel()
    print("OK: скрипт и модель работают")

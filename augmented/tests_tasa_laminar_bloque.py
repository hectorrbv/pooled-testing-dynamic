"""Tests of the single-block laminar rate c*(G, q)."""
import csv
from fractions import Fraction as F
from pathlib import Path

from augmented.tasa_laminar_bloque import rate, rate_float

ROOT = Path(__file__).resolve().parents[1]


def test_small_values_by_hand():
    assert rate(1)[0] == F(1, 2)          # individual tests
    assert rate(2)[0] == F(2, 3)          # pair, then one member if the count is 1
    assert rate(3)[0] == F(9, 13)
    assert rate(4)[0] == F(14, 19)


def test_rate_grows_and_stays_below_francisco_potential_at_half():
    values = [rate(G)[0] for G in range(1, 13)]
    assert all(a <= b for a, b in zip(values, values[1:]))
    assert all(v <= 2 for v in values)


def test_float_matches_exact():
    for G in (3, 6, 10):
        assert abs(rate_float(G)[0] - float(rate(G)[0])) < 1e-10


def test_grid_never_exceeds_rate_times_budget():
    path = ROOT / 'results/precio_q05_G/malla_2026-10-14_completa.csv'
    rates = {}
    for row in csv.DictReader(path.open()):
        G, B = int(row['G']), int(row['B'])
        rates.setdefault(G, rate(G)[0])
        assert F(row['exact_laminar']) <= rates[G] * B, row


def test_q_above_half_beats_two_per_test():
    # Francisco's constant 2 is calibrated to q = 1/2; at q = 0.9 the laminar rate exceeds it.
    assert rate(8, F(9, 10))[0] > 2

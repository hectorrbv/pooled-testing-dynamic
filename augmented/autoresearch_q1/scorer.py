"""Depth-D lookahead, top-4 then top-2 inner, memoised across root candidates."""
from q1_tools import expected_gain, branches, State

D = 4
_memo = {}


def _val(state, depth):
    if state.tests_left <= 0:
        return 0
    key = (state.support, state.cleared, frozenset(state.past), state.tests_left, depth, state.inst)
    if key in _memo:
        return _memo[key]
    best = 0
    gs = sorted(((expected_gain(state, p), p) for p in state.candidates), reverse=True)
    k = 1 if depth <= 1 else (4 if depth == D - 1 else 1)
    sel = gs[:k]
    if depth > 1:
        seen = {bin(p).count('1') for _, p in sel}
        for g, p in gs:
            z = bin(p).count('1')
            if z not in seen:
                seen.add(z)
                sel.append((g, p))
    for g, p in (sel if depth > 1 else sel[:1]):
        v = g
        if depth > 1 and state.tests_left > 1:
            for pr, child, new in branches(state, p):
                st = State(state.inst, state.tests_left - 1, child, state.cleared | new, state.past + (p,))
                v += pr * _val(st, depth - 1)
        if v > best:
            best = v
    _memo[key] = best
    return best


def choose(state):
    best = None
    for p in state.candidates:
        v = expected_gain(state, p)
        if state.tests_left > 1:
            for pr, child, new in branches(state, p):
                st = State(state.inst, state.tests_left - 1, child, state.cleared | new, state.past + (p,))
                v += pr * _val(st, D - 1)
        if best is None or v > best[0]:
            best = (v, p)
    return None if best is None else best[1]

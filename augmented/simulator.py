"""
Simulator: run a DAPTS on a fixed infection profile Z.

Given F and z_mask, returns the terminal history h_B(F,Z),
the cleared individuals, and the realized utility u(F,Z).
"""

from augmented.core import indices_from_mask, test_result


def apply_dapts(F, z_mask, n, u):
    """Simulate DAPTS F on infection profile z_mask.

    Returns (terminal_history, cleared_mask, u_realized).
    - cleared_mask: individuals in at least one pool with result 0.
    - u_realized: sum of u[i] for cleared individuals.
    """
    history = ()
    cleared_mask = 0

    for k in range(1, F.B + 1):
        pool = F.choose(k, history)
        r = test_result(pool, z_mask)
        history = history + ((pool, r),)
        if r == 0:
            cleared_mask |= pool

    u_val = sum(u[i] for i in indices_from_mask(cleared_mask, n))
    return history, cleared_mask, u_val

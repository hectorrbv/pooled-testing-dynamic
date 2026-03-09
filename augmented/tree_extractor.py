"""
Decision tree extraction and visualization for DAPTS strategies.

Given a solved DAPTS policy, extracts the full decision tree showing:
  - Which pool is chosen at each step
  - All possible outcomes (test results)
  - Cleared individuals at each node
  - Posterior beliefs after each observation

Supports text output and DOT format for Graphviz visualization.
"""

from augmented.core import mask_str, indices_from_mask, test_result, popcount
from augmented.bayesian import bayesian_update_single_test


def extract_tree(policy, p, u, n):
    """Extract the full decision tree from a DAPTS policy.

    Parameters
    ----------
    policy : DAPTS
        A solved DAPTS strategy.
    p : list[float]
        Prior infection probabilities.
    u : list[float]
        Individual utilities.
    n : int
        Population size.

    Returns
    -------
    dict
        Nested tree structure. Each node has:
          - 'step': test number (1-indexed)
          - 'pool': pool mask chosen
          - 'pool_str': human-readable pool
          - 'history': history leading to this node
          - 'cleared': cleared mask at this point
          - 'posteriors': current p values
          - 'children': dict mapping outcome r -> child node
          - 'utility': expected utility at terminal nodes
    """
    def _build(k, history, cleared_mask, current_p):
        if k > policy.B:
            # Terminal node
            utility = sum(u[i] for i in indices_from_mask(cleared_mask, n))
            return {
                'step': k,
                'terminal': True,
                'history': history,
                'cleared': cleared_mask,
                'cleared_str': mask_str(cleared_mask, n),
                'posteriors': list(current_p),
                'utility': utility,
            }

        # Get the pool chosen at this step for this history
        try:
            pool = policy.choose(k, history)
        except KeyError:
            # History not reachable in the policy
            utility = sum(u[i] for i in indices_from_mask(cleared_mask, n))
            return {
                'step': k,
                'terminal': True,
                'history': history,
                'cleared': cleared_mask,
                'cleared_str': mask_str(cleared_mask, n),
                'posteriors': list(current_p),
                'utility': utility,
            }

        pool_size = popcount(pool)
        children = {}

        if pool == 0:
            # Empty pool (wasted test) — only one outcome
            children[0] = _build(k + 1, history + ((0, 0),),
                                 cleared_mask, current_p)
        else:
            # Branch on all possible outcomes r = 0, 1, ..., |pool|
            for r in range(pool_size + 1):
                new_cleared = cleared_mask | pool if r == 0 else cleared_mask
                new_p = bayesian_update_single_test(current_p, pool, r, n)
                new_history = history + ((pool, r),)
                children[r] = _build(k + 1, new_history, new_cleared, new_p)

        return {
            'step': k,
            'terminal': False,
            'pool': pool,
            'pool_str': mask_str(pool, n),
            'history': history,
            'cleared': cleared_mask,
            'cleared_str': mask_str(cleared_mask, n),
            'posteriors': list(current_p),
            'children': children,
        }

    return _build(1, (), 0, list(p))


def print_tree(tree, n, indent=0, file=None):
    """Print a human-readable text representation of the decision tree.

    Parameters
    ----------
    tree : dict
        Tree from extract_tree().
    n : int
        Population size.
    indent : int
        Current indentation level.
    file : file-like or None
        Output destination (default: stdout).
    """
    prefix = "  " * indent
    import sys
    out = file or sys.stdout

    if tree.get('terminal'):
        cleared = tree['cleared_str']
        utility = tree['utility']
        post = [f"{pi:.3f}" for pi in tree['posteriors']]
        out.write(f"{prefix}[TERMINAL] cleared={cleared} utility={utility:.2f}"
                  f" posteriors=[{', '.join(post)}]\n")
        return

    step = tree['step']
    pool = tree['pool_str']
    cleared = tree['cleared_str']
    post = [f"{pi:.3f}" for pi in tree['posteriors']]

    out.write(f"{prefix}Step {step}: test pool {pool}"
              f" | cleared={cleared}"
              f" | p=[{', '.join(post)}]\n")

    for r in sorted(tree['children'].keys()):
        child = tree['children'][r]
        out.write(f"{prefix}  r={r}:\n")
        print_tree(child, n, indent + 2, file=out)


def tree_to_string(tree, n):
    """Return the tree as a string instead of printing."""
    import io
    buf = io.StringIO()
    print_tree(tree, n, file=buf)
    return buf.getvalue()


def export_tree_dot(tree, n, title="DAPTS Decision Tree"):
    """Export the decision tree to DOT format for Graphviz.

    Parameters
    ----------
    tree : dict
        Tree from extract_tree().
    n : int
        Population size.
    title : str
        Graph title.

    Returns
    -------
    str
        DOT-format string. Save to .dot file and render with:
        dot -Tpng tree.dot -o tree.png
    """
    lines = ['digraph DAPTS {']
    lines.append(f'  label="{title}";')
    lines.append('  labelloc="t";')
    lines.append('  node [shape=box, fontname="monospace", fontsize=10];')
    lines.append('  edge [fontname="monospace", fontsize=9];')
    lines.append('')

    node_id = [0]  # mutable counter

    def _add_node(tree_node, parent_id=None, edge_label=None):
        nid = node_id[0]
        node_id[0] += 1

        if tree_node.get('terminal'):
            cleared = tree_node['cleared_str']
            utility = tree_node['utility']
            label = f"DONE\\ncleared={cleared}\\nutility={utility:.2f}"
            lines.append(f'  n{nid} [label="{label}", '
                         f'style=filled, fillcolor="#d4edda"];')
        else:
            step = tree_node['step']
            pool = tree_node['pool_str']
            cleared = tree_node['cleared_str']
            label = f"Step {step}\\ntest {pool}\\ncleared={cleared}"
            lines.append(f'  n{nid} [label="{label}", '
                         f'style=filled, fillcolor="#cce5ff"];')

        if parent_id is not None and edge_label is not None:
            lines.append(f'  n{parent_id} -> n{nid} '
                         f'[label=" r={edge_label} "];')

        if not tree_node.get('terminal') and 'children' in tree_node:
            for r in sorted(tree_node['children'].keys()):
                _add_node(tree_node['children'][r], nid, r)

    _add_node(tree)
    lines.append('}')
    return '\n'.join(lines)

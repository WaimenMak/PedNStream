import numpy as np
from scipy.optimize import linprog


def solve(node, s: np.ndarray, r: np.ndarray, type: str = "classic") -> np.ndarray:
    """
    Compute the flow vector q for the given node.

    Args:
        node: The Node instance (must have .node_type and relevant attributes).
        s: Sending flow array of shape (source_num,).
        r: Receiving flow array of shape (dest_num,).
        type: Solver variant — "classic" or "optimal" (only used for regular nodes).

    Returns:
        q: Flow vector of shape (source_num + dest_num,).
    """
    if node.node_type == "onetoone":
        return _solve_onetoone(node, s, r)
    elif node.node_type == "regular":
        return _solve_regular(node, s, r, type=type)
    else:
        raise ValueError(f"Unknown node type: {node.node_type}")

# ------------------------------------------------------------------
# One-to-one node solver
# ------------------------------------------------------------------
def _solve_onetoone(node, s: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    flows = q = [S0, S1, R0, R1], S1 and R1 are virtual links
    """
    flows = np.array(
        [
            np.min([s[0], r[1]]),
            np.min([s[1], r[0]]),
            np.min([s[1], r[0]]),
            np.min([s[0], r[1]]),
        ]
    )
    if np.any(flows < 0):
        raise Warning(f"Negative flows detected: {flows}")
        # the methods below return data with the same shape as q.
    return flows

# ------------------------------------------------------------------
# Regular node solver
# ------------------------------------------------------------------

def _solve_regular(node, s: np.ndarray, r: np.ndarray, type: str = "classic") -> np.ndarray:
    if type == "optimal":
        return _solve_regular_optimal(node, s, r)
    elif type == "classic":
        return _solve_regular_classic(node, s, r)
    else:
        raise ValueError(f"Invalid type: {type}")

def _solve_regular_optimal(node, s: np.ndarray, r: np.ndarray) -> np.ndarray:
    A_eq =_build_matrix_A_eq(
        node.turning_fractions, node.edge_num, node.dest_num
    )
    # solve the linear programming problem
    w = node.w * np.ones(2 * node.edge_num)  # variables for the penalty term
    c = -1 * np.ones(node.edge_num)  # variables for the flow
    c = np.concatenate((c, w))
    assert node.edge_num > 0
    b_ub = np.concatenate((s, r))

    if A_eq is not None:
        res = linprog(
            c,
            A_ub=node.A_ub,
            A_eq=A_eq,
            b_ub=b_ub,
            b_eq=np.zeros(node.edge_num),
            method="highs",
        )
    else:
        res = linprog(c, A_ub=node.A_ub, b_ub=b_ub, method="highs")
    if res.success:
        flows = node.A_ub @ np.floor(res.x)
        # Ensure non-negative flows and round down to nearest integer
        return np.maximum(0, flows)
    return None

def _build_matrix_A_eq(turning_fractions: np.ndarray, edge_num: int, dest_num: int) -> np.ndarray:
    """
    Build the turning fractions matrix A_eq.

    Args:
        turning_fractions: Array of turning fraction values
        edge_num: Number of edges
        dest_num: Number of destination links
    """
    if turning_fractions is None:
        return None

    assert len(turning_fractions) == edge_num

    # Initialize A_eq matrix with zeros
    A_eq = np.zeros((edge_num, edge_num + 2 * edge_num))

    for i in range(edge_num):
        source_idx = i // (dest_num - 1)
        start_ind = source_idx * (dest_num - 1)

        # Set turning fractions for all destinations from this source
        A_eq[i, start_ind : start_ind + dest_num - 1] = turning_fractions[i]
        # the penalty term for the edge from the same source-destination pair should be 0
        A_eq[i, i] = turning_fractions[i] - 1  # the lth column is phi - 1
        A_eq[i, edge_num + i * 2 : edge_num + (i + 1) * 2] = np.array([1, -1])  # the penalty term

    return A_eq

def _solve_regular_classic(node, s: np.ndarray, r: np.ndarray) -> np.ndarray:
    # use the update method originally from LTM paper
    s_tiled = np.tile(s, (node.source_num, 1))
    p = node.turning_fractions.reshape(node.dest_num, node.source_num - 1)
    # Add zero diagonal elements to make it a square matrix (m x m)
    p_square = np.zeros((node.dest_num, node.source_num))
    # Fill non-diagonal elements with the reshaped turning fractions
    p_square[node.mask] = p.flatten()
    p = p_square

    weighted_s_frac = p * s_tiled.T  # 2
    row_sums = np.sum(weighted_s_frac, axis=0, keepdims=True)  # 2
    weighted_s = weighted_s_frac / np.where(row_sums != 0, row_sums, 1e-5)  # 2
    weighted_sr = r * weighted_s  # 2

    flows_list = []
    for i in range(node.edge_num):
        source_idx = i // (node.dest_num - 1)
        destination_idx = i % (node.dest_num - 1)
        g_ij = min(
            node.turning_fractions[i] * s[source_idx],
            weighted_sr[source_idx][node.mask[source_idx, :]][destination_idx],
        )  # do not use min # 2
        flows_list.append(g_ij)
    flows = node.A_ub[:, : node.edge_num] @ np.floor(np.array(flows_list))
    return np.maximum(0, flows)

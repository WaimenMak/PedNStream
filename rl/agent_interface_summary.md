# Agent Interface Summary: Dual-Gate Control System

Following the latest updates, the reinforcement learning multi-agent framework has been extended to a **Dual-Gate Control System**, giving gater agents granular control over traffic flow at specific junctions. Below is a detailed summary of the agent's observation space, action space, and environmental impact.

---

## 1. Observation Layout (`rl/builders.py`)

The observation for a gater agent is a **flattened array concatenated over its outgoing links**. The observation provides information on the state of the traffic and current gate states, padding up to the maximum out-degree in the network so matrix sizes remain constant.

Depending on the chosen `obs_mode` (e.g., `option2`), each block of link features includes items such as inflow, reverse flow, density, and importantly, finishes with current gate widths.

For a specific link $i$ (outgoing link $u \to v$, where $u$ is the gater node), the tail end of the feature block strictly follows this order:
```python
[
    ..., # (other link features like density, inflow)
    back_gate_width,            # width of the back gate on link u -> v
    rev_front_gate_width        # width of the front gate on the reverse link v -> u
]
```

---

## 2. Action Space Layout (`rl/spaces.py` & `rl/rl_utils.py`)

A single gater agent outputs continuous variables describing the target delta adjustments or absolute actions to be applied to gates located at the agent's physical junction.

The action dimension (`act_dim`) is `num_links * 2`. The components are arranged sequentially, interleaving the back-gate of the outgoing link and the front-gate of the corresponding incoming (reverse) link:
```python
[
    back_gate_0,          # Action for outgoing link 0 (u -> v_0)
    rev_front_gate_0,     # Action for incoming (reverse) link 0 (v_0 -> u)
    back_gate_1,          # Action for outgoing link 1 (u -> v_1)
    rev_front_gate_1,     # Action for incoming (reverse) link 1 (v_1 -> u)
    ...
]
```

### Constraints and Bounds
The policy output actions are clipped to their physical bounds based on the widths of the respective physical roads:
* `back_gate_i` bounds: `[0, link_i.width]`
* `rev_front_gate_i` bounds: `[0, link_i.reverse_link.width]`

---

## 3. Environmental Impact & Physical Meaning

The new paired-action mechanism reflects the physical reality of controlling a junction. An active gater agent positioned at node $u$ controls the *entrance to* an outgoing link and the *exit from* an incoming link.

### 🔵 Back Gate of the Outgoing Link (`link_i.back_gate_width`)
* **Location:** At node $u$ (source end of outgoing link $u \to v$).
* **Role:** Controls the **receiving flow**, or the amount of traffic allowed to *spill into* link $u \to v$ from junction $u$.
* **Impact of restricting (closing):** Traffic backs up into junction $u$ and potentially backward into preceding upstream links. This creates a bottleneck for flow trying to enter $u \to v$.

### 🟢 Front Gate of the Reverse Link (`link_i.reverse_link.front_gate_width`)
* **Location:** Also at node $u$ (destination end of incoming link $v \to u$).
* **Role:** Controls the **sending flow**, or the amount of traffic allowed to *exit* link $v \to u$ into junction $u$.
* **Impact of restricting (closing):** Traffic backs up entirely *within* the incoming link $v \to u$ before it can reach junction $u$. This protects the junction from becoming overcrowded.

### Conclusion of the Mechanism
By interleaving `[back_gate, rev_front_gate]` at the same junction block, the agent now receives and acts upon the symmetric, localized context. At any given intersection $u$:
1. It reads the capacity of the current incoming link vs the outgoing road.
2. It can **hold traffic back** in the incoming link by shrinking the front gate of the reverse link.
3. It can **meter traffic down** into the outgoing link by shrinking the back gate of the outgoing link.

*Note: Visual markers on the graph explicitly label these points—the **blue triangle pointing away from the junction** is the back gate of the outgoing link, while the **green triangle pointing into the junction** is the front gate of the incoming link.*

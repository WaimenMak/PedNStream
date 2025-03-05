# Implementation of the LTM pedestrian simulator

## OneToOneNode
[-] The node between two nodes.

## RegularNode

```
project_root/
├── src/
│   ├── models/           # Core model components
│   │   ├── __init__.py
│   │   ├── link.py
│   │   ├── node.py
│   │   └── network.py
│   ├── visualization/    # Visualization related code
│   │   ├── __init__.py
│   │   └── plot.py
│   └── utils/           # True utilities (if needed later)
│       ├── __init__.py
│       └── functions.py  # Generic helper functions
├── examples/
│   └── simple_network.py
└── LTM.py
```

## Vsiualization of the simulation

<img src="./README.assets/network_animation.gif" alt="network_animation" style="zoom: 67%;" />

## Path Finder
[*] `node_to_od_pairs`: store the od pairs of each node.  
[*] `find_all_paths`: find all paths from origin to destination.  
[*] `calculate_all_turn_probs`: calculate the turn probabilities of each node in paths.  
    - `turns_od_dict`: store the ods in each turn.  
    - `od_turn_distances`: store the remaining distance to the destination of this turn. (current node to destination)   
    - `turns_by_upstream`: the turns in the upstream node.  
    -`od_turn_distances`: current turns in the od with distance as the value

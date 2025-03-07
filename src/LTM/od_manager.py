import numpy as np

class ODManager:
    """Manages Origin-Destination flows and operations"""
    def __init__(self, simulation_steps: int):
        self.od_flows = {}  # {(o,d): array of flows}
        self.simulation_steps = simulation_steps

    def init_od_flows(self, origin_nodes: list, destination_nodes: list, od_flows: dict = None):
        """
        Initialize OD flows with user-defined values.
        
        Args:
            origin_nodes: List of origin node IDs
            destination_nodes: List of destination node IDs
            od_flows: Dictionary {(o,d): array or float} of flows
                     If float provided, will be expanded to constant array
        """
        if od_flows:
            self._set_predefined_flows(od_flows)
        else:
            # Initialize with ones, the ratio are all equal
            print(f"No OD flows provided, initializing with ones")
            for o in origin_nodes:
                for d in destination_nodes:
                    if o != d:
                        self.od_flows[(o, d)] = np.ones(self.simulation_steps)

    def _set_predefined_flows(self, od_flows: dict):
        """Set predefined OD flows"""
        for (o, d), flow in od_flows.items():
            if isinstance(flow, (int, float)):
                self.od_flows[(o, d)] = np.ones(self.simulation_steps) * flow
            else:
                if len(flow) != self.simulation_steps:
                    raise ValueError(f"Flow array length for OD pair ({o},{d}) must match simulation_steps")
                self.od_flows[(o, d)] = np.array(flow)

    def get_od_flow(self, origin: int, destination: int, time_step: int) -> float:
        """Get flow for specific OD pair at time step"""
        return self.od_flows.get((origin, destination),
                               np.zeros(self.simulation_steps))[time_step]

    # def get_total_origin_flow(self, origin: int, time_step: int) -> float:
    #     """Get total outflow from an origin at time step"""
    #     return sum(self.get_od_flow(origin, dest, time_step)
    #               for dest in set(d for o, d in self.od_flows.keys() if o == origin))
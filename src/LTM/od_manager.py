import numpy as np

class ODManager:
    """Manages Origin-Destination related data and operations"""
    def __init__(self, simulation_steps: int):
        self.od_ratios = {}  # {(o,d): array of ratios}
        self.simulation_steps = simulation_steps

    def init_od_ratios(self, origin_nodes: list, destination_nodes: list, od_ratios: dict = None):
        """Initialize OD ratios with user-defined values or uniform distribution"""
        if od_ratios is None:
            # Use uniform distribution
            for o in origin_nodes:
                valid_dests = [d for d in destination_nodes if d != o]
                default_ratio = 1.0 / len(valid_dests)
                for d in valid_dests:
                    self.od_ratios[(o, d)] = np.ones(self.simulation_steps) * default_ratio
        else:
            self._set_predefined_ratios(od_ratios, origin_nodes)
            self.verify_od_ratios()

    def _set_predefined_ratios(self, od_ratios: dict, origin_nodes: list):
        """Set predefined OD ratios"""
        for o in origin_nodes:
            o_dests = [d for (o_, d), ratio in od_ratios.items() if o_ == o]
            for d in o_dests:
                ratio = od_ratios.get((o, d))
                if ratio is not None:
                    self.od_ratios[(o, d)] = (np.ones(self.simulation_steps) * ratio 
                        if isinstance(ratio, (int, float)) else np.array(ratio))

    def verify_od_ratios(self):
        """Verify ratios sum to 1 for each origin"""
        origins = set(o for o, _ in self.od_ratios.keys())
        for o in origins:
            ratio_sum = sum(self.od_ratios[(o, d)] 
                          for d in set(d for o_, d in self.od_ratios.keys() if o_ == o))
            if not np.allclose(ratio_sum, 1.0, rtol=1e-5):
                raise ValueError(f"OD ratios for origin {o} do not sum to 1")

    def get_od_ratio(self, origin: int, destination: int, time_step: int) -> float:
        """Get ratio for specific OD pair at time step"""
        return self.od_ratios.get((origin, destination), 
                                np.zeros(self.simulation_steps))[time_step] 
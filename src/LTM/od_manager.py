from typing import Dict, Callable
from dataclasses import dataclass
import numpy as np
import logging

@dataclass
class DemandConfig:
    """Configuration for demand generation with defaults"""
    peak_lambda: float = 10.0
    base_lambda: float = 5.0
    seed: int = 42
    pattern: str = 'gaussian_peaks'

class ODManager:
    """Manages Origin-Destination flows and operations"""
    def __init__(self, simulation_steps: int, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.od_flows = {}  # {(o,d): array of flows}
        self.simulation_steps = simulation_steps
        self._default_zero_flow = np.zeros(self.simulation_steps)

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
            self.logger.info("No OD flows provided, initializing with ones")
            for o in origin_nodes:
                for d in destination_nodes:
                    if o != d:
                        self.od_flows[(o, d)] = np.ones(self.simulation_steps)

    def _set_predefined_flows(self, od_flows: dict):
        """Set predefined OD flows"""
        for (o, d), flow in od_flows.items():
            if isinstance(flow, (int, float)):  # if the element in od_flows is a single value
                self.od_flows[(o, d)] = np.full(self.simulation_steps, flow)
            else:
                if len(flow) != self.simulation_steps: # if the element in od_flows is a list or array
                    raise ValueError(f"Flow array length for OD pair ({o},{d}) must match simulation_steps")
                self.od_flows[(o, d)] = np.array(flow)

    def get_od_flow(self, origin: int, destination: int, time_step: int) -> float:
        """Get flow for specific OD pair at time step"""
        return self.od_flows.get((origin, destination), self._default_zero_flow)[time_step]


class DemandGenerator:
    """
    Generate demand patterns for origin nodes.
    """
    def __init__(self, simulation_steps: int, params: dict, logger: logging.Logger):
        self.logger = logger
        self.simulation_steps = simulation_steps
        self.params = params
        self.time = np.arange(simulation_steps)

        # Dictionary to store built-in and custom demand patterns
        self.demand_patterns: Dict[str, Callable] = {
            'gaussian_peaks': self.generate_gaussian_peaks,
            'constant': self.generate_constant,
            'sudden_demand': self.generate_sudden_demand,
        }

    def register_pattern(self, pattern_name: str, pattern_func: Callable):
        """
        Register a new demand pattern function.

        Args:
            pattern_name: Name of the pattern to register
            pattern_func: Function that takes (origin_id, params) and returns np.array
        """
        if not callable(pattern_func):
            raise ValueError("pattern_func must be callable")
        self.demand_patterns[pattern_name] = pattern_func

    def _get_demand_config(self, origin_id: int) -> DemandConfig:
        """Get demand configuration for origin with fallback to defaults"""
        try:
            origin_config = self.params['demand'][f'origin_{origin_id}']
            return DemandConfig(
                peak_lambda=origin_config.get('peak_lambda', 10.0),
                base_lambda=origin_config.get('base_lambda', 5.0),
                seed=origin_config.get('seed', 42),
                pattern=origin_config.get('pattern', 'gaussian_peaks')
            )
        except KeyError:
            self.logger.info(f"No demand configuration found for origin {origin_id}, using defaults")
            return DemandConfig()

    def generate_gaussian_peaks(self, origin_id: int, params=None) -> np.ndarray:
        """Built-in gaussian peaks demand pattern"""
        config = self._get_demand_config(origin_id)
        return self._generate_base_gaussian_pattern(config)

    def generate_constant(self, origin_id: int, params=None) -> np.ndarray:
        """Built-in constant demand pattern"""
        config = self._get_demand_config(origin_id)
        return np.full(self.simulation_steps, config.base_lambda)
    
    def generate_sudden_demand(self, origin_id: int, params=None) -> np.ndarray:
        """Built-in sudden demand pattern with configurable parameters"""
        config = self._get_demand_config(origin_id)
        
        # Generate base pattern using shared method
        demand = self._generate_base_gaussian_pattern(config)
        
        # Add sudden demand spike
        sudden_period = np.random.randint(10, 20)
        start_step = np.random.randint(0, max(1, self.simulation_steps - sudden_period))
        demand[start_step:start_step + sudden_period] += np.random.randint(20, 50)
        
        return demand

    def generate_custom(self, origin_id: int, pattern: str) -> np.ndarray:
        """
        Generate demand based on specified pattern name. To use custom patterns, register them first.

        Args:
            origin_id: ID of the origin node
            pattern: Name of the pattern to use

        Returns:
            np.ndarray: Generated demand array

        Raises:
            ValueError: If pattern name is not registered
        """
        if pattern not in self.demand_patterns:
            raise ValueError(f"Unknown demand pattern: {pattern}. "
                           f"Available patterns: {list(self.demand_patterns.keys())}")
        # the argument params is optional for some patterns, it is the params setting in the yaml file
        return self.demand_patterns[pattern](origin_id, params=self.params)

    def _generate_base_gaussian_pattern(self, config: DemandConfig) -> np.ndarray:
        """Generate base gaussian pattern - shared by multiple methods"""
        t = self.simulation_steps
        morning_peak = config.peak_lambda * np.exp(-(self.time - t/4)**2 / (2 * (t/20)**2))
        evening_peak = config.peak_lambda * np.exp(-(self.time - 3*t/4)**2 / (2 * (t/20)**2))
        lambda_t = config.base_lambda + morning_peak + evening_peak
        
        np.random.seed(config.seed)
        return np.random.poisson(lam=lambda_t)

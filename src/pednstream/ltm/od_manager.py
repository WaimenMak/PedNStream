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
    peak_position: float = 0.5  # Position of peak in simulation (0.0-1.0)

class ODManager:
    """Manages Origin-Destination flows and operations"""

    def __init__(self, simulation_steps: int, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.od_flows = {}  # {(o,d): array of flows}
        self.simulation_steps = simulation_steps
        self._default_zero_flow = np.zeros(self.simulation_steps + 1)

    def init_od_flows(
        self, origin_nodes: list, destination_nodes: list, od_flows: dict = None
    ):
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
                        self.od_flows[(o, d)] = np.ones(self.simulation_steps + 1) # Necessary at network initialization, one per simulation

    def _set_predefined_flows(self, od_flows: dict): # Set predefined OD flows from YAML file
        """Set predefined OD flows"""
        for (o, d), flow in od_flows.items():
            if isinstance(
                flow, (int, float)
            ):  # if the element in od_flows is a single value
                self.od_flows[(o, d)] = np.full(self.simulation_steps + 1, flow)
            else:
                if (
                    len(flow) != self.simulation_steps + 1
                ):  # if the element in od_flows is a list or array
                    raise ValueError(
                        f"Flow array length for OD pair ({o},{d}) must match simulation_steps"
                    )
                self.od_flows[(o, d)] = np.array(flow)

    def get_od_flow(self, origin: int, destination: int, time_step: int) -> float:
        """Get flow for specific OD pair at time step"""
        return self.od_flows.get((origin, destination), self._default_zero_flow)[
            time_step
        ]


class DemandGenerator:  # Generate a function for the simulation at initialization,  One per simulation.
    """
    Generate demand patterns for origin nodes.
    """

    # TODO: Refactor to reduce repetion.
    def __init__(self, simulation_steps: int, params: dict, logger: logging.Logger):
        self.logger = logger
        self.simulation_steps = simulation_steps
        self.params = params
        self.time = np.arange(simulation_steps)
        self.seed = params.get(
            "seed", None
        )  # the seed from the simulation params, can be None

        # Dictionary to store built-in and custom demand patterns
        self.demand_patterns: Dict[str, Callable] = {
            'gaussian_peaks': self.generate_gaussian_peaks,
            'constant': self.generate_constant,
            'sudden_demand': self.generate_sudden_demand,
            'multi_peaks': self.generate_multi_peaks,
            'single_peak': self.generate_single_peak,
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
            origin_config = self.params["demand"][f"origin_{origin_id}"]
            return DemandConfig(
                peak_lambda=origin_config.get("peak_lambda", 10.0),
                base_lambda=origin_config.get("base_lambda", 5.0),
                seed=self.seed,  # use the seed from the simulation params
                pattern=origin_config.get('pattern', 'gaussian_peaks'),
                peak_position=origin_config.get('peak_position', 0.5)
            )
        except KeyError:
            if self.logger:
                self.logger.info(
                    f"No demand configuration found for origin {origin_id}, using defaults"
                )
            return DemandConfig()

    def generate_gaussian_peaks(self, origin_id: int, params=None) -> np.ndarray:
        """Built-in gaussian peaks demand pattern"""
        config = self._get_demand_config(origin_id)
        return self._generate_base_gaussian_pattern(config)

    def generate_constant(self, origin_id: int, params=None) -> np.ndarray:
        """Built-in constant demand pattern"""
        config = self._get_demand_config(origin_id)
        return np.full(self.simulation_steps + 1, config.base_lambda)

    def generate_sudden_demand(self, origin_id: int, params=None) -> np.ndarray:
        """Built-in sudden demand pattern with configurable parameters"""
        config = self._get_demand_config(origin_id)

        # Generate base pattern using shared method
        demand = self._generate_base_gaussian_pattern(config)

        # Add sudden demand spike
        sudden_period = np.random.randint(10, 20)
        start_step = np.random.randint(0, max(1, self.simulation_steps - sudden_period))
        demand[start_step : start_step + sudden_period] += np.random.randint(20, 50)

        return demand

    def generate_multi_peaks(self, origin_id: int, params=None) -> np.ndarray:
        """
        Generate demand pattern with multiple peaks of different values.
        Uses same config as other patterns (base_lambda, peak_lambda).

        Randomly generates 2-5 peaks at evenly spaced positions with random
        multipliers (0.5x-1.5x) applied to peak_lambda.

        Configuration in YAML (same as other patterns):
        demand:
          origin_0:
            pattern: 'multi_peaks'
            base_lambda: 5.0
            peak_lambda: 15.0
        """
        config = self._get_demand_config(origin_id)
        t = self.simulation_steps

        # Random number of peaks (1-4)
        num_peaks = np.random.randint(1, 7)

        # Evenly space peaks across simulation time (avoid edges)
        positions = np.linspace(t * 0.15, t * 0.85, num_peaks)

        # Random multipliers for each peak (0.5x to 1.5x)
        multipliers = np.random.uniform(0.5, 1.5, num_peaks)

        # Random widths for each peak (t/30 to t/10)
        widths = np.random.uniform(t / 30, t / 10, num_peaks)

        # Generate peaks
        lambda_t = np.full(self.simulation_steps, config.base_lambda, dtype=float)
        for pos, mult, width in zip(positions, multipliers, widths):
            peak = (
                config.peak_lambda
                * mult
                * np.exp(-((self.time - pos) ** 2) / (2 * width**2))
            )
            lambda_t += peak

        # Only reseed if seed is explicitly provided, otherwise use current random state
        if self.seed is not None:
            np.random.seed(self.seed)
        return np.random.poisson(lam=lambda_t)

    def generate_single_peak(self, origin_id: int, params=None) -> np.ndarray:
        """
        Generate demand pattern with a single peak at a configurable position.
        
        Uses config.peak_position (0.0-1.0) to determine where the peak occurs
        in the simulation timeline. This allows different origins to have peaks
        at different times, creating staggered demand patterns.
        
        Configuration in YAML:
        demand:
          origin_0:
            pattern: 'single_peak'
            base_lambda: 5.0
            peak_lambda: 15.0
            peak_position: 0.3  # Peak at 30% of simulation time
        """
        config = self._get_demand_config(origin_id)
        t = self.simulation_steps
        
        # Calculate center step based on peak_position (0.0-1.0)
        center_step = t * config.peak_position
        
        # Sigma controls the width of the peak (simulation_steps / 10)
        sigma = t / 10
        
        # Generate single Gaussian peak
        peak = config.peak_lambda * np.exp(-(self.time - center_step)**2 / (2 * sigma**2))
        lambda_t = config.base_lambda + peak
        
        # Only reseed if seed is explicitly provided, otherwise use current random state
        if self.seed is not None:
            np.random.seed(self.seed)
        return np.random.poisson(lam=lambda_t)

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
            raise ValueError(
                f"Unknown demand pattern: {pattern}. "
                f"Available patterns: {list(self.demand_patterns.keys())}"
            )
        # the argument params is optional for some patterns, it is the params setting in the yaml file
        return self.demand_patterns[pattern](origin_id, params=self.params)

    def _generate_base_gaussian_pattern(self, config: DemandConfig) -> np.ndarray:
        """Generate base gaussian pattern - shared by multiple methods"""
        t = self.simulation_steps
        morning_peak = config.peak_lambda * np.exp(
            -((self.time - t / 4) ** 2) / (2 * (t / 20) ** 2)
        )
        evening_peak = config.peak_lambda * np.exp(
            -((self.time - 3 * t / 4) ** 2) / (2 * (t / 20) ** 2)
        )
        lambda_t = config.base_lambda + morning_peak + evening_peak

        # Only reseed if seed is explicitly provided, otherwise use current random state
        if self.seed is not None:
            np.random.seed(self.seed)
        return np.random.poisson(lam=lambda_t)

"""Simulation run: a network plus everything that belongs to one run.

The Network holds the topology and the time-series state. A Simulation owns the parts
that only make sense for one run, built from a SimulationConfig: the time horizon, the
origin demand, the OD flows, the path finder, and the time loop.

Typical use::

    from pednstream.utils.config import load_config
    from pednstream.ltm.simulation import Simulation

    simulation = Simulation.from_configs(load_config("sim_params.yaml"))
    simulation.run()

The setup follows the reference implementation (92fa2a3 / v0.0.2-alpha) step by step,
because the order of random draws decides the results of a seeded run:
demand in node discovery order, then OD flows, then the path finder.
"""

import logging
from typing import Callable, Iterable, Optional

from pednstream.ltm.network import Network, NetworkConfig
from pednstream.ltm.od_manager import DemandGenerator, ODManager
from pednstream.ltm.path_finder import PathFinder

DEFAULT_DEMAND_PATTERN = "gaussian_peaks"


class Simulation:
    """One simulation run on a network."""

    def __init__(
        self,
        network_config: NetworkConfig,
        simulation_config,
        demand_patterns: Optional[Iterable[Callable]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Build the network and prepare the run (demand, OD flows, paths).

        Args:
            network_config (NetworkConfig): topology of the network.
            simulation_config (SimulationConfig): parameters of the run.
            demand_patterns: custom demand functions ``f(origin_id, params) -> ndarray``,
                registered under their function name and selectable with
                ``demand.origin_<id>.pattern`` in the configuration.
            logger (logging.Logger | None): logger for the demand generator and path finder.
        """
        self.config = simulation_config
        self.logger = logger
        self.time_step = 0  # last simulated time step

        self.network = Network(network_config)
        self.network.create()
        self.network.allocate(simulation_config.simulation_steps)

        self.demand_generator = DemandGenerator(
            simulation_config.simulation_steps, self.params, logger
        )
        for func in demand_patterns or []:
            self.demand_generator.register_pattern(func.__name__, func)
        self._assign_demand()

        self.od_manager: Optional[ODManager] = None
        self.path_finder: Optional[PathFinder] = None
        if self.destination_nodes:
            self._init_route_choice()

    @classmethod
    def from_configs(cls, configs: dict, **kwargs) -> "Simulation":
        """Build a Simulation from the output of ``utils.config.build_configs``/``load_config``."""
        return cls(configs["network"], configs["simulation"], **kwargs)

    # ------------------------------------------------------------------
    # Run parameters
    # ------------------------------------------------------------------

    @property
    def simulation_steps(self) -> int:
        """Number of steps of the run."""
        return self.config.simulation_steps

    @property
    def unit_time(self) -> float:
        """Duration of one time step."""
        return self.config.unit_time

    @property
    def origin_nodes(self) -> list:
        """Origin node ids."""
        return list(self.network.config.origin_nodes)

    @property
    def destination_nodes(self) -> list:
        """Destination node ids."""
        return list(self.network.config.destination_nodes)

    @property
    def params(self) -> dict:
        """Flat parameter dict, the format DemandGenerator and PathFinder expect.

        Custom demand patterns receive this dict as their ``params`` argument.
        """
        config = self.config
        return {
            "simulation_steps": config.simulation_steps,
            "unit_time": config.unit_time,
            "assign_flows_type": config.assign_flows_type,
            "seed": config.seed,
            "path_finder": dict(config.path_finder or {}),
            "demand": {
                f"origin_{origin}": dict(vars(demand))
                for origin, demand in (config.demand_params or {}).items()
            },
        }

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _assign_demand(self) -> None:
        """Generate the demand of every origin node.

        Origins are processed in node discovery order, as the reference created them,
        so seeded runs reproduce the reference results.
        """
        origins = set(self.origin_nodes)
        demand_config = self.params["demand"]
        for node_id in self.network.loading_order:
            if node_id not in origins:
                continue
            pattern = demand_config.get(f"origin_{node_id}", {}).get(
                "pattern", DEFAULT_DEMAND_PATTERN
            )
            demand = self.demand_generator.generate_custom(node_id, pattern)
            self.network.set_demand(node_id, demand)
            if self.logger:
                self.logger.info(f"Total demand of origin node {node_id}: {demand.sum()}")

    def _controllers(self) -> tuple[set, list]:
        """Controller nodes and links, as the path finder expects them.

        Controller nodes include both ends of every controller (separator) link.
        """
        links = [link_id for link_id, link in self.network.links.items() if link.is_controller]
        nodes = {node.id for node in self.network.nodes.values() if node.is_controller}
        for start, end in links:
            nodes.update((start, end))
        return nodes, links

    def _init_route_choice(self) -> None:
        """OD flows and path finder (only with destination nodes)."""
        self.od_manager = ODManager(self.simulation_steps, logger=self.logger)
        self.od_manager.init_od_flows(
            self.origin_nodes, self.destination_nodes, self.config.od_flows or None
        )

        controller_nodes, controller_links = self._controllers()
        # The path finder reads link state at step 0, so the network must be allocated.
        self.path_finder = PathFinder(
            self.network.links,
            params=self.params,
            controller_nodes=controller_nodes,
            controller_links=controller_links,
            logger=self.logger,
        )
        self.path_finder.find_od_paths(
            od_pairs=self.od_manager.od_flows.keys(), nodes=self.network.nodes
        )

    # ------------------------------------------------------------------
    # Time loop
    # ------------------------------------------------------------------

    def step(self) -> int:
        """Simulate the next time step and return it."""
        if self.time_step >= self.simulation_steps:
            raise RuntimeError(
                f"Simulation finished: all {self.simulation_steps} steps have been run."
            )
        self.time_step += 1
        self.network.network_loading(
            self.time_step,
            path_finder=self.path_finder,
            od_manager=self.od_manager,
            assign_flows_type=self.config.assign_flows_type,
        )
        return self.time_step

    def run(self, until: Optional[int] = None) -> None:
        """Simulate up to time step ``until`` (default: simulation_steps).

        Args:
            until (int | None): last time step to simulate.
        """
        until = self.simulation_steps if until is None else until
        if not self.time_step <= until <= self.simulation_steps:
            raise ValueError(
                f"until must be between {self.time_step} and {self.simulation_steps}, got {until}."
            )
        while self.time_step < until:
            self.step()
        return None

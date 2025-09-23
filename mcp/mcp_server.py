# -*- coding: utf-8 -*-
# @Time    : 19/09/2025 16:57
# @Author  : mmai
# @FileName: mcp_server
# @Software: PyCharm

"""
FastMCP Server for PedNStream Simulation
Provides tools for environment creation, simulation running, output saving, and visualization.
"""

import os
import sys
import json
import uuid
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging
from dataclasses import dataclass, asdict
from enum import Enum

# Add project root to path
project_root = Path(__file__).resolve().parent.parent  # Go up one more level
sys.path.append(str(project_root))

# Set matplotlib to headless backend for server use
import matplotlib
matplotlib.use('Agg')

from fastmcp import FastMCP
from src.utils.env_loader import NetworkEnvGenerator
from src.LTM.network import Network
from handlers.output_handler import OutputHandler
from src.utils.visualizer import NetworkVisualizer
import numpy as np

class SimulationStatus(Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class SimulationState:
    sim_id: str
    config_name: str
    overrides: Dict[str, Any]
    status: SimulationStatus
    network: Optional[Network]
    current_step: int
    total_steps: int
    output_dir: Optional[Path]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]

    def to_dict(self):
        """Convert to dict for JSON serialization"""
        return {
            'sim_id': self.sim_id,
            'config_name': self.config_name,
            'overrides': self.overrides,
            'status': self.status.value,
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'output_dir': str(self.output_dir) if self.output_dir else None,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message
        }

class SimulationManager:
    """Manages simulation state and execution"""

    def __init__(self, base_output_dir: str = "outputs"):
        self.simulations: Dict[str, SimulationState] = {}
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        self._lock = threading.Lock()

    def create_simulation(self, config_name: str, overrides: Dict[str, Any] = None) -> str:
        """Create a new simulation"""
        sim_id = str(uuid.uuid4())[:8]  # Short UUID

        with self._lock:
            self.simulations[sim_id] = SimulationState(
                sim_id=sim_id,
                config_name=config_name,
                overrides=overrides or {},
                status=SimulationStatus.CREATED,
                network=None,
                current_step=0,
                total_steps=0,
                output_dir=None,
                created_at=datetime.now(),
                started_at=None,
                completed_at=None,
                error_message=None
            )

        return sim_id

    def get_simulation(self, sim_id: str) -> Optional[SimulationState]:
        """Get simulation state"""
        with self._lock:
            return self.simulations.get(sim_id)

    def list_simulations(self) -> List[Dict[str, Any]]:
        """List all simulations"""
        with self._lock:
            return [sim.to_dict() for sim in self.simulations.values()]

    def update_status(self, sim_id: str, status: SimulationStatus,
                     current_step: int = None, error_message: str = None):
        """Update simulation status"""
        with self._lock:
            if sim_id in self.simulations:
                sim = self.simulations[sim_id]
                sim.status = status
                if current_step is not None:
                    sim.current_step = current_step
                if error_message is not None:
                    sim.error_message = error_message
                if status == SimulationStatus.RUNNING and sim.started_at is None:
                    sim.started_at = datetime.now()
                elif status in [SimulationStatus.COMPLETED, SimulationStatus.FAILED]:
                    sim.completed_at = datetime.now()

# Initialize server and simulation manager
mcp = FastMCP(
    name="PedNStream-Server",
    instructions="""
    This server provides tools for pedestrian traffic simulation using the Link Transmission Model (LTM).
    
    Available capabilities:
    - Create simulation environments from configuration files
    - Run pedestrian flow simulations  
    - Save simulation outputs and time series data
    - Generate visualizations and animations
    - Analyze simulation results
    
    Typical workflow:
    1. create_environment() to set up a network
    2. run_simulation() to execute the simulation
    3. save_outputs() to persist results
    4. visualize_snapshot() or animate() to create visuals
    """
)

sim_manager = SimulationManager()

def _create_environment_impl(config_name: str, overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Core implementation for creating a simulation environment.
    Used by tool entrypoints to avoid tool-to-tool calls.
    """
    # Create simulation entry
    sim_id = sim_manager.create_simulation(config_name, overrides)
    print(config_name)

    # Load and create network environment
    env_generator = NetworkEnvGenerator()

    # Apply overrides to config if provided
    if overrides:
        # Load base config first
        network_data = env_generator.load_network_data(config_name)

        # Normalize and apply overrides to the loaded config
        def deep_update(base_dict, override_dict):
            for key, value in override_dict.items():
                if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value

        try:
            normalized_overrides = _normalize_config(overrides)
        except Exception:
            normalized_overrides = overrides
        deep_update(env_generator.config, normalized_overrides)

    # Create network
    network = env_generator.create_network(config_name)

    # Update simulation state
    sim = sim_manager.get_simulation(sim_id)
    sim.network = network
    sim.total_steps = network.simulation_steps

    return {
        'sim_id': sim_id,
        'config_name': config_name,
        'total_steps': network.simulation_steps,
        'origin_nodes': network.origin_nodes,
        'destination_nodes': network.destination_nodes,
        'num_nodes': len(network.nodes),
        'num_links': len(network.links),
        'overrides_applied': overrides or {}
    }

@mcp.tool
def create_environment(config_name: str, overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create a simulation environment from a configuration file.

    Args:
        config_name: Name of configuration in data/ directory (e.g., 'delft', 'melbourne', 'od_flow_example')
        overrides: Optional parameter overrides (e.g., {'simulation': {'simulation_steps': 1000}})

    Returns:
        Dictionary with sim_id and environment details
    """
    try:
        return _create_environment_impl(config_name, overrides)
    except Exception as e:
        # Best-effort: retrieve sim_id from manager if present in overrides path is not trivial; skip if unavailable
        raise Exception(f"Failed to create environment: {str(e)}")

@mcp.tool
def run_simulation(sim_id: str, steps: int = None, until: int = None) -> Dict[str, Any]:
    """
    Run a simulation for the specified number of steps.

    Args:
        sim_id: Simulation ID from create_environment
        steps: Number of steps to run (optional, uses config default if not specified)
        until: Run until this time step (alternative to steps)

    Returns:
        Dictionary with execution status and progress
    """
    sim = sim_manager.get_simulation(sim_id)
    if not sim:
        raise ValueError(f"Simulation {sim_id} not found")

    if not sim.network:
        raise ValueError(f"Simulation {sim_id} has no network - create environment first")

    try:
        sim_manager.update_status(sim_id, SimulationStatus.RUNNING)

        # Determine end step
        if until is not None:
            end_step = min(until, sim.total_steps)
        elif steps is not None:
            end_step = min(sim.current_step + steps, sim.total_steps)
        else:
            end_step = sim.total_steps

        # Run simulation steps
        start_step = max(1, sim.current_step + 1)  # network_loading starts from 1

        for t in range(start_step, end_step):
            sim.network.network_loading(t)
            sim_manager.update_status(sim_id, SimulationStatus.RUNNING, current_step=t)

        # Mark as completed if we reached the end
        final_status = SimulationStatus.COMPLETED if end_step >= sim.total_steps else SimulationStatus.RUNNING
        sim_manager.update_status(sim_id, final_status, current_step=end_step)

        return {
            'sim_id': sim_id,
            'status': final_status.value,
            'steps_completed': end_step,
            'total_steps': sim.total_steps,
            'progress': end_step / sim.total_steps * 100
        }

    except Exception as e:
        sim_manager.update_status(sim_id, SimulationStatus.FAILED, error_message=str(e))
        raise Exception(f"Simulation failed: {str(e)}")

@mcp.tool
def save_outputs(sim_id: str, include_time_series: bool = True) -> Dict[str, Any]:
    """
    Save simulation outputs to disk.

    Args:
        sim_id: Simulation ID
        include_time_series: Whether to save time series CSV data

    Returns:
        Dictionary with output directory and saved files
    """
    sim = sim_manager.get_simulation(sim_id)
    if not sim:
        raise ValueError(f"Simulation {sim_id} not found")

    if not sim.network:
        raise ValueError(f"Simulation {sim_id} has no network")

    try:
        # Create output handler
        output_handler = OutputHandler(
            base_dir=str(sim_manager.base_output_dir),
            simulation_dir=f"sim_{sim_id}"
        )

        # Save network state
        output_handler.save_network_state(sim.network)

        saved_files = ['link_data.json', 'node_data.json', 'network_params.json']

        # Save time series if requested
        if include_time_series:
            output_handler.save_time_series(sim.network)
            saved_files.append('time_series.csv')

        # Update simulation state
        sim.output_dir = output_handler.simulation_dir

        return {
            'sim_id': sim_id,
            'output_dir': str(output_handler.simulation_dir),
            'files_saved': saved_files
        }

    except Exception as e:
        raise Exception(f"Failed to save outputs: {str(e)}")

@mcp.tool
def visualize_snapshot(sim_id: str, time_step: int, edge_property: str = "density",
                      figsize: tuple = (10, 8)) -> Dict[str, Any]:
    """
    Create a visualization of network state at a specific time step.

    Args:
        sim_id: Simulation ID
        time_step: Time step to visualize
        edge_property: Property to visualize ('density', 'flow', 'speed', 'num_pedestrians')
        figsize: Figure size as (width, height)

    Returns:
        Dictionary with path to saved visualization
    """
    sim = sim_manager.get_simulation(sim_id)
    if not sim:
        raise ValueError(f"Simulation {sim_id} not found")

    if not sim.output_dir:
        raise ValueError(f"No outputs saved for simulation {sim_id} - call save_outputs first")

    try:
        # Create visualizer from saved data
        visualizer = NetworkVisualizer(simulation_dir=str(sim.output_dir))

        # Generate visualization
        fig, ax = visualizer._visualize_network_nx(
            time_step=time_step,
            edge_property=edge_property,
            with_colorbar=True,
            figsize=figsize
        )

        # Save figure
        filename = f"snapshot_t{time_step}_{edge_property}.png"
        filepath = sim.output_dir / filename
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        matplotlib.pyplot.close(fig)

        return {
            'sim_id': sim_id,
            'time_step': time_step,
            'edge_property': edge_property,
            'file_path': str(filepath),
            'filename': filename
        }

    except Exception as e:
        raise Exception(f"Failed to create visualization: {str(e)}")

@mcp.tool
def animate(sim_id: str, start_time: int = 0, end_time: int = None,
           edge_property: str = "density", interval: int = 100,
           figsize: tuple = (10, 8)) -> Dict[str, Any]:
    """
    Create an animation of network evolution over time.

    Args:
        sim_id: Simulation ID
        start_time: Starting time step
        end_time: Ending time step (defaults to simulation end)
        edge_property: Property to visualize ('density', 'flow', 'speed', 'num_pedestrians')
        interval: Animation interval in milliseconds
        figsize: Figure size as (width, height)

    Returns:
        Dictionary with path to saved animation
    """
    sim = sim_manager.get_simulation(sim_id)
    if not sim:
        raise ValueError(f"Simulation {sim_id} not found")

    if not sim.output_dir:
        raise ValueError(f"No outputs saved for simulation {sim_id} - call save_outputs first")

    try:
        # Create visualizer from saved data
        visualizer = NetworkVisualizer(simulation_dir=str(sim.output_dir))

        # Set end time if not provided
        if end_time is None:
            end_time = sim.current_step

        # Create animation
        anim = visualizer.animate_network(
            start_time=start_time,
            end_time=end_time,
            interval=interval,
            figsize=figsize,
            edge_property=edge_property
        )

        # Save animation as MP4
        filename = f"animation_{start_time}to{end_time}_{edge_property}.mp4"
        filepath = sim.output_dir / filename

        # Use matplotlib's FFMpegWriter
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=10, metadata=dict(artist='PedNStream'))
        anim.save(str(filepath), writer=writer)

        return {
            'sim_id': sim_id,
            'start_time': start_time,
            'end_time': end_time,
            'edge_property': edge_property,
            'file_path': str(filepath),
            'filename': filename
        }

    except Exception as e:
        raise Exception(f"Failed to create animation: {str(e)}")

@mcp.tool
def get_status(sim_id: str) -> Dict[str, Any]:
    """
    Get the current status of a simulation.

    Args:
        sim_id: Simulation ID

    Returns:
        Dictionary with simulation status and progress
    """
    sim = sim_manager.get_simulation(sim_id)
    if not sim:
        raise ValueError(f"Simulation {sim_id} not found")

    return sim.to_dict()

@mcp.tool
def list_simulations() -> Dict[str, Any]:
    """
    List all simulations and their status.

    Returns:
        Dictionary with list of all simulations
    """
    return {
        'simulations': sim_manager.list_simulations(),
        'total_count': len(sim_manager.simulations)
    }

@mcp.tool
def cancel_simulation(sim_id: str) -> Dict[str, Any]:
    """
    Cancel a running simulation.

    Args:
        sim_id: Simulation ID

    Returns:
        Dictionary with cancellation status
    """
    sim = sim_manager.get_simulation(sim_id)
    if not sim:
        raise ValueError(f"Simulation {sim_id} not found")

    if sim.status == SimulationStatus.RUNNING:
        sim_manager.update_status(sim_id, SimulationStatus.CANCELLED)
        return {'sim_id': sim_id, 'status': 'cancelled'}
    else:
        return {'sim_id': sim_id, 'status': sim.status.value, 'message': 'Simulation not running'}

# Resource for accessing saved simulation data
@mcp.resource("sim://{sim_id}/link_data")
def get_link_data(sim_id: str) -> str:
    """Get link data JSON for a simulation"""
    sim = sim_manager.get_simulation(sim_id)
    if not sim or not sim.output_dir:
        raise ValueError(f"No output data for simulation {sim_id}")

    link_data_path = sim.output_dir / "link_data.json"
    if not link_data_path.exists():
        raise ValueError(f"Link data not found for simulation {sim_id}")

    with open(link_data_path, 'r') as f:
        return f.read()

@mcp.resource("sim://{sim_id}/node_data")
def get_node_data(sim_id: str) -> str:
    """Get node data JSON for a simulation"""
    sim = sim_manager.get_simulation(sim_id)
    if not sim or not sim.output_dir:
        raise ValueError(f"No output data for simulation {sim_id}")

    node_data_path = sim.output_dir / "node_data.json"
    if not node_data_path.exists():
        raise ValueError(f"Node data not found for simulation {sim_id}")

    with open(node_data_path, 'r') as f:
        return f.read()

@mcp.resource("sim://{sim_id}/network_params")
def get_network_params(sim_id: str) -> str:
    """Get network parameters JSON for a simulation"""
    sim = sim_manager.get_simulation(sim_id)
    if not sim or not sim.output_dir:
        raise ValueError(f"No output data for simulation {sim_id}")

    params_path = sim.output_dir / "network_params.json"
    if not params_path.exists():
        raise ValueError(f"Network params not found for simulation {sim_id}")

    with open(params_path, 'r') as f:
        return f.read()

@mcp.resource("sim://{sim_id}/time_series")
def get_time_series(sim_id: str) -> str:
    """Get time series CSV data for a simulation"""
    sim = sim_manager.get_simulation(sim_id)
    if not sim or not sim.output_dir:
        raise ValueError(f"No output data for simulation {sim_id}")

    time_series_path = sim.output_dir / "time_series.csv"
    if not time_series_path.exists():
        raise ValueError(f"Time series data not found for simulation {sim_id}")

    with open(time_series_path, 'r') as f:
        return f.read()


# ---------------------------
# Config authoring utilities
# ---------------------------
import re
from copy import deepcopy

# def _deep_get(d, path, default=None):
#     cur = d
#     for p in path:
#         if not isinstance(cur, dict) or p not in cur:
#             return default
#         cur = cur[p]
#     return cur

def _normalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize incoming config to canonical form expected by env_loader:
    {
      'params': { simulation_steps, unit_time, assign_flows_type, path_finder, default_link, links },
      'origin_nodes': [...],
      'destination_nodes': [...],
      'adjacency_matrix': [[...]] (optional),
      'demand': {...} (optional),
      'od_flows': {...} (optional)
    }
    Supports legacy keys: 'simulation', 'default_link', 'links', 'network'.
    """
    cfg = deepcopy(cfg) if isinstance(cfg, dict) else {}
    normalized: Dict[str, Any] = {}

    # Top-level carry-over
    for key in ['origin_nodes', 'destination_nodes', 'demand', 'od_flows', 'adjacency_matrix']:
        if key in cfg:
            normalized[key] = cfg[key]

    # From legacy 'network' block
    net = cfg.get('network', {})
    if isinstance(net, dict):
        if 'origin_nodes' in net and 'origin_nodes' not in normalized:
            normalized['origin_nodes'] = net['origin_nodes']
        if 'destination_nodes' in net and 'destination_nodes' not in normalized:
            normalized['destination_nodes'] = net['destination_nodes']
        if 'adjacency_matrix' in net and 'adjacency_matrix' not in normalized:
            normalized['adjacency_matrix'] = net['adjacency_matrix']

    # Build params
    params: Dict[str, Any] = {}


    # Legacy blocks → params
    sim = cfg.get('simulation', {})
    if isinstance(sim, dict):
        if 'simulation_steps' in sim:
            params['simulation_steps'] = sim['simulation_steps']
        if 'unit_time' in sim:
            params['unit_time'] = sim['unit_time']
        if 'assign_flows_type' in sim:
            params['assign_flows_type'] = sim['assign_flows_type']
        if 'path_finder' in sim and isinstance(sim['path_finder'], dict):
            params['path_finder'] = sim['path_finder']

    if isinstance(cfg.get('default_link'), dict):
        params['default_link'] = cfg['default_link']

    if isinstance(cfg.get('links'), dict):
        params['links'] = cfg['links']

    # Ensure required containers exist
    if 'links' not in params:
        params['links'] = {}
    if 'path_finder' not in params:
        params['path_finder'] = {}

    normalized['params'] = params
    return normalized

def _validate_config_struct(cfg: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Minimal validation tailored to env_loader expectations.
    Returns a list of error dicts: { 'path': 'a.b.c', 'message': '...' }
    """
    errors: List[Dict[str, str]] = []

    def err(path, msg):
        errors.append({'path': path, 'message': msg})

    # params required (normalized structure)
    params = cfg.get('params')
    if not isinstance(params, dict):
        err('params', 'params must be an object')
        return errors

    # required numeric fields
    sim_steps = params.get('simulation_steps')
    if not isinstance(sim_steps, int) or sim_steps <= 0:
        err('params.simulation_steps', 'must be a positive integer')

    unit_time = params.get('unit_time')
    if unit_time is not None and (not isinstance(unit_time, int) or unit_time <= 0):
        err('params.unit_time', 'must be a positive integer if provided')

    # default_link required subset
    dl = params.get('default_link')
    if not isinstance(dl, dict):
        err('params.default_link', 'default_link must be an object')
    else:
        for k in ['length', 'width', 'free_flow_speed', 'k_critical', 'k_jam']:
            if k not in dl:
                err(f'params.default_link.{k}', 'required')
        # basic numeric checks
        for k in ['length', 'width', 'free_flow_speed']:
            if k in dl and (not isinstance(dl[k], (int, float)) or dl[k] <= 0):
                err(f'params.default_link.{k}', 'must be > 0')
        for k in ['k_critical', 'k_jam']:
            if k in dl and (not isinstance(dl[k], (int, float)) or dl[k] <= 0):
                err(f'params.default_link.{k}', 'must be > 0')

    # links optional
    links = params.get('links', {})
    if not isinstance(links, dict):
        err('params.links', 'must be an object/map')
    else:
        key_re = re.compile(r'^\d+_\d+$')
        for lk, lconf in links.items():
            if not key_re.match(str(lk)):
                err(f'params.links.{lk}', 'key should be "u_v" where u and v are integers')
            if not isinstance(lconf, dict):
                err(f'params.links.{lk}', 'link config must be an object')

    # origin/destination nodes optional
    for key in ['origin_nodes', 'destination_nodes']:
        arr = cfg.get(key)
        if arr is not None:
            if not (isinstance(arr, list) and all(isinstance(x, int) for x in arr)):
                err(key, 'must be a list of integers')

    # adjacency_matrix if provided
    am = cfg.get('adjacency_matrix')
    if am is not None:
        if not (isinstance(am, list) and all(isinstance(row, list) for row in am)):
            err('adjacency_matrix', 'must be a list of lists')
        else:
            n = len(am)
            if n == 0:
                err('adjacency_matrix', 'must be non-empty when provided')
            else:
                for i, row in enumerate(am):
                    if len(row) != n:
                        err(f'adjacency_matrix[{i}]', 'matrix must be square (NxN)')
                    for j, v in enumerate(row):
                        if not isinstance(v, (int, float)):
                            err(f'adjacency_matrix[{i}][{j}]', 'entries must be numeric')

    # od_flows if present
    of = cfg.get('od_flows')
    if of is not None:
        if not isinstance(of, dict):
            err('od_flows', 'must be an object/map')
        else:
            key_re = re.compile(r'^\d+_\d+$')
            for k, v in of.items():
                if not key_re.match(str(k)):
                    err(f'od_flows.{k}', 'key should be "o_d" where o and d are integers')
                if not isinstance(v, (int, float)) or v < 0:
                    err(f'od_flows.{k}', 'value must be a non-negative number')

    return errors

def _example_yaml() -> str:
    return """(canonical)
network:
  origin_nodes: [1]
  destination_nodes: [5] # note that the node id starts from 0, max is the number of nodes - 1
# Optional: inline adjacency or provide files in data/<name>/
  adjacency_matrix: [[0,1,1,0,0,0], [1,0,0,1,0,0], [1,0,0,1,1,0], [0,1,1,0,0,1], [0,0,1,0,0,1], [0,0,0,1,1,0]]

simulation:
  simulation_steps: 500
  unit_time: 10
  assign_flows_type: classic
  path_finder: # Optional, only if user asks for specific path finder params
    k_paths: 4
    theta: 10
    alpha: 1
    beta: 0.5
    omega: 0.8
default_link:
  length: 100
  width: 1.0
  free_flow_speed: 1.1
  k_critical: 2
  k_jam: 10
  gamma: 0.01
links:  # Optional, only if user asks for specific links params
  "1_2":
  length: 120
  width: 1.5
demand: # only for origin nodes 
  origin_1:
    pattern: gaussian_peaks
    peak_lambda: 35
    base_lambda: 5
    seed: 42
od_flows: # only for origin and destination pairs
  "1_5": 30
"""

@mcp.tool
def list_config_schema() -> Dict[str, Any]:
    """
    Return a canonical schema description and an example YAML to guide config authoring.
    """
    schema = {
        "type": "object",
        "properties": {
            "params": {
                "type": "object",
                "properties": {
                    "simulation_steps": {"type": "integer", "minimum": 1},
                    "unit_time": {"type": "integer", "minimum": 1},
                    "assign_flows_type": {"type": "string"},
                    "path_finder": {"type": "object"},
                    "default_link": {"type": "object"},
                    "links": {"type": "object"}, # Optional, only if user asks for specific links params
                },
                "required": ["simulation_steps", "default_link"]
            },
            "origin_nodes": {"type": "array", "items": {"type": "integer"}},
            "destination_nodes": {"type": "array", "items": {"type": "integer"}},
            "adjacency_matrix": {"type": "array"},
            "demand": {"type": "object"},
            "od_flows": {"type": "object"}
        },
        "required": ["params"]
    }
    return {
        "schema": schema,
        "example_yaml": _example_yaml(),
        "notes": " "
    }


def _validate_config_impl(config: Dict[str, Any] = None, yaml_text: str = None) -> Dict[str, Any]:
    """
    Validate and normalize a proposed config. Accepts either a dict 'config' or a YAML string 'yaml_text'.
    Returns: { ok: bool, errors?: [...], normalized?: dict }
    """
    if config is None and yaml_text is None:
        return {"ok": False, "errors": [{"path": "", "message": "Provide either 'config' or 'yaml_text'"}]}

    raw = None
    if yaml_text is not None:
        try:
            import yaml  # Lazy import to avoid hard dependency at server import time
        except Exception as e:
            return {"ok": False, "errors": [{"path": "yaml_text", "message": f"PyYAML not available: {e}"}]}
        try:
            raw = yaml.safe_load(yaml_text) or {}
        except Exception as e:
            return {"ok": False, "errors": [{"path": "yaml_text", "message": f"YAML parse error: {e}"}]}
    else:
        raw = config

    if not isinstance(raw, dict):
        return {"ok": False, "errors": [{"path": "", "message": "Parsed content must be an object"}]}

    normalized = _normalize_config(raw)
    errors = _validate_config_struct(normalized)
    return {"ok": len(errors) == 0, "errors": errors, "normalized": normalized}


@mcp.tool
def validate_config(config: Dict[str, Any] = None, yaml_text: str = None) -> Dict[str, Any]:
    """
    Validate and normalize a proposed config. Accepts either a dict 'config' or a YAML string 'yaml_text'.
    Returns: { ok: bool, errors?: [...], normalized?: dict }
    """
    return _validate_config_impl(config=config, yaml_text=yaml_text)


@mcp.tool
def validate_config_file(yaml_file_path: str) -> Dict[str, Any]:
    """
    Validate and normalize a YAML config provided by file path (e.g., input/exp1.yaml).
    Returns: { ok: bool, errors?: [...], normalized?: dict }
    """
    try:
        file_path = Path(yaml_file_path)
        if not file_path.exists():
            return {"ok": False, "errors": [{"path": "yaml_file_path", "message": f"YAML file not found: {yaml_file_path}"}]}
        with open(file_path, 'r', encoding='utf-8') as f:
            yaml_text = f.read()
        return _validate_config_impl(yaml_text=yaml_text)
    except Exception as e:
        return {"ok": False, "errors": [{"path": "yaml_file_path", "message": str(e)}]}


def _sanitize_name(name: str) -> str:
    import re as _re
    if not isinstance(name, str) or not _re.match(r"^[A-Za-z0-9_\-]+$", name):
        raise ValueError("Invalid name. Use only letters, numbers, '-', '_' ")
    return name


@mcp.tool
def upsert_config(name: str, config: Dict[str, Any] = None, yaml_text: str = None) -> Dict[str, Any]:
    """
    Validate, normalize, and persist a config to input/<name>.yaml.
    Returns: { ok, path, wrote_bytes, normalized_preview }
    """
    name = _sanitize_name(name)

    # Validate/normalize using existing logic
    val = _validate_config_impl(config=config, yaml_text=yaml_text)
    if not val.get("ok"):
        return {"ok": False, "errors": val.get("errors", [])}

    normalized = val.get("normalized", {})

    try:
        import yaml  # Lazy import
    except Exception as e:
        return {"ok": False, "errors": [{"path": "", "message": f"PyYAML not available: {e}"}]}

    data_root = project_root / "mcp" / "input"
    target_dir = data_root / name + ".yaml"
    target_dir.mkdir(parents=True, exist_ok=True)
    # target_path = target_dir / "sim_params.yaml"
    target_path = target_dir + ".yaml"

    # Write canonical YAML
    yaml_text_out = yaml.safe_dump(normalized, sort_keys=False)
    with open(target_path, "w", encoding="utf-8") as f:
        wrote = f.write(yaml_text_out)

    return {
        "ok": True,
        "path": str(target_path),
        "wrote_bytes": wrote,
        "normalized_preview": normalized,
    }


@mcp.tool
def read_config(name: str) -> Dict[str, Any]:
    """
    Read the raw YAML text for input/<name>.yaml.
    Returns: { name, yaml_text }
    """
    name = _sanitize_name(name)
    target = project_root / "mcp" / "input" / name + ".yaml"
    if not target.exists():
        raise ValueError(f"Config not found: {target}")
    with open(target, "r", encoding="utf-8") as f:
        return {"name": name, "yaml_text": f.read()}


# @mcp.tool
# def create_environment_from_config(base_config_name: str, config: Dict[str, Any] = None, yaml_text: str = None) -> Dict[str, Any]:
#     """
#     Create an environment using a provided config (dict or YAML) applied as overrides
#     on top of a base dataset under data/<base_config_name>/, without persisting a new file.
#     Returns same shape as create_environment.
#     """
#     # Validate/normalize
#     val = _validate_config_impl(config=config, yaml_text=yaml_text)
#     if not val.get("ok"):
#         return {"ok": False, "errors": val.get("errors", [])}
#     overrides = val.get("normalized", {})

#     # Use core implementation to avoid calling tool-decorated functions directly
#     try:
#         result = _create_environment_impl(config_name=base_config_name, overrides=overrides)
#         result["ok"] = True
#         return result
#     except Exception as e:
#         return {"ok": False, "error": str(e)}

@mcp.tool
def create_environment_from_file(yaml_file_path: str) -> Dict[str, Any]:
    """
    Create an environment directly from a YAML file path (e.g., input/my_config.yaml).
    The YAML file should be complete and self-contained.
    Returns same shape as create_environment.
    """
    try:
        # Read the YAML file (no re-validation here; assume caller validated already)
        file_path = Path(yaml_file_path)
        if not file_path.exists():
            return {"ok": False, "error": f"YAML file not found: {yaml_file_path}"}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            yaml_text = f.read()
        
        # Create simulation entry with file path as config name
        sim_id = sim_manager.create_simulation(str(file_path), overrides={})
        
        # The env_loader expects a directory containing sim_params.yaml.
        # We'll create a temporary directory structure to accommodate this.
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_subdir_name = "temp_config"
            config_dir = Path(temp_dir) / config_subdir_name
            config_dir.mkdir()

            temp_yaml_path = config_dir / "sim_params.yaml"
            with open(temp_yaml_path, 'w', encoding='utf-8') as f:
                f.write(yaml_text)
            
            # Create network environment pointing to our temporary data directory
            env_generator = NetworkEnvGenerator(data_dir=temp_dir)
            network = env_generator.create_network(config_subdir_name)
        
        # Update simulation state
        sim = sim_manager.get_simulation(sim_id)
        sim.network = network
        sim.total_steps = network.simulation_steps
        
        return {
            "ok": True,
            'sim_id': sim_id,
            'config_name': str(file_path),
            'total_steps': network.simulation_steps,
            'origin_nodes': network.origin_nodes,
            'destination_nodes': network.destination_nodes,
            'num_nodes': len(network.nodes),
            'num_links': len(network.links),
            'config_source': 'file'
        }
        
    except Exception as e:
        return {"ok": False, "error": str(e)}

@mcp.tool
def list_input_files() -> Dict[str, Any]:
    """
    List available YAML files in the /input directory.
    Returns: { files: [{"name": "file.yaml", "path": "full/path", "size": 1234, "modified": "2023-..."}, ...] }
    """
    try:
        input_dir = project_root / "mcp" / "input"
        files = []
        
        if input_dir.exists():
            for file_path in input_dir.glob("*.yaml"):
                if file_path.is_file():
                    stat = file_path.stat()
                    files.append({
                        "name": file_path.name,
                        "path": str(file_path),
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
        
        return {
            "ok": True,
            "files": sorted(files, key=lambda x: x["modified"], reverse=True),  # Most recent first
            "count": len(files)
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

if __name__ == "__main__":
    # Run the server
    mcp.run(transport="http", host="127.0.0.1", port=8000)
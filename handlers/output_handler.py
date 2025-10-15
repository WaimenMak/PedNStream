

import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from src.LTM.network import Network

class OutputHandler:
    def __init__(self, base_dir="outputs", simulation_dir=None):
        """
        Initialize output handler
        :param base_dir: Base directory for outputs
        """
        self.base_dir = Path(base_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if simulation_dir is not None:
            self.simulation_dir = self.base_dir / simulation_dir
        else:
            self.simulation_dir = self.base_dir / f"sim_{self.timestamp}"
        
        # Create output directories if they don't exist
        self.simulation_dir.mkdir(parents=True, exist_ok=True)
        
    def save_network_state(self, network: Network):
        """
        Save complete network state including link and node data
        network: Network object
        """
        # Save link data
        link_data = {}
        for (u, v), link in network.links.items():
            link_data[f"{u}-{v}"] = {
                'density': link.density.tolist(),
                'link_flow': link.link_flow.tolist(),
                'speed': link.speed.tolist(),
                'travel_time': link.travel_time.tolist(),
                'inflow': link.inflow.tolist(),
                'outflow': link.outflow.tolist(),
                'num_pedestrians': link.num_pedestrians.tolist(),
                'cumulative_inflow': link.cumulative_inflow.tolist(),
                'cumulative_outflow': link.cumulative_outflow.tolist(),
                'sending_flow': link.sending_flow.tolist(),
                'receiving_flow': link.receiving_flow.tolist(),
                'parameters': {
                    'length': link.length,
                    'width': link.width,
                    'free_flow_speed': link.free_flow_speed,
                    'k_critical': link.k_critical,
                    'k_jam': link.k_jam
                }
            }
        
        # Save node data
        node_data = {}
        for node_id, node in network.nodes.items():
            node_data[node.node_id] = {
                'demand': node.demand.tolist() if node.demand is not None else [],
                'incoming_links': [link.link_id for link in node.incoming_links],
                'outgoing_links': [link.link_id for link in node.outgoing_links]
            }
        
        # Save network parameters
        network_params = {
            'simulation_steps': network.simulation_steps,
            'unit_time': network.unit_time,
            'destination_nodes': network.destination_nodes,
            'origin_nodes': network.origin_nodes,
            # Add other relevant network parameters
            # save the od paths
            'od_paths': (
                {f"{k[0]}-{k[1]}": v for k, v in network.path_finder.od_paths.items()}
                if getattr(network, 'path_finder', None) is not None else {}
            )
        }
        
        # Save to files
        self._save_json(link_data, 'link_data.json')
        self._save_json(node_data, 'node_data.json')
        self._save_json(network_params, 'network_params.json')
        
    def save_time_series(self, network: Network):
        """
        Save time series data in CSV format for easy analysis
        network: Network object
        """
        # Prepare link time series data
        link_series = []
        for (u, v), link in network.links.items():
            for t in range(network.simulation_steps):
                link_series.append({
                    'time_step': t,
                    'link_id': f"{u}-{v}",
                    'density': link.density[t],
                    'speed': link.speed[t],
                    'inflow': link.inflow[t],
                    'outflow': link.outflow[t],
                    'num_pedestrians': link.num_pedestrians[t],
                    'cumulative_inflow': link.cumulative_inflow[t],
                    'cumulative_outflow': link.cumulative_outflow[t]
                })
        
        # Save to CSV
        df = pd.DataFrame(link_series)
        df.to_csv(self.simulation_dir / 'time_series.csv', index=False)
    
    def _save_json(self, data, filename):
        """Helper method to save JSON data"""
        filepath = self.simulation_dir / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def load_simulation(simulation_dir: str):
        """
        Load saved simulation data
        :param simulation_dir: Directory containing simulation data
        :return: Dictionary containing simulation data
        """
        data = {}
        simulation_path = Path(simulation_dir)
        
        # Load JSON files
        for filename in ['link_data.json', 'node_data.json', 'network_params.json']:
            filepath = simulation_path / filename
            if filepath.exists():
                with open(filepath, 'r') as f:
                    data[filename.replace('.json', '')] = json.load(f)
        
        # Load time series data if it exists
        csv_path = simulation_path / 'time_series.csv'
        if csv_path.exists():
            data['time_series'] = pd.read_csv(csv_path)
            
        return data
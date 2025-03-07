import yaml
import numpy as np
from typing import Dict, Any

def load_config(config_path: str) -> dict:
    """
    Load and validate configuration from YAML file
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        dict: Processed configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate configuration
    validate_config(config)
    
    # Convert adjacency matrix to numpy array
    config['network']['adjacency_matrix'] = np.array(
        config['network']['adjacency_matrix']
    )
    
    # Combine all parameters into link_params
    link_params = {
        'simulation_steps': config['simulation']['simulation_steps'],
        'unit_time': config['simulation']['unit_time'],
        'default_link': config['default_link'],
        'links': config.get('links', {}),
        'demand': config.get('demand', {})
    }
    
    # Convert OD flows from string keys to tuple keys
    od_flows = {}
    for od_pair, flow in config.get('od_flows', {}).items():
        origin, dest = map(int, od_pair.split('_'))
        od_flows[(origin, dest)] = flow
    
    return {
        'adjacency_matrix': config['network']['adjacency_matrix'],
        'params': link_params,
        'origin_nodes': config['network']['origin_nodes'],
        'destination_nodes': config['network'].get('destination_nodes', []),
        'od_flows': od_flows
    }

def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration parameters
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_fields = {
        'network': ['adjacency_matrix', 'origin_nodes'],
        'simulation': ['simulation_steps', 'unit_time'],
        'default_link': ['length', 'width', 'free_flow_speed', 'k_critical', 'k_jam'],
        'demand': [],
        # 'od_flows': []
    }
    
    for section, fields in required_fields.items():
        if section not in config:
            raise ValueError(f"Missing required section: {section}")
        
        for field in fields:
            if field not in config[section]:
                raise ValueError(f"Missing required field: {field} in section {section}")
    
    # Validate adjacency matrix dimensions
    adj_matrix = np.array(config['network']['adjacency_matrix'])
    if adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError("Adjacency matrix must be square")
    
    # Validate origin and destination nodes
    max_node = adj_matrix.shape[0] - 1
    # for node in config['network']['origin_nodes'] + config['network']['destination_nodes']:
    #     if node > max_node:
    #         raise ValueError(f"Node {node} exceeds network size")
            
    # Validate demand patterns
    for origin_id, demand_config in config.get('demand', {}).items():
        if 'pattern' not in demand_config:
            raise ValueError(f"Missing pattern in demand config for {origin_id}") 
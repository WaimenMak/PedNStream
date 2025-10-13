import yaml
import numpy as np
from typing import Dict, Any

def load_config(config_path: str) -> dict:
    """
    Load and validate configuration from a YAML file with a flattened structure.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        dict: Processed configuration dictionary, structured for the Network class.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 1. Assemble the 'params' dictionary required by the Network class
    path_finder_params = config['simulation'].get('path_finder', {})
    
    # Add controllers configuration to path_finder params if present
    if 'controllers' in config:
        path_finder_params['controllers'] = {
            'enabled': config['controllers'].get('enabled', True),
            'nodes': config['controllers'].get('nodes', []),
            'links': config['controllers'].get('links', []),
            'schedule': config['controllers'].get('schedule', {})
        }
    else:
        # Default: no controllers
        path_finder_params['controllers'] = {
            'enabled': False,
            'nodes': [],
            'links': [],
            'schedule': {}
        }
    
    params = {
        'simulation_steps': config['simulation']['simulation_steps'],
        'unit_time': config['simulation']['unit_time'],
        'assign_flows_type': config['simulation'].get('assign_flows_type', 'classic'),
        'seed': config['simulation'].get('seed', None),
        'path_finder': path_finder_params,
        'default_link': config['default_link'],
        'links': config.get('links', {}),
        'demand': config.get('demand', {})
    }

    # 2. Assemble the final result dictionary
    result = {
        'params': params,
        'origin_nodes': config['network']['origin_nodes'],
        'destination_nodes': config['network'].get('destination_nodes', [])
    }

    if 'adjacency_matrix' in config['network']:
        result['adjacency_matrix'] = np.array(config['network']['adjacency_matrix'])

    # 4. Handle optional 'od_flows'
    if 'od_flows' in config:
        od_flows = {}
        for od_pair, flow in config['od_flows'].items():
            origin, dest = map(int, od_pair.split('_'))
            od_flows[(origin, dest)] = flow
        result['od_flows'] = od_flows

    return result

def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration parameters
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_fields = {
        'network': ['origin_nodes'],
        'simulation': ['simulation_steps', 'unit_time'],
        'default_link': ['length', 'width', 'free_flow_speed', 'k_critical', 'k_jam'],
    }
    
    for section, fields in required_fields.items():
        if section not in config:
            raise ValueError(f"Missing required section: {section}")
        
        for field in fields:
            if field not in config[section]:
                raise ValueError(f"Missing required field: {field} in section {section}")
    
    # Validate origin and destination nodes
    # ... (validation logic can be updated if needed) ... 
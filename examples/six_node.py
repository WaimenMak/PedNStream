import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
from handlers.output_handler import OutputHandler
from src.utils.visualizer import NetworkVisualizer
from src.LTM.network import Network
# from src.utils.config import load_config
from src.utils.env_loader import NetworkEnvGenerator

def main():

    # Load configuration
    # config_path = os.path.join(project_root, 'sim_params.yaml')
    # config = load_config(config_path)
    env_generator = NetworkEnvGenerator()
    network_env = env_generator.create_network('od_flow_example')
    
    # Initialize network and run simulation
    # network_env = Network(**config)
    
    for t in range(1, env_generator.config['params']['simulation_steps']):
        network_env.network_loading(t)
    
    # Save and visualize results
    # output_dir = os.path.join(project_root, "outputs")
    output_dir = os.path.join('..', 'output')
    output_handler = OutputHandler(base_dir=output_dir, simulation_dir="path_finder")
    output_handler.save_network_state(network_env)
    
    # Create animation
    matplotlib.use('macosx')
    visualizer = NetworkVisualizer(
        simulation_dir=os.path.join(output_dir, "path_finder")
    )
    anim = visualizer.animate_network(
        start_time=0,
        end_time=env_generator.config['params']['simulation_steps'],
        interval=100,
        edge_property='density'
    )
    
    plt.show()

if __name__ == "__main__":
    main()
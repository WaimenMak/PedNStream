import os
import sys
import matplotlib.pyplot as plt
import matplotlib
from handlers.output_handler import OutputHandler
from src.utils.visualizer import NetworkVisualizer
from src.LTM.network import Network
from src.utils.config import load_config

def main():
    # Setup paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)
    
    # Load configuration
    config_path = os.path.join(project_root, 'examples', 'configs', 'od_flow_example.yaml')
    config = load_config(config_path)
    
    # Initialize network and run simulation
    network_env = Network(**config)
    
    for t in range(1, config['params']['simulation_steps']):
        network_env.network_loading(t)
    
    # Save and visualize results
    output_dir = os.path.join(project_root, "outputs")
    output_handler = OutputHandler(base_dir=output_dir, simulation_dir="path_finder")
    output_handler.save_network_state(network_env)
    
    # Create animation
    matplotlib.use('macosx')
    visualizer = NetworkVisualizer(
        simulation_dir=os.path.join(output_dir, "path_finder")
    )
    anim = visualizer.animate_network(
        start_time=0,
        end_time=config['params']['simulation_steps'],
        interval=100,
        edge_property='density'
    )
    
    plt.show()

if __name__ == "__main__":
    main()
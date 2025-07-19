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

    # Setup paths
    project_root = Path(__file__).resolve().parent.parent
    sys.path.append(str(project_root))

    # Load configuration
    # config_path = os.path.join(project_root, 'sim_params.yaml')
    # config = load_config(config_path)
    env_generator = NetworkEnvGenerator()
    network_env = env_generator.create_network('od_flow_example')
    
    # Initialize network and run simulation
    # network_env = Network(**config)
    
    for t in range(1, env_generator.config['params']['simulation_steps']):
        network_env.network_loading(t)
        if t in [100, 101, 102, 103, 104, 105, 106, 107, 108]:
            network_env.links[(3, 5)].back_gate_width -= 0.1
        print(network_env.nodes[1].turning_fractions)
    
    # Save and visualize results
    output_dir = project_root / "outputs"
    simulation_dir_name = "six_node_exp"
    output_handler = OutputHandler(base_dir=str(output_dir), simulation_dir=simulation_dir_name)
    output_handler.save_network_state(network_env)
    
    # Create animation
    matplotlib.use('macosx')
    visualizer = NetworkVisualizer(
        simulation_dir=os.path.join(output_dir, simulation_dir_name)
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
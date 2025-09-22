# -*- coding: utf-8 -*-
# @Time    : 19/09/2025 16:59
# @Author  : mmai
# @FileName: run_mcp
# @Software: PyCharm

#!/usr/bin/env python3
"""
Example usage of the PedNStream MCP Server
"""

from fastmcp import Client
import asyncio
import json

async def example_workflow():
    """Example workflow demonstrating the MCP server capabilities"""

    # Connect to the server (assuming it's running with STDIO transport)
    client = Client("stdio://python mcp_server.py")

    try:
        # 1. Create environment
        print("Creating simulation environment...")
        env_result = await client.call_tool("create_environment", {
            "config_name": "delft",
            "overrides": {
                "simulation": {"simulation_steps": 100}  # Shorter run for demo
            }
        })

        sim_id = env_result["sim_id"]
        print(f"Created simulation {sim_id}")
        print(f"Network: {env_result['num_nodes']} nodes, {env_result['num_links']} links")

        # 2. Run simulation
        print("Running simulation...")
        run_result = await client.call_tool("run_simulation", {
            "sim_id": sim_id,
            "steps": 50  # Run first 50 steps
        })
        print(f"Simulation progress: {run_result['progress']:.1f}%")

        # 3. Save outputs
        print("Saving outputs...")
        save_result = await client.call_tool("save_outputs", {
            "sim_id": sim_id,
            "include_time_series": True
        })
        print(f"Saved to: {save_result['output_dir']}")

        # 4. Create visualization
        print("Creating snapshot visualization...")
        viz_result = await client.call_tool("visualize_snapshot", {
            "sim_id": sim_id,
            "time_step": 25,
            "edge_property": "density"
        })
        print(f"Visualization saved: {viz_result['filename']}")

        # 5. Access data via resources
        print("Accessing network parameters...")
        network_params = await client.read_resource(f"sim://{sim_id}/network_params")
        params = json.loads(network_params)
        print(f"Simulation steps: {params['simulation_steps']}")

        # 6. Check status
        status = await client.call_tool("get_status", {"sim_id": sim_id})
        print(f"Final status: {status['status']}")

    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(example_workflow())
#!/usr/bin/env python3
"""
HTTP client demo for the PedNStream MCP server
"""
import asyncio
import json
import os
from fastmcp import Client

async def test_validate_config():
    """Test the validate_config tool with exp1.yaml"""
    # Connect to the HTTP server
    async with Client("http://127.0.0.1:8000/mcp") as client:
        await client.ping()
        print("✓ Server reachable")

        # Read the exp1.yaml file
        yaml_file_path = os.path.join(os.path.dirname(__file__), "input", "exp1.yaml")
        print(f"Reading YAML file: {yaml_file_path}")
        
        try:
            with open(yaml_file_path, "r", encoding="utf-8") as f:
                yaml_content = f.read()
            print(f"YAML content:\n{yaml_content}")
        except FileNotFoundError:
            print(f"❌ File not found: {yaml_file_path}")
            return
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return

        # Test validate_config with yaml_text
        print("\n--- Testing validate_config with yaml_text ---")
        try:
            result = await client.call_tool("validate_config", {
                "yaml_text": yaml_content
            })
            print(f"Validation result: {json.dumps(result.data, indent=2)}")
            
            if result.data.get("ok"):
                print("✅ Validation successful!")
                print(f"Normalized config keys: {list(result.data.get('normalized', {}).keys())}")
            else:
                print("❌ Validation failed!")
                for error in result.data.get("errors", []):
                    print(f"  Error at '{error.get('path', '')}': {error.get('message', '')}")
                    
        except Exception as e:
            print(f"❌ Error calling validate_config: {e}")

        # If validation successful, test the full workflow
        if result.data.get("ok"):
            print("\n--- Testing complete workflow: create environment and run simulation ---")
            try:
                # Test create_environment_from_file
                env_result = await client.call_tool("create_environment_from_file", {
                    "yaml_file_path": yaml_file_path
                })
                print(f"Environment creation result: {json.dumps(env_result.data, indent=2)}")
                
                if env_result.data.get("ok", True):
                    sim_id = env_result.data.get("sim_id")
                    print(f"✅ Environment created! Sim ID: {sim_id}")
                    
                    # Run simulation for a few steps
                    print(f"\n--- Running simulation {sim_id} for 30 steps ---")
                    run_result = await client.call_tool("run_simulation", {
                        "sim_id": sim_id,
                        "steps": 30
                    })
                    print(f"Simulation run result: {json.dumps(run_result.data, indent=2)}")
                    
                    if run_result.data.get("ok", True):
                        progress = run_result.data.get("progress", 0)
                        print(f"✅ Simulation completed! Progress: {progress:.1f}%")
                        
                        # Get final status
                        status_result = await client.call_tool("get_status", {"sim_id": sim_id})
                        print(f"Final status: {json.dumps(status_result.data, indent=2)}")
                        
                        # Save outputs
                        print(f"\n--- Saving outputs for {sim_id} ---")
                        save_result = await client.call_tool("save_outputs", {
                            "sim_id": sim_id,
                            "include_time_series": True
                        })
                        print(f"Save result: {json.dumps(save_result.data, indent=2)}")
                        
                        if save_result.data.get("ok", True):
                            output_dir = save_result.data.get("output_dir")
                            print(f"✅ Outputs saved to: {output_dir}")
                        else:
                            print("❌ Failed to save outputs")
                    else:
                        print("❌ Simulation run failed")
                else:
                    print("❌ Environment creation failed!")
                    for error in env_result.data.get("errors", []):
                        print(f"  Error: {error}")
                        
            except Exception as e:
                print(f"❌ Error in complete workflow test: {e}")

        # Also test parsing the YAML manually to see what Python sees
        print("\n--- Manual YAML parsing test ---")
        try:
            import yaml
            parsed = yaml.safe_load(yaml_content)
            print(f"Manually parsed YAML type: {type(parsed)}")
            print(f"Manually parsed YAML content: {json.dumps(parsed, indent=2)}")
        except Exception as e:
            print(f"❌ Manual YAML parsing error: {e}")

async def http_demo():
    """Original demo - kept for reference"""
    # Connect to the HTTP server
    async with Client("http://127.0.0.1:8000/mcp") as client:
        # Optional: verify connectivity
        await client.ping()
        print("✓ Server reachable")

        # 1) List simulations (should be empty at first)
        sims = await client.call_tool("list_simulations", {})
        print(f"Simulations: {sims.data}")

        # 2) Create environment
        env = await client.call_tool("create_environment", {
            "config_name": "delft",
            "overrides": {"params": {"simulation_steps": 50}}
        })
        sim_id = env.data["sim_id"]
        print(f"✓ Environment created: {sim_id}")

        # 3) Run a few steps
        run = await client.call_tool("run_simulation", {
            "sim_id": sim_id,
            "steps": 25
        })
        print(f"Progress: {run.data['progress']:.1f}%")

        # 4) Save outputs (required before visualization/resources)
        save = await client.call_tool("save_outputs", {
            "sim_id": sim_id,
            "include_time_series": True
        })
        print(f"✓ Outputs saved to: {save.data['output_dir']}")

        # 5) Create a snapshot
        viz = await client.call_tool("visualize_snapshot", {
            "sim_id": sim_id,
            "time_step": 10,
            "edge_property": "density"
        })
        print(f"✓ Snapshot: {viz.data['filename']}")

        # 6) Read a resource (network params)
        content = await client.read_resource(f"sim://{sim_id}/network_params")
        params = json.loads(content[0].text)
        print(f"Simulation steps (from resource): {params['simulation_steps']}")

        # 7) Get status
        status = await client.call_tool("get_status", {"sim_id": sim_id})
        print(f"Status: {status.data['status']} at step {status.data['current_step']} / {status.data['total_steps']}")

if __name__ == "__main__":
    # Run the validation test
    asyncio.run(test_validate_config())
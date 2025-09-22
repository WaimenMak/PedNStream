#!/usr/bin/env python3
"""
Assistant harness to co-create a sim_params config with an LLM,
validate it via MCP tools, and run a short simulation.

Usage (in separate terminal from the server):
  export MCP_SERVER_URL=http://127.0.0.1:8000/mcp
  # Choose one provider and set credentials
  export LLM_PROVIDER=anthropic            # or: openai|openai_compat
  export ANTHROPIC_API_KEY=...             # for anthropic
  # export OPENAI_API_KEY=...              # for openai/openai_compat
  # export OPENAI_BASE_URL=...             # for openai_compat
  export ASSISTANT_MODEL=claude-3-5-sonnet-latest
  python mcp/assistant_harness.py

Optional envs:
  BASE_CONFIG_NAME=od_flow_example     # dataset under data/ used as base
  SIM_STEPS=50                         # steps to run
  PERSIST_CONFIG_NAME=my_config        # if set, upsert config to data/<name>/sim_params.yaml
"""

import os
import re
import json
import asyncio
from typing import Optional, Dict, Any

from fastmcp import Client


SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/mcp")
# BASE_CONFIG_NAME = os.getenv("BASE_CONFIG_NAME", "od_flow_example")
ASSISTANT_MODEL = os.getenv("ASSISTANT_MODEL", "gpt-4o-mini")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # openai | anthropic | openai_compat
SIM_STEPS = int(os.getenv("SIM_STEPS", "50"))
PERSIST_CONFIG_NAME = os.getenv("PERSIST_CONFIG_NAME")


def _extract_yaml(text: str) -> str:
    """Extract a YAML code block if present; otherwise return an empty string."""
    if not text:
        return ""
    # Only accept ```yaml blocks to avoid capturing non-YAML code fences
    m = re.search(r"```yaml(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # If no yaml code block is found, it's a conversational turn.
    return ""


async def llm_get_response(prompt: str, history: list = None) -> str:
    """Generate a response from an LLM using the selected provider."""
    provider = LLM_PROVIDER.lower()
    messages = history or []
    messages.append({"role": "user", "content": prompt})

    if provider == "anthropic":
        try:
            import anthropic  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Anthropic SDK not available: {e}")
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=os.getenv("ASSISTANT_MODEL", "claude-3-5-sonnet-latest"),
            max_tokens=2000,
            messages=messages,
        )
        text = "".join([b.text for b in msg.content if getattr(b, "type", "") == "text"])
        return text

    elif provider == "openai" or provider == "openai_compat":
        try:
            from openai import OpenAI  # type: ignore
            import openai
        except Exception as e:
            raise RuntimeError(f"openai SDK not available: {e}")

        client_args = {}
        if provider == "openai_compat":
            base_url = os.getenv("OPENAI_BASE_URL")
            api_key = os.getenv("OPENAI_API_KEY")
            if not base_url or not api_key:
                raise RuntimeError("OPENAI_BASE_URL and OPENAI_API_KEY are required for openai_compat provider")
            client_args['base_url'] = base_url
            client_args['api_key'] = api_key
            headers_env = os.getenv("OPENAI_DEFAULT_HEADERS")
            if headers_env:
                client_args['default_headers'] = json.loads(headers_env)
        
        client = OpenAI(**client_args)
        
        resp = client.chat.completions.create(
            model=os.getenv("ASSISTANT_MODEL", "gpt-4o-mini"),
            messages=messages,
            temperature=0.2,
        )
        text = resp.choices[0].message.content or ""
        return text

    else:
        raise RuntimeError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")


async def call_tool_data(client: Client, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    res = await client.call_tool(name, args)
    # Handle different FastMCP versions
    if hasattr(res, 'data'):
        return res.data
    elif hasattr(res, 'content'):
        # Some versions use content instead of data
        if isinstance(res.content, list) and len(res.content) > 0:
            return res.content[0].text if hasattr(res.content[0], 'text') else res.content[0]
        return res.content
    else:
        # Fallback: return the result directly if it's already a dict
        return res if isinstance(res, dict) else {}


async def main():
    print(f"Connecting to MCP server at {SERVER_URL} ...")
    async with Client(SERVER_URL) as client:
        await client.ping()
        print("✓ Server reachable")
        # Read base name locally to avoid any static-analysis warnings
        base_name = os.getenv("BASE_CONFIG_NAME", "od_flow_example")

        schema = await call_tool_data(client, "list_config_schema", {})
        example_yaml = schema.get("example_yaml", "")

        initial_prompt = f"""
You are an expert assistant for the PedNStream simulation tool. Your goal is to help me create `sim_params.yaml` configuration files, run simulations, and analyze results.

AVAILABLE TOOLS:
- list_input_files(): List saved YAML files in mcp/input/
- create_environment_from_file(yaml_file_path): Run simulation from a saved YAML file
- validate_config(yaml_text): Validate YAML configuration
- run_simulation(sim_id, steps): Execute simulation steps
- save_outputs(sim_id): Save simulation results

CAPABILITIES:
1. **Create YAML configs**: When asked to create/modify a simulation configuration, generate YAML in ```yaml blocks adhering to the canonical 'params' structure.

2. **Run simulations intelligently**: When asked to run a simulation:
   - First call list_input_files() to see available YAML files
   - If you find a relevant file, use create_environment_from_file(path) with the full file path
   - If no suitable file exists, ask the user to create one first

3. **Interactive workflow**: After creating/validating YAML, users can:
   - [r]un simulation: Create environment and run simulation directly from the validated YAML
   - [s]ave config: Save YAML to mcp/input/ for future use
   - [e]dit prompt: Modify the configuration
   - [q]uit: End session

4. **File matching**: Match user requests to existing files intelligently (e.g., "run the network with 4 nodes" → find "4_node_network.yaml")

YAML REQUIREMENTS:
- Create adjacency matrix when building networks (symmetric matrix)
- Use default_link params unless specific link params are requested
- Include origin_nodes, destination_nodes, demand, and od_flows

Example format:
```yaml
{example_yaml}
```

Always be helpful and proactive in suggesting the best workflow!
"""
        history = [{"role": "system", "content": initial_prompt}]
        last_saved_yaml_path: Optional[str] = None
        
        print("\n--- PedNStream Interactive Assistant ---")
        print("Describe the simulation you want to run. Type 'quit' or 'exit' to end.")

        loop = asyncio.get_running_loop()

        while True:
            try:
                user_prompt = await loop.run_in_executor(None, lambda: input("\n[user]> "))
            except (EOFError, KeyboardInterrupt):
                break

            if user_prompt.lower() in ["quit", "exit"]:
                break
            if not user_prompt:
                continue

            print("\nGenerating response...")
            full_response = await llm_get_response(user_prompt, history=history)
            
            print("\n--- Assistant ---")
            print(full_response)
            print("-----------------")

            # Always save the full conversation turn to history
            history.append({"role": "user", "content": user_prompt})
            history.append({"role": "assistant", "content": full_response})

            yaml_text = _extract_yaml(full_response)

            if not yaml_text:
                # No YAML in the reply: try running an existing saved YAML from mcp/input/
                print("\nNo YAML block found. Searching existing configs in mcp/input/ ...")
                try:
                    listing = await call_tool_data(client, "list_input_files", {})
                except Exception as e:
                    print(f"  ✗ Could not list input files: {e}")
                    continue
                files = listing.get("files") or []
                if not files:
                    print("  No saved YAML files found. Ask the assistant to generate one in a ```yaml block.")
                    continue
                chosen = files[0]
                yaml_file_to_use = chosen.get("path") or chosen.get("filepath") or chosen.get("full_path")
                if not yaml_file_to_use:
                    print(f"  ✗ Invalid file listing entry: {chosen}")
                    continue
                print(f"  Using most recent saved YAML: {yaml_file_to_use}")
                # Create environment from file
                env_result = await call_tool_data(client, "create_environment_from_file", {
                    "yaml_file_path": yaml_file_to_use
                })
                if not env_result.get("ok", True):
                    print(f"  ✗ Failed to create environment: {env_result}")
                    continue
                sim_id = env_result.get("sim_id")
                print(f"  ✓ Environment created! Sim ID: {sim_id}")
                # Run simulation
                run_result = await call_tool_data(client, "run_simulation", {
                    "sim_id": sim_id,
                    "steps": SIM_STEPS
                })
                if not run_result.get("ok", True):
                    print(f"  ✗ Simulation failed: {run_result}")
                    continue
                progress = run_result.get("progress", 0)
                print(f"  ✓ Simulation completed! Progress: {progress:.1f}%")
                # Save outputs
                save_result = await call_tool_data(client, "save_outputs", {
                    "sim_id": sim_id,
                    "include_time_series": True
                })
                if save_result.get("ok", True):
                    print(f"  ✓ Outputs saved to: {save_result.get('output_dir')}")
                else:
                    print("  ✗ Failed to save outputs")
                continue
            
            print("\nFound YAML config in response. Validating...")
            val = await call_tool_data(client, "validate_config", {"yaml_text": yaml_text})
            
            if not val.get("ok"):
                print("\nValidation FAILED:")
                for e in val.get("errors", []):
                    path = e.get('path', '')
                    msg = e.get('message', '')
                    print(f"- {path}: {msg}")
                # Print compact YAML preview (first 40 lines)
                preview_lines = yaml_text.splitlines()[:40]
                print("\nYAML preview (first lines):")
                print("\n".join(preview_lines))
                # Fallback: try existing saved YAMLs
                print("\nFalling back to existing saved YAMLs in mcp/input/ ...")
                try:
                    listing = await call_tool_data(client, "list_input_files", {})
                except Exception as e:
                    print(f"  ✗ Could not list input files: {e}")
                    continue
                files = listing.get("files") or []
                if not files:
                    print("  No saved YAML files found. Please refine your request or ask the assistant to produce a YAML config in a ```yaml block.")
                    continue
                chosen = files[0]
                yaml_file_to_use = chosen.get("path") or chosen.get("filepath") or chosen.get("full_path")
                if not yaml_file_to_use:
                    print(f"  ✗ Invalid file listing entry: {chosen}")
                    continue
                print(f"  Using most recent saved YAML: {yaml_file_to_use}")
                env_result = await call_tool_data(client, "create_environment_from_file", {
                    "yaml_file_path": yaml_file_to_use
                })
                if not env_result.get("ok", True):
                    print(f"  ✗ Failed to create environment: {env_result}")
                    continue
                sim_id = env_result.get("sim_id")
                print(f"  ✓ Environment created! Sim ID: {sim_id}")
                run_result = await call_tool_data(client, "run_simulation", {
                    "sim_id": sim_id,
                    "steps": SIM_STEPS
                })
                if not run_result.get("ok", True):
                    print(f"  ✗ Simulation failed: {run_result}")
                    continue
                progress = run_result.get("progress", 0)
                print(f"  ✓ Simulation completed! Progress: {progress:.1f}%")
                save_result = await call_tool_data(client, "save_outputs", {
                    "sim_id": sim_id,
                    "include_time_series": True
                })
                if save_result.get("ok", True):
                    print(f"  ✓ Outputs saved to: {save_result.get('output_dir')}")
                else:
                    print("  ✗ Failed to save outputs")
                continue

            print("\n✓ Config is valid.")
            normalized_config = val.get("normalized")

            while True:
                choice_prompt = "\nNext action: [r]un simulation, [s]ave config, [e]dit prompt, [q]uit session? "
                try:
                    choice = await loop.run_in_executor(None, lambda: input(choice_prompt).lower())
                except (EOFError, KeyboardInterrupt):
                    choice = "q"

                if choice == 'r':
                    print("\n--- Running Simulation ---")
                    try:
                        # If we have a saved file, use it; otherwise create a temporary one
                        yaml_file_to_use = None
                        
                        if last_saved_yaml_path and os.path.exists(last_saved_yaml_path):
                            yaml_file_to_use = last_saved_yaml_path
                            print(f"Using saved YAML file: {yaml_file_to_use}")
                        else:
                            # Create temporary file for the run
                            import tempfile
                            base_dir = os.path.dirname(__file__)
                            input_dir = os.path.join(base_dir, "input")
                            os.makedirs(input_dir, exist_ok=True)
                            
                            temp_name = "temp_run_config.yaml"
                            yaml_file_to_use = os.path.join(input_dir, temp_name)
                            
                            with open(yaml_file_to_use, "w", encoding="utf-8") as fh:
                                fh.write(yaml_text)
                            print(f"Created temporary YAML file: {yaml_file_to_use}")
                        
                        # Create environment from file
                        print("Creating environment from YAML file...")
                        env_result = await call_tool_data(client, "create_environment_from_file", {
                            "yaml_file_path": yaml_file_to_use
                        })
                        
                        if not env_result.get("ok", True):
                            print(f"❌ Failed to create environment: {env_result}")
                            continue
                            
                        sim_id = env_result.get("sim_id")
                        print(f"✅ Environment created! Sim ID: {sim_id}")
                        
                        # Run simulation
                        print(f"Running simulation for {SIM_STEPS} steps...")
                        run_result = await call_tool_data(client, "run_simulation", {
                            "sim_id": sim_id,
                            "steps": SIM_STEPS
                        })
                        
                        if not run_result.get("ok", True):
                            print(f"❌ Simulation failed: {run_result}")
                            continue
                            
                        progress = run_result.get("progress", 0)
                        print(f"✅ Simulation completed! Progress: {progress:.1f}%")
                        
                        # Get final status
                        status_result = await call_tool_data(client, "get_status", {"sim_id": sim_id})
                        current_step = status_result.get("current_step", 0)
                        total_steps = status_result.get("total_steps", 0)
                        print(f"Status: {current_step}/{total_steps} steps completed")
                        
                        # Save outputs
                        print("Saving simulation outputs...")
                        save_result = await call_tool_data(client, "save_outputs", {
                            "sim_id": sim_id,
                            "include_time_series": True
                        })
                        
                        if save_result.get("ok", True):
                            output_dir = save_result.get("output_dir")
                            print(f"✅ Outputs saved to: {output_dir}")
                        else:
                            print("❌ Failed to save outputs")
                            
                        # Clean up temporary file if it was created
                        if yaml_file_to_use and yaml_file_to_use.endswith("temp_run_config.yaml") and os.path.exists(yaml_file_to_use):
                            try:
                                os.remove(yaml_file_to_use)
                                print("🧹 Cleaned up temporary file")
                            except Exception:
                                pass  # Ignore cleanup errors
                            
                    except Exception as e:
                        print(f"❌ Error during simulation: {e}")
                        # Clean up temporary file on error too
                        if 'yaml_file_to_use' in locals() and yaml_file_to_use and yaml_file_to_use.endswith("temp_run_config.yaml") and os.path.exists(yaml_file_to_use):
                            try:
                                os.remove(yaml_file_to_use)
                            except Exception:
                                pass
                    
                    # Continue to show menu again
                    continue

                elif choice == 's':
                    try:
                        save_name = await loop.run_in_executor(None, lambda: input("  Enter name for this config (e.g., 'high_demand_test'): "))
                        if save_name:
                            # Save locally to mcp/input/<name>.yaml
                            base_dir = os.path.dirname(__file__)
                            input_dir = os.path.join(base_dir, "input")
                            os.makedirs(input_dir, exist_ok=True)

                            # Sanitize filename and ensure .yaml extension
                            safe_name = re.sub(r"[^A-Za-z0-9_\-]+", "_", save_name).strip("_") or "config"
                            if not safe_name.lower().endswith(".yaml"):
                                safe_name += ".yaml"

                            out_path = os.path.join(input_dir, safe_name)
                            try:
                                with open(out_path, "w", encoding="utf-8") as fh:
                                    fh.write(yaml_text)
                                print(f"  ✓ Saved to {out_path}")
                                last_saved_yaml_path = out_path
                            except Exception as e:
                                print(f"  ✗ Save failed: {e}")
                    except (EOFError, KeyboardInterrupt):
                        pass
                    # After saving, we continue the inner loop to ask for the next action again
                    continue

                elif choice == 'e':
                    # Break inner loop to go back to the user prompt
                    break
                
                elif choice == 'q':
                    break
            
            if choice == 'q':
                break

        print("\nSession ended.")

if __name__ == "__main__":
    asyncio.run(main())



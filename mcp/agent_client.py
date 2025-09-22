#!/usr/bin/env python3
"""
Agent-driven MCP client for PedNStream.
- The LLM decides which MCP tools to call via a structured JSON action.
- This script only orchestrates: it fetches tools, prompts the LLM, dispatches actions,
  returns observations, and enforces safety limits.

Usage:
  export MCP_SERVER_URL=http://127.0.0.1:8000/mcp
  export LLM_PROVIDER=openai   # or anthropic|openai_compat
  export ASSISTANT_MODEL=gpt-4o-mini
  python mcp/agent_client.py
"""
import os
import json
import asyncio
import re
from typing import Any, Dict, List, Optional

from fastmcp import Client

# Reuse simple LLM adapter from assistant_harness
from assistant_harness import llm_get_response, call_tool_data, SERVER_URL

MAX_TURNS = int(os.getenv("AGENT_MAX_TURNS", "15"))
ALLOWED_TOOLS = os.getenv("AGENT_ALLOWED_TOOLS", "").split(",") if os.getenv("AGENT_ALLOWED_TOOLS") else None

# Pseudo-tools handled by this client (not on the server)
CLIENT_PSEUDO_TOOLS = {"client_save_yaml"}

ACTION_SCHEMA = {
    "type": "object",
    "required": ["tool"],
    "properties": {
        "tool": {"type": "string"},
        "args": {"type": "object"},
        "reason": {"type": "string"},
        "stop": {"type": "boolean"}
    }
}

SYSTEM_INSTRUCTIONS = """
You are an autonomous assistant for the PedNStream simulation tool.
Engage in a helpful conversation to achieve the user's high-level goal.

If the user explicitly asks to create a network/config, do the following:
1) Generate a complete YAML configuration in a ```yaml code block, following the exemplar format below.
YAML REQUIREMENTS:
- Create adjacency matrix when building networks (symmetric matrix), the number of 1 in the upper triangular matrix is the number of edges in the network
- Use default_link params unless specific link params are requested
- Include origin_nodes, destination_nodes, demand, and od_flows
2) Then emit a single JSON action in a ```json code block to save the YAML:
   {"tool": "client_save_yaml", "args": {"filename": "<safe_name>.yaml", "yaml_text": "<YAML content as a single string>"}, "reason": "..."}
3) After it's saved, validate it with:
   {"tool": "validate_config_file", "args": {"yaml_file_path": "<full path>"}, "reason": "..."}
4) Only if ok==true, create the environment and run simulations using:
   {"tool": "create_environment_from_file", ...}, then {"tool": "run_simulation", ...}, and {"tool": "save_outputs", ...}

When you need to use a tool, briefly explain your reasoning in plain text, then output only the single JSON action in a ```json block. Do not chain multiple actions in one response.
""".strip()


def summarize_tools(tools: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for t in tools:
        # Support both dicts and Tool objects
        name = getattr(t, "name", None)
        if name is None and isinstance(t, dict):
            name = t.get("name", "")
        desc = getattr(t, "description", None)
        if desc is None and isinstance(t, dict):
            desc = t.get("description", "")
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def get_initial_prompt(tools: List[Dict[str, Any]], example_yaml: str) -> List[Dict[str, str]]:
    tools_summary = summarize_tools(tools)
    exemplar = example_yaml or ""
    return [
        {"role": "system", "content": f"{SYSTEM_INSTRUCTIONS}\n\nAvailable tools:\n{tools_summary}\n\nExemplar YAML format (guide only):\n```yaml\n{exemplar}\n```"},
    ]


def parse_action(text: str) -> Optional[Dict[str, Any]]:
    # Find JSON code block
    match = re.search(r"```json(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if not match:
        # Fallback for raw JSON
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        json_text = text[start:end+1]
    else:
        json_text = match.group(1).strip()

    try:
        obj = json.loads(json_text)
        if not isinstance(obj, dict):
            return None
        return obj
    except Exception:
        return None


def validate_action(obj: Dict[str, Any], available_tool_names: List[str]) -> Optional[str]:
    # stop action
    if obj.get("stop") is True:
        return None
    tool = obj.get("tool")
    if not isinstance(tool, str):
        return "'tool' must be a string or use {'stop': true}"
    # Allow pseudo-tools locally
    if tool in CLIENT_PSEUDO_TOOLS:
        pass
    else:
        if ALLOWED_TOOLS and tool not in ALLOWED_TOOLS:
            return f"tool '{tool}' is not allowed"
        if tool not in available_tool_names:
            return f"unknown tool '{tool}'"
    args = obj.get("args", {})
    if not isinstance(args, dict):
        return "'args' must be an object"
    return None


async def _client_save_yaml(args: Dict[str, Any]) -> Dict[str, Any]:
    """Pseudo-tool: save YAML to mcp/input and return the full path."""
    filename = args.get("filename") or args.get("name") or "config.yaml"
    yaml_text = args.get("yaml_text")
    if not isinstance(yaml_text, str) or len(yaml_text.strip()) == 0:
        return {"ok": False, "error": "yaml_text must be a non-empty string"}

    # sanitize filename
    safe_name = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", filename).strip("_") or "config.yaml"
    if not safe_name.lower().endswith(".yaml"):
        safe_name += ".yaml"

    base_dir = os.path.dirname(__file__)
    input_dir = os.path.join(base_dir, "input")
    os.makedirs(input_dir, exist_ok=True)
    out_path = os.path.join(input_dir, safe_name)

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(yaml_text)
        return {"ok": True, "path": out_path}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def main():
    print(f"Connecting to MCP server at {SERVER_URL} ...")
    async with Client(SERVER_URL) as client:
        await client.ping()
        print("✓ Server reachable, listing tools...")

        # Get the list of available tools
        tool_list = await client.list_tools()
        tool_names = [t.name if hasattr(t, 'name') else t.get('name') for t in tool_list]
        print(f"✓ Available tools: {tool_names}")

        # Fetch exemplar YAML from server schema (same as assistant_harness approach)
        example_yaml = ""
        try:
            schema = await call_tool_data(client, "list_config_schema", {})
            example_yaml = schema.get("example_yaml", "") if isinstance(schema, dict) else ""
        except Exception:
            pass

        # Track which YAML files were validated successfully this session
        validated_files_ok: set = set()

        # Main interactive loop
        history = get_initial_prompt(tool_list, example_yaml)
        print("\n--- PedNStream Interactive Agent ---")
        print("Describe your high-level goal. Type 'quit' or 'exit' to end.")

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

            history.append({"role": "user", "content": user_prompt})
            print("\nGenerating response...")

            response = await llm_get_response("", history=history.copy())
            print("\n--- Assistant ---")
            print(response)
            print("-----------------")

            action = parse_action(response)
            if not action:
                print("\nNo action found in response. Continuing conversation.")
                history.append({"role": "assistant", "content": response})
                continue

            # Action was found, proceed with validation and approval
            err = validate_action(action, tool_names)
            if err:
                print(f"✗ Invalid action: {err}")
                history.append({"role": "assistant", "content": response})
                history.append({"role": "user", "content": f"Invalid action: {err}. Provide a corrected action."})
                continue

            if action.get("stop") is True:
                print(f"✓ Agent requested stop: {action.get('reason', 'done')}")
                history.append({"role": "assistant", "content": response})
                continue

            # Interactive approval step
            tool = action["tool"]
            args = action.get("args", {})
            reason = action.get("reason", "")
            print(f"\nProposed action:\n  tool: {tool}\n  args: {json.dumps(args)}\n  reason: {reason}")
            try:
                user_choice = input("Approve? [y]es / [e]dit / [s]kip / [q]uit: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                user_choice = "y"

            if user_choice == "q":
                print("User quit.")
                break
            if user_choice == "s":
                print("Skipping this action; asking LLM for a new one...")
                history.append({"role": "assistant", "content": response})
                history.append({"role": "user", "content": "Action skipped by user; propose a different next step."})
                continue
            if user_choice == "e":
                try:
                    new_tool = input(f"Tool name [{tool}]: ").strip() or tool
                    new_args_raw = input(f"Args JSON [{json.dumps(args)}]: ").strip() or json.dumps(args)
                    new_args = json.loads(new_args_raw)
                    # Update action with edits
                    action = {"tool": new_tool, "args": new_args, "reason": action.get("reason", "")}
                    # Re-validate edited action
                    err2 = validate_action(action, tool_names)
                    if err2:
                        print(f"✗ Edited action invalid: {err2}")
                        history.append({"role": "assistant", "content": json.dumps(action)})
                        history.append({"role": "user", "content": f"Edited action invalid: {err2}. Provide a corrected action."})
                        continue
                    tool, args = new_tool, new_args
                except Exception as e:
                    print(f"✗ Edit parse error: {e}")
                    continue

            print(f"→ Calling tool: {tool} {args}")

            # Enforce validate->create flow for file-based environments
            if tool == "create_environment_from_file":
                yaml_path = args.get("yaml_file_path") if isinstance(args, dict) else None
                if yaml_path and yaml_path not in validated_files_ok:
                    msg = (
                        "create_environment_from_file requires prior validation. "
                        f"Please call validate_config_file with yaml_file_path='{yaml_path}' first."
                    )
                    print(f"✗ {msg}")
                    # Feed back to the assistant so it proposes the correct next step
                    history.append({"role": "assistant", "content": response})
                    history.append({"role": "user", "content": msg})
                    continue

            try:
                if tool in CLIENT_PSEUDO_TOOLS:
                    result = await _client_save_yaml(args)
                else:
                    result = await call_tool_data(client, tool, args)
                observation = json.dumps(result)
                print(f"← Observation: {observation[:500]}{'...' if len(observation)>500 else ''}")

                # Record successful validation
                if tool == "validate_config_file":
                    yaml_path = args.get("yaml_file_path") if isinstance(args, dict) else None
                    if yaml_path and isinstance(result, dict) and result.get("ok") is True:
                        validated_files_ok.add(yaml_path)

                history.append({"role": "assistant", "content": response}) # original response containing the action
                history.append({"role": "user", "content": f"Observation: {observation}"})
            except Exception as e:
                err_msg = f"Tool call error: {e}"
                print(f"✗ {err_msg}")
                history.append({"role": "assistant", "content": response})
                history.append({"role": "user", "content": err_msg})
                continue
        
        print("\nSession ended.")


if __name__ == "__main__":
    asyncio.run(main())

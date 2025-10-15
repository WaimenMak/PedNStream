export LLM_PROVIDER=openai_compat
export MCP_SERVER_URL=http://127.0.0.1:8000/mcp
export OPENAI_BASE_URL=https://ai.gitee.com/v1
export OPENAI_API_KEY=DXQESZHFRKMNNEEXHNJ3WI6PSPBVHJCT721FHF1V
export OPENAI_DEFAULT_HEADERS='{"X-Failover-Enabled": "true"}'
export ASSISTANT_MODEL=Qwen3-8B # Check provider docs for exact model name
export OPENAI_USE_RESPONSES=0

# run the server
# python mcp_server.py
# run the client (LLM)
python agent_client.py
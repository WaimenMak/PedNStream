## PRD: MCP Config Authoring Tools (sim_params.yaml)

- **Owner**: `mcp/`
- **Server**: `mcp/mcp_server.py`
- **Goal**: Let an LLM (or user) define simulation configs via MCP tools with validation and persistence under `data/<name>/sim_params.yaml`.

### Objectives
- Provide a guardrailed workflow to author configs that `env_loader.py` can consume.
- Normalize legacy keys and enforce a canonical schema (`params`).
- Make configs first-class, versionable artifacts (reproducibility).

### Background
- `src/utils/env_loader.py` expects `self.config['params']` and passes `params` into `Network`.
- Some existing YAMLs (e.g., `data/od_flow_example/sim_params.yaml`) use top-level `simulation/default_link/network`.
- Server flow: `create_environment(config_name, overrides)` → `NetworkEnvGenerator` → `create_network(config_name)` → uses `data/<config_name>/sim_params.yaml`.

### Scope
- In-scope tools:
  - `list_config_schema()`
  - `validate_config(config | yaml_text)`
  - `upsert_config(name, config | yaml_text)`
  - `list_configs()`, `read_config(name)`
- Optional:
  - `create_environment_from_config(config | yaml_text)` (no write, ephemeral)
  - `example_config()` resource with YAML skeleton
- Out of scope: UI/editor, remote storage, schema migration tooling.

### Canonical Config (normalized)
```yaml
params:
  simulation_steps: 500
  unit_time: 10
  assign_flows_type: classic
  path_finder:
    k_paths: 4
    theta: 10
    alpha: 1
    beta: 0.5
    omega: 0.8
  default_link:
    length: 100
    width: 1.0
    free_flow_speed: 1.1
    k_critical: 2
    k_jam: 10
    gamma: 0.01
  links:
    "1_2":
      length: 120
      width: 1.5
origin_nodes: [1]
destination_nodes: [5]
# Optional: adjacency_matrix inline or via files under data/<name>/
# adjacency_matrix: [[0,1,1,0,0,0], ...]
demand:
  origin_1:
    pattern: gaussian_peaks
    peak_lambda: 35
    base_lambda: 5
    seed: 42
od_flows:
  "1_5": 30
```

### Tool API (MCP)
- `list_config_schema() -> { schema: dict, example_yaml: str, notes: str }`
- `validate_config(config?: dict, yaml_text?: str) -> { ok: bool, errors?: [{path, message}], normalized?: dict }`
- `upsert_config(name: str, config?: dict, yaml_text?: str) -> { ok: bool, path: str, wrote_bytes: int, normalized_preview: dict }`
  - Writes `data/<name>/sim_params.yaml`; creates directory if missing.
  - Normalizes legacy keys: 
    - `simulation.simulation_steps -> params.simulation_steps`
    - `default_link -> params.default_link`
    - `links -> params.links`
    - `network.adjacency_matrix -> params.adjacency_matrix` (optional)
- `list_configs() -> { names: [str] }` (from `data/` subdirs with `sim_params.yaml`)
- `read_config(name: str) -> { yaml_text: str }`
- Optional: `create_environment_from_config(config?: dict, yaml_text?: str) -> { sim_id, ... }` (no write)

### Validation Rules (non-exhaustive)
- Required: `params.simulation_steps` (int > 0), `params.unit_time` (int > 0), `params.default_link.*` (types, ranges).
- Optional: `params.links` map of `"u_v"` → partial overrides.
- Optional: `origin_nodes`, `destination_nodes` arrays of ints.
- `od_flows` keys `"o_d"` with numeric values ≥ 0.
- Reject path traversal in `name` (allow `^[a-zA-Z0-9_\-]+$`).
- Limit inline adjacency size (e.g., ≤ 500x500) to prevent oversized payloads.

### File Layout
- Write: `data/<name>/sim_params.yaml`
- Optional additional files allowed (e.g., `adj_matrix.npy`, `edge_distances.pkl`, `node_positions.json`), but not created by these tools.

### Flows
- Authoring:
  1) `list_config_schema()` → client drafts YAML
  2) `validate_config(yaml_text)` → fix until `ok: true`
  3) `upsert_config(name, yaml_text)` → persisted
  4) `create_environment(config_name=name)` → run as usual
- Ephemeral:
  - `create_environment_from_config(yaml_text)` for quick tests without writing.

### Testing
- Unit: normalizer, validator, writer (happy path + edge cases).
- Integration: create config → create environment → run few steps → save outputs.
- Negative: missing fields, invalid types, huge matrices, bad `name`.

### Milestones
- M1: Schema + normalizer + `list_config_schema`, `validate_config` (Day 1–2)
- M2: `upsert_config` with safe write + `list_configs`, `read_config` (Day 2)
- M3: Optional ephemeral create + docs/examples (Day 3)
- M4: Full integration test via `test_http_client.py` (Day 3)

### Risks & Mitigations
- Schema drift vs `env_loader.py`: codify normalization; add tests tied to `NetworkEnvGenerator`.
- Large payloads: enforce size limits.
- Backward compatibility: support legacy keys via normalizer.

### Success Metrics
- ≥ 95% validation pass rate for LLM-generated configs after at most one correction loop.
- Time to first valid run < 1 minute with guided schema + example.
- No runtime failures due to malformed YAML in integration test suite.

### References
- FastMCP Client: https://gofastmcp.com/clients/client
- FastMCP Server: https://gofastmcp.com/servers/server
- Current loader: `src/utils/env_loader.py`
- Server tools: `mcp/mcp_server.py`

### Status Log
- [ ] M1 schema/normalizer/tools scaffolded
- [ ] M2 persistence utilities implemented
- [ ] M3 ephemeral create + docs
- [ ] M4 end-to-end test updated (`mcp/test_http_client.py`)
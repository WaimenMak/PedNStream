import textwrap
from pathlib import Path

import numpy as np
import pytest
import yaml

from pednstream.utils.config import load_config, validate_config
from pednstream.exceptions import RequiredConfigError, InvalidConfigError


def write_yaml(tmp_path, name, content: str) -> Path:
    """Helper to write a small YAML file in a tmp directory."""
    path = tmp_path / name
    path.write_text(textwrap.dedent(content))
    return path


class TestLoadConfig:
    def test_minimal_valid_config_uses_defaults(self, tmp_path):
        """Minimal YAML: check required fields and defaulted options."""
        yaml_content = """
        simulation:
          simulation_steps: 10
          unit_time: 1.0
        network:
          origin_nodes: [0, 1]
        default_link:
          length: 100
          width: 2
          free_flow_speed: 1.1
          k_critical: 2
          k_jam: 6
        """
        cfg_path = write_yaml(tmp_path, "minimal.yaml", yaml_content)

        config = load_config(str(cfg_path))

        params = config["params"]
        assert params["simulation_steps"] == 10
        assert params["unit_time"] == 1.0
        assert params["assign_flows_type"] == "classic"
        assert params["seed"] is None
        assert params["links"] == {}
        assert params["demand"] == {}
        assert params["controllers"] == {}

        assert config["origin_nodes"] == [0, 1]
        assert config["destination_nodes"] == []
        assert config["adjacency_matrix"] is None  # present but None when not in YAML
        assert "od_flows" not in config

    def test_assemble_network_config_with_adjacency_matrix(self):
        """_assemble_network_config should convert adjacency_matrix to np.ndarray."""
        params = {"some": "value"}
        config = {
            "network": {
                "origin_nodes": [0],
                "destination_nodes": [1],
                "adjacency_matrix": [
                    [0, 1],
                    [1, 0],
                ],
            }
        }

        from pednstream.utils.config import _assemble_network_config

        network_config = _assemble_network_config(params, config)

        assert network_config["params"] is params
        assert network_config["origin_nodes"] == [0]
        assert network_config["destination_nodes"] == [1]

        assert "adjacency_matrix" in network_config
        adj = network_config["adjacency_matrix"]
        assert isinstance(adj, np.ndarray)
        assert adj.shape == (2, 2)
        assert np.array_equal(adj, np.array([[0, 1], [1, 0]]))


class TestValidateConfig:
    def test_valid_config_passes(self):
        """Fully specified config should not raise."""
        cfg = {
            "network": {"origin_nodes": [0, 1]},
            "simulation": {"simulation_steps": 10, "unit_time": 1.0},
            "default_link": {
                "length": 10,
                "width": 1,
                "free_flow_speed": 1.0,
                "k_critical": 1,
                "k_jam": 2,
            },
        }

        validate_config(cfg)  # no exception expected

    @pytest.mark.parametrize("missing_section", ["network", "simulation", "default_link"])
    def test_missing_section_raises(self, missing_section):
        """If a whole section is missing, raise RequiredConfigError."""
        cfg = {
            "network": {"origin_nodes": [0]},
            "simulation": {"simulation_steps": 10, "unit_time": 1.0},
            "default_link": {
                "length": 10,
                "width": 1,
                "free_flow_speed": 1.0,
                "k_critical": 1,
                "k_jam": 2,
            },
        }
        cfg.pop(missing_section)

        with pytest.raises(RequiredConfigError) as excinfo:
            validate_config(cfg)

        msg = str(excinfo.value)
        assert f"Missing required section in configuration: {missing_section}" in msg

    def test_validate_config_with_real_delft_sim_params(self):
        """validate_config should accept the real Delft sim_params structure."""
        here = Path(__file__).resolve().parent
        cfg_path = here / "data" / "delft" / "sim_params.yaml"

        with cfg_path.open("r") as f:
            raw_cfg = yaml.safe_load(f)

        validate_config(raw_cfg)

    @pytest.mark.parametrize(
        "section, field",
        [
            ("network", "origin_nodes"),
            ("simulation", "simulation_steps"),
            ("simulation", "unit_time"),
            ("default_link", "length"),
            ("default_link", "width"),
            ("default_link", "free_flow_speed"),
            ("default_link", "k_critical"),
            ("default_link", "k_jam"),
        ],
    )
    def test_missing_required_field_in_any_section_raises(self, section, field):
        base_cfg = {
            "network": {"origin_nodes": [0, 1]},
            "simulation": {"simulation_steps": 10, "unit_time": 1.0},
            "default_link": {
                "length": 10,
                "width": 1,
                "free_flow_speed": 1.0,
                "k_critical": 1,
                "k_jam": 2,
            },
        }

        cfg = {k: v.copy() for k, v in base_cfg.items()}
        cfg[section].pop(field)

        with pytest.raises(RequiredConfigError) as excinfo:
            validate_config(cfg)

        msg = str(excinfo.value)
        assert (
            f"Missing required field in configuration: {field} in section {section}"
            in msg
        )


class TestValidateOdFlows:
    """Tests for od_flows validation in validate_config."""

    @pytest.fixture
    def base_config(self):
        return {
            "network": {
                "origin_nodes": [0, 1, 2],
                "destination_nodes": [3, 4, 5],
            },
            "simulation": {"simulation_steps": 10, "unit_time": 1.0},
            "default_link": {
                "length": 10,
                "width": 1,
                "free_flow_speed": 1.0,
                "k_critical": 1,
                "k_jam": 2,
            },
        }

    def test_valid_od_flows_passes(self, base_config):
        """od_flows with valid origin/destination pairs should pass."""
        cfg = base_config
        cfg["od_flows"] = {
            "0_3": 10,
            "1_4": 20,
            "2_5": 30,
        }

        validate_config(cfg)  # should not raise

    def test_od_flows_without_destination_nodes_raises(self, base_config):
        """od_flows requires destination_nodes to be defined."""
        cfg = base_config
        cfg["network"].pop("destination_nodes")
        cfg["od_flows"] = {"0_3": 10}

        with pytest.raises(InvalidConfigError) as excinfo:
            validate_config(cfg)

        msg = str(excinfo.value)
        assert "destination_nodes" in msg

    def test_od_flows_with_empty_destination_nodes_raises(self, base_config):
        """od_flows requires destination_nodes to be non-empty."""
        cfg = base_config
        cfg["network"]["destination_nodes"] = []
        cfg["od_flows"] = {"0_3": 10}

        with pytest.raises(InvalidConfigError) as excinfo:
            validate_config(cfg)

        msg = str(excinfo.value)
        assert "destination_nodes" in msg

    def test_od_flows_invalid_origin_raises(self, base_config):
        """od_flows key with origin not in origin_nodes should raise."""
        cfg = base_config
        cfg["od_flows"] = {"99_3": 10}  # 99 is not in origin_nodes [0, 1, 2]

        with pytest.raises(InvalidConfigError) as excinfo:
            validate_config(cfg)

        msg = str(excinfo.value)
        assert "origin 99" in msg
        assert "not in origin_nodes" in msg

    def test_od_flows_invalid_destination_raises(self, base_config):
        """od_flows key with destination not in destination_nodes should raise."""
        cfg = base_config
        cfg["od_flows"] = {"0_99": 10}  # 99 is not in destination_nodes [3, 4, 5]

        with pytest.raises(InvalidConfigError) as excinfo:
            validate_config(cfg)

        msg = str(excinfo.value)
        assert "destination 99" in msg
        assert "not in destination_nodes" in msg

    def test_od_flows_malformed_key_raises(self, base_config):
        """od_flows key with bad format should raise."""
        cfg = base_config
        cfg["od_flows"] = {"not_a_valid_key": 10}

        with pytest.raises(InvalidConfigError) as excinfo:
            validate_config(cfg)

        msg = str(excinfo.value)
        assert "Invalid od_flows key format" in msg


class TestValidateDemand:
    """Tests for demand validation in validate_config."""

    @pytest.fixture
    def base_config(self):
        return {
            "network": {
                "origin_nodes": [0, 1, 2],
            },
            "simulation": {"simulation_steps": 10, "unit_time": 1.0},
            "default_link": {
                "length": 10,
                "width": 1,
                "free_flow_speed": 1.0,
                "k_critical": 1,
                "k_jam": 2,
            },
        }

    def test_valid_demand_passes(self, base_config):
        """demand with valid origin references should pass."""
        cfg = base_config
        cfg["demand"] = {
            "origin_0": {"peak_lambda": 10, "base_lambda": 5},
            "origin_1": {"peak_lambda": 20, "base_lambda": 10},
        }

        validate_config(cfg)  # should not raise

    def test_demand_invalid_origin_raises(self, base_config):
        """demand key referencing non-existent origin should raise."""
        cfg = base_config
        cfg["demand"] = {
            "origin_99": {"peak_lambda": 10, "base_lambda": 5},  # 99 not in origin_nodes
        }

        with pytest.raises(InvalidConfigError) as excinfo:
            validate_config(cfg)

        msg = str(excinfo.value)
        assert "origin 99" in msg
        assert "not in origin_nodes" in msg

    def test_demand_malformed_key_raises(self, base_config):
        """demand key with bad format should raise."""
        cfg = base_config
        cfg["demand"] = {
            "bad_key": {"peak_lambda": 10},
        }

        with pytest.raises(InvalidConfigError) as excinfo:
            validate_config(cfg)

        msg = str(excinfo.value)
        assert "Invalid demand key format" in msg

class TestValidateLinks:
    """Tests for links validation in validate_config."""

    @pytest.fixture
    def base_config(self):
        return {
            "network": {
                "origin_nodes": [0],
            },
            "simulation": {"simulation_steps": 10, "unit_time": 1.0},
            "default_link": {
                "length": 10,
                "width": 1,
                "free_flow_speed": 1.0,
                "k_critical": 1,
                "k_jam": 2,
            },
        }

    @pytest.fixture
    def adjacency_matrix(self):
        # 0 -- 1 -- 2 (linear chain)
        return np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    def test_valid_links_passes(self, base_config, adjacency_matrix):
        """links referencing existing edges should pass."""
        cfg = base_config
        cfg["links"] = {
            "0_1": {"length": 50},
            "1_2": {"length": 60},
        }

        validate_config(cfg, adjacency_matrix=adjacency_matrix)  # should not raise

    def test_links_non_existent_edge_raises(self, base_config, adjacency_matrix):
        """links referencing non-existent edge should raise."""
        cfg = base_config
        cfg["links"] = {
            "0_2": {"length": 50},  # no direct edge between 0 and 2
        }

        with pytest.raises(InvalidConfigError) as excinfo:
            validate_config(cfg, adjacency_matrix=adjacency_matrix)

        msg = str(excinfo.value)
        assert "no edge exists" in msg
        assert "0_2" in msg

    def test_links_out_of_bounds_raises(self, base_config, adjacency_matrix):
        """links referencing node index out of bounds should raise."""
        cfg = base_config
        cfg["links"] = {
            "0_99": {"length": 50},  # 99 is out of bounds
        }

        with pytest.raises(InvalidConfigError) as excinfo:
            validate_config(cfg, adjacency_matrix=adjacency_matrix)

        msg = str(excinfo.value)
        assert "out of bounds" in msg

    def test_links_malformed_key_raises(self, base_config, adjacency_matrix):
        """links key with bad format should raise."""
        cfg = base_config
        cfg["links"] = {
            "not_valid": {"length": 50},
        }

        with pytest.raises(InvalidConfigError) as excinfo:
            validate_config(cfg, adjacency_matrix=adjacency_matrix)

        msg = str(excinfo.value)
        assert "Invalid links key format" in msg

    def test_links_without_adjacency_skips_validation(self, base_config):
        """links present but no adjacency_matrix passed should skip validation."""
        cfg = base_config
        cfg["links"] = {
            "0_99": {"length": 50},  # would fail if validated
        }

        validate_config(cfg)  # should not raise (no adjacency passed)
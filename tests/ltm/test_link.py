"""Unit-test for LTM module"""

import pytest
from pednstream.ltm.link import LinkConfig
@pytest.fixture
def link_config():
    link_id = 20
    start_node = 1
    end_node = 2
    simulation_steps = 10
    unit_time = 1
    length =  12
    width = 2
    free_flow_speed = 1.4
    k_critical = 0.5
    k_jam = 5.0
    is_controller = False
    activity_probability = 0.0
    gamma = 2.1e-3 # Defaults to diffusion coefficient
    bi_factor = 1
    fd_type = "yperman"
    speed_noise_std  = 0
    front_gate_width = None
    back_gate_width = None

    return LinkConfig(link_id, start_node, end_node, 
                      simulation_steps, unit_time, 
                       length, width, 
                      free_flow_speed, k_critical, 
                      k_jam, is_controller, activity_probability, 
                      gamma, bi_factor, fd_type, 
                      speed_noise_std, front_gate_width, 
                      back_gate_width)


class TestLinkConfig:
    """Tests for the LinkConfig dataclass"""

    def test_initialization(self, link_config):
        """Test conditional default values in LinkConfig"""

        assert link_config.front_gate_width  == link_config.width
        assert link_config.back_gate_width == link_config.width

    def test_value_error_on_gates_width(self):
        """Test that ValueError is raised when gate widths are not provided and width is None"""

        with pytest.raises(ValueError):
            LinkConfig(
                link_id=21,
                start_node=1,
                end_node=2,
                simulation_steps=10,
                unit_time=1,
                length=12,
                width=None,  # Width is None
                free_flow_speed=1.4,
                k_critical=0.5,
                k_jam=5.0,
                is_controller=False,
                activity_probability=0.0,
                gamma=2.1e-3,
                bi_factor=1,
                fd_type="yperman",
                speed_noise_std=0,
                front_gate_width=None,  # Gate widths not provided
                back_gate_width=None
            )

    def test_value_error_on_k_values(self):
        """Test that ValueError is raised when k_jam is not greater than k_critical"""

        # Create a LinkConfig with k_jam less than or equal to k_critical
        with pytest.raises(ValueError):
            LinkConfig(
                link_id=22,
                start_node=1,
                end_node=2,
                simulation_steps=10,
                unit_time=1,
                length=12,
                width=2,
                free_flow_speed=1.4,
                k_critical=0.5,
                k_jam=0.5,  # k_jam is not greater than k_critical
                is_controller=False,
                activity_probability=0.0,
                gamma=2.1e-3,
                bi_factor=1,
                fd_type="yperman",
                speed_noise_std=0,
                front_gate_width=None,
                back_gate_width=None
            )
  

class TestLink:
    """Test for the Link class"""

    def test_initialization(self, link_config):
        """Test the initialization of the Link class"""
        from pednstream.ltm.link import Separator, Regular, Link
        separator = Separator(link_config)
        regular = Regular(link_config)

        assert isinstance(separator, Link)
        assert isinstance(regular, Link)
        assert separator.back_gate_width is not None
        assert separator.front_gate_width is not None
    
    def test_invalid_reverse_link(self, link_config):
        """Test that ValueError is raised when reserse link is self-referential"""
        from pednstream.ltm.link import Separator
        # revrese_link is a property of any Link
        link = Separator(link_config)
        with pytest.raises(ValueError):
            link.reverse_link = link  # Setting reverse link to itself should raise ValueError
        

    def test_is_controller_true(self, link_config):
        """Test that a Separtor is created when is_controller is True"""
        from pednstream.ltm.link import Separator
        link = Separator(link_config)
        assert link.is_controller  # Should always True for Separator
"""Representation for links in a transportation network."""

import numpy as np
from pednstream.utils.functions import BiDirectionalFd, cal_link_flow_kv
from typing import Any
from numpy import ndarray
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LinkConfig:
    """Configuration dataclass for Link parameters."""

    link_id: int 
    start_node: int  
    end_node : int  
    simulation_steps : int 
    unit_time: int
    length: float
    width: float
    free_flow_speed: float
    k_critical: float
    k_jam: float
    is_controller: bool = False
    activity_probability: float = 0.0
    gamma: float = 2e-3 # Defaults to diffusion coefficient
    bi_factor: float = 1
    fd_type: str = "yperman"
    speed_noise_std: int = 0
    front_gate_width: float | None = None
    back_gate_width: float | None = None

    def __post_init__(self) -> None:
        """Applies conditional defaults."""
        if self.front_gate_width is None:
            self.front_gate_width = self.width    

        if self.back_gate_width is None:
            self.back_gate_width = self.width

        # Makes sure this values are never none after instance creation
        if self.back_gate_width is None or self.front_gate_width is None:
            raise ValueError("Gate widths must be provided or default to link width. " \
            "Please provide valid values for front_gate_width and back_gate_width.")
        
        if self.k_jam - self.k_critical <= 0:
            raise ValueError("k_jam must be greater than k_critical " \
            "to avoid division by zero")
        

class LinkCreator(ABC):

    @abstractmethod
    def create_link(self, config: LinkConfig) -> Link:
        """Creates a network link."""
        pass

class SeparatorCreator(LinkCreator):
    """Creator for the Separator link type."""

    def create_link(self, config) -> Link:
        return Separator(config)
    

class RegularCreator(LinkCreator):
    """Creator for the Regular link type."""

    def create_link(self, config: LinkConfig) -> Link:
        return Regular(config)
    

class Link(ABC):
    """Interface for different Link types."""

    def __init__(self, config: LinkConfig) -> None:
        self.config = config

        self.link_id = config.link_id
        self.start_node = config.start_node
        self.end_node = config.end_node
        self.simulation_steps = config.simulation_steps
        self._is_controller = config.is_controller
        # Dynamic attributes
        self.inflow = np.zeros(self.simulation_steps + 1)  # adjust index
        self.outflow = np.zeros(self.simulation_steps + 1)
        self.cumulative_inflow = np.zeros(self.simulation_steps + 1)
        self.cumulative_outflow = np.zeros(self.simulation_steps + 1)
        self.sending_flow = -1 * np.ones(self.simulation_steps + 1)
        self.receiving_flow = -1 * np.ones(self.simulation_steps + 1)
        self.num_pedestrians = np.zeros(self.simulation_steps + 1, dtype=np.float32)
        self.density = np.zeros(self.simulation_steps + 1, dtype=np.float32)
        self.speed = np.zeros(self.simulation_steps + 1, dtype=np.float32)
        self.link_flow = np.zeros(self.simulation_steps + 1, dtype=np.float32)
        self.gamma = config.gamma  
        self._reverse_link: Link | None = None # TODO: Consider having this on Regular Links
        self.activity_probability = config.activity_probability
        # Physical attributes
        self.length = config.length
        self._width = config.width  # width of link
        self.free_flow_speed = config.free_flow_speed
        self.capacity = self.free_flow_speed * config.k_critical
        self.k_jam = config.k_jam
        self.k_critical = config.k_critical
        self.shockwave_speed = self.capacity / (self.k_jam - self.k_critical)
        self.current_speed = self.free_flow_speed
        self.max_travel_time = (
            self.length / 0.05
        )  # Jam threshold, equivalent to speed of 0.01 m/s
        self._front_gate_width = config.front_gate_width
        self._back_gate_width = config.back_gate_width
        self.back_gate_width_data = self._back_gate_width * np.ones(
            self.simulation_steps + 1
        )
        self.front_gate_width_data = self._front_gate_width * np.ones(
            self.simulation_steps + 1
        )
        self.speed_density_fd = BiDirectionalFd(
            v_f=self.free_flow_speed,
            k_critical=self.k_critical,
            k_jam=self.k_jam,
            bi_factor=config.bi_factor,
            model_type=config.fd_type,
            noise_std=config.speed_noise_std)
        self._exponent = (
            0.8  # private attribute for the releasing factor exponent, default is 1
        )
        self._travel_time_running_sum = self.travel_time[0]
        self.unit_time = config.unit_time
        self.free_flow_tau = round(self.travel_time[0] / self.unit_time)


    @property
    def reverse_link(self) -> Link | None:
        """Gets the reverse link."""
        return self._reverse_link
    
    @reverse_link.setter
    def reverse_link(self, link: Link | None) -> None:
        """Sets the reverse link.
        
        Args:
            link (Link | None): the reverse link to set. If None, unsets the reverse link.
        """
        self._reverse_link = link

        if link == self:
            raise ValueError("A link cannot be its own reverse link.")

    @property
    def is_controller(self) -> bool:
        """If link is a controller."""
        return self._is_controller
    
    @is_controller.setter
    def is_controller(self, value: bool) -> None:
        """Sets the value for is_controller."""
        self._is_controller = value

    @property
    def avg_travel_time_window(self) -> int:
        """Average travel time window."""
        # Using moving average for efficiency
        return round(100 / self.unit_time)

    @property
    def avg_travel_time(self) -> ndarray:
        """Average travel time"""
        avg = np.ones(self.simulation_steps + 1, dtype=np.float32) * self.travel_time[0]
        return avg

    @property
    def travel_time(self) -> ndarray:
        """Travel time."""
        _time = np.zeros(self.simulation_steps + 1, dtype=np.float32)

        _time[0] = min(self.length / self.free_flow_speed, self.max_travel_time)
        return _time

    @property
    def width(self):
        """Width of the link."""
        return self._width

    @property
    def front_gate_width(self):
        """Width of the front gate."""
        return self._back_gate_width

    @front_gate_width.setter
    def front_gate_width(self, value: float):
        self._front_gate_width = value

    @property
    def back_gate_width(self):
        """Width of the back gate."""
        return self._back_gate_width

    @back_gate_width.setter
    def back_gate_width(self, value: float):
        self._back_gate_width = value

    @property
    @abstractmethod
    def area(self) -> float:
        """Area of the link."""
        pass

    def update_cum_outflow(self, q_j: float, time_step: int) -> None:
        """Update commulative outflow for given time step.
        
        Args: 
            q_j (float): flow for outflow at time step
            time_step (int): time step

        Returns:
            None
        """
        PREVIOUS_STEP = time_step - 1
        self.outflow[time_step] = q_j
        self.cumulative_outflow[time_step] = (
            self.cumulative_outflow[PREVIOUS_STEP] + q_j
        )
        return None

    def update_cum_inflow(self, q_i: float, time_step: int) -> None:
        """Update commulative inflow for given time step.
        
        Args:
            q_i (float): flow for inflow at time step
            time_step (int): time step

        Returns:
            None
        """
        self.inflow[time_step] = q_i
        self.cumulative_inflow[time_step] = self.cumulative_inflow[time_step - 1] + q_i
        return None

    def update_density_flow(self, time_step: int) -> None:
        """Updates the link density flow.

        Args:
            time_step (int): current time step

        Returns:
            None
        """
        PREVIOUS_STEP = time_step - 1
        num_peds = self.inflow[time_step] - self.outflow[time_step]
        self.num_pedestrians[time_step] = self.num_pedestrians[PREVIOUS_STEP] + num_peds
        self.density[time_step] = self.num_pedestrians[time_step] / self.area
        return None
    
    def compute_density(self, time_step: int) -> float:
        """Computes the density of the link for the current time step.

        Args:
            time_step (int): time step

        Returns:
            float: density value
        """
        reverse_num_peds = 0
        if self.reverse_link is not None:
            reverse_num_peds = self.reverse_link.num_pedestrians[time_step]
        return (self.num_pedestrians[time_step] + reverse_num_peds) / self.area

    def compute_outflow(self, time_step: int, tau: int) -> int:
        """Computes outflow with diffusion behavior for the current time step.

        Args:
            time_step (int): current time step
            tau (int): tau

        Returns:
            int: outflow value
        """
        PREVIOUS_STEP = time_step - 1
        travel_time = self.avg_travel_time[
            PREVIOUS_STEP
        ]  # use average travel time to calculate

        F = 1 / (1 + self.gamma * travel_time)

        idx = min(max(0, time_step - tau), len(self.inflow) - 1)
        idx1 = max(0, idx - 1)
        idx2 = max(0, idx - 2)
        idx3 = max(0, idx - 3)
        sending_flow = (
            F * self.inflow[idx]
            + F * (1 - F) * self.inflow[idx1]
            + F * (1 - F) ** 2 * self.inflow[idx2]
            + F * (1 - F) ** 3 * self.inflow[idx3]
        )

        return max(np.ceil(sending_flow), 0)


    def calculate_sending_flow(self, time_step: int) -> float:
        """Calculate the sending flow of the link at a given time step.

        Args:
            time_step (int): Current time step

        Returns:
            float: sending flow
        """
        PREVIOUS_STEP = time_step - 1
        # get the total density
        density = self.get_density(PREVIOUS_STEP)

        tau = round(
            self.avg_travel_time[PREVIOUS_STEP] / self.unit_time
        )  # use average travel time to calculate tau

        """ for the initial stage """
        # if time_step - tau < 0:
        if PREVIOUS_STEP <= self.free_flow_tau:
            self.sending_flow[PREVIOUS_STEP] = 0
            return self.sending_flow[PREVIOUS_STEP]

        else:
            # if time_step - tau + 1 < 0: # congestion stage
            """ for the normal stage or the congestion stage """

            idx = min(max(0, PREVIOUS_STEP - tau), len(self.cumulative_inflow) - 1)

            # congestion_factor = np.clip(self.density[time_step] / self.k_jam, 0, 1)
            congestion_factor = np.clip(
                (self.density[PREVIOUS_STEP] - self.k_critical)
                / (self.k_jam - self.k_critical),
                0,
                1,
            )  # this one is theoratically more realistic for unidirectional condition

            boundary_congestion = self.num_pedestrians[PREVIOUS_STEP] / self.area
            boundary_freeflow = max(
                0, self.cumulative_inflow[idx] - self.cumulative_outflow[PREVIOUS_STEP]
            )

            sending_flow_boundary = (
                congestion_factor * boundary_congestion
                + (1 - congestion_factor) * boundary_freeflow
            )

            sending_flow_max = (
                self.front_gate_width
                * self.k_critical
                * self.free_flow_speed
                * self.unit_time
            )
            sending_flow = min(sending_flow_boundary, sending_flow_max)
            # TODO: fix the flow release logic: if sending flow >0, then use diffusion flow

        """ The purpose is to mitigate the maximum sending flow to avoid unrealistic high flow (Added)"""
        # TODO: split this into more manageble and testable pieces.
        original_sending_flow = sending_flow
        if sending_flow > 0:
            releasing_factor = np.clip(density / self.k_jam, 0, 1)
            releasing_prob = 0.7 + (0.85 - 0.7) * releasing_factor**self._exponent

            # free flow stage
            if density <= self.k_critical:
                diffusion_flow = self.compute_outflow(time_step, tau)
                #     # If diffusion flow is active, it represents the arrival of a platoon.
                if diffusion_flow > 0:
                    weight = 0.8
                    sending_flow = int(
                        np.floor(
                            min(
                                weight * diffusion_flow + (1 - weight) * sending_flow,
                                sending_flow,
                            )
                        )
                    )

                else:  # Prevent maximum flow when the inflow is 0, therefore duffusion flow is 0
                    num_passing_peds = int(np.floor(sending_flow))
                    num_leave = np.random.binomial(
                        n=num_passing_peds, p=releasing_prob
                    )  # 70% of the people will leave
                    sending_flow = num_leave
            else:
                num_passing_peds = int(np.floor(sending_flow))
                num_leave = np.random.binomial(
                    n=num_passing_peds, p=releasing_prob
                )  # 70% of the people will leave
                sending_flow = num_leave
            if sending_flow < 0:
                raise ValueError(
                    f"Negative sending flow detected, {sending_flow} at {time_step}"
                )

        """ Stochastic model for pedestrians performing activities (Added)"""
        if self.activity_probability > 0 and sending_flow > 1:
            # Number of pedestrians who could potentially leave
            potential_leavers = int(np.floor(sending_flow))
            # Each pedestrian has an independent probability of staying for an "activity".
            # We use a binomial distribution to find out how many stay.
            num_staying = np.random.binomial(
                n=potential_leavers, p=self.activity_probability
            )
            # Reduce the sending flow by the number of people who decided to stay
            sending_flow -= num_staying

        """ Smooth the sending flow to avoid unrealistic high flow (Added) """
        sending_flow = max(0, sending_flow)
        sending_flow = min(
            np.floor(0.8 * sending_flow + 0.2 * self.sending_flow[PREVIOUS_STEP]),
            original_sending_flow,
        )
        if sending_flow < 0:
            raise ValueError(
                "Negative sending flow detected, sending flow more than original flow"
            )
        self.sending_flow[PREVIOUS_STEP] = sending_flow
        return self.sending_flow[PREVIOUS_STEP]
    
    @abstractmethod
    def get_density(self, time_step: int) -> float:
        """Get density for a time step."""
        pass
    
    @abstractmethod
    def update_speeds(self, time_step: int) -> None:
        """Update the speeds of the link based on the density."""
        pass

    @abstractmethod
    def calculate_receiving_flow(self, time_step: int, with_reverse: bool = False) -> float:
        """Calculate the receiving flow of the link at a given time step.
        
        Args:
            time_step (int): current time step
            with_reverse (bool): whether to consider reverse link interaction. Default is False.
        """
        pass

    @abstractmethod
    def operations(self)-> str:
        pass

# TODO: Double check how computation are using time_step. Check correct logic.
# Adopt the following conventions when naming time step variables:
# - time_step: current time step in the simulation, Used for accessing arrays and updating values for the current time step. 
# - previous_step: time step before the current time step, that is time_step - 1


class Separator(Link):

    def __init__(self, config: LinkConfig) -> None:
        super().__init__(config)

        # Original width must be kept for downstream calculations.
        separator_width = self._width / 2    
        # Modify values specific to separator
        self._separator_width =  separator_width
        self._front_gate_width = separator_width
        self._back_gate_width = separator_width
        # FIXME: Is the link collecting simulations results? Then maybe should be move to component responsible for the results collection.
        self.separator_width_data = separator_width * np.ones(config.simulation_steps + 1)

    @property
    def area(self) -> float:
        """Area of the separator link."""
        return self.length * self.separator_width
    
    @property
    def separator_width(self):
        """Width of the separator."""
        return self._separator_width

    @separator_width.setter
    def separator_width(self, value: float):
        self._separator_width = value

    def get_density(self, time_step: int) -> float:
        """Get density for a time step.

        Args:
            time_step (int): time step

        Returns:
            float: density value
        """
        return self.density[time_step]
    
    def update_speeds(self, time_step: int) -> None:
        """Update the speed of the link based on the density.

        Args: 
            time_step (int): current time step.

        Returns:
            None 
        """
        k_self = self.density[time_step]
        speed = self.speed_density_fd(k_self, 0)
        # Update travel time and speed
        self.speed[time_step] = speed
        self.travel_time[time_step] = (
            self.length / speed if speed > 0 else self.max_travel_time
        )  # avoid infinite travel time

        self.link_flow[time_step] = cal_link_flow_kv(
            self.density[time_step], self.speed[time_step]
        )

        self._travel_time_running_sum += self.travel_time[time_step]
        if time_step >= self.avg_travel_time_window:
            self._travel_time_running_sum -= self.travel_time[
                time_step - self.avg_travel_time_window
            ]
            self.avg_travel_time[time_step] = (
                self._travel_time_running_sum / self.avg_travel_time_window
            )

        self.separator_width_data[time_step] = (
            self._separator_width
        )  # for visualization of the separator width
        self.back_gate_width_data[time_step] = (
            self._separator_width
        )  # for visualization of the back gate width


    def calculate_receiving_flow(self, time_step: int, with_reverse: bool = False) -> float:
        """Calculate the receiving flow of the link at a given time step.

        Args:
            time_step (int): current time step
            with_reverse (bool): whether to consider reverse link interaction. Default is False.
        
        Returns:
            float: recieving flow
        """
        PREVIOUS_STEP = time_step - 1

        # TODO: is using length the correct way to calculate receiving flow?
        tau_shockwave = round(self.length / (self.shockwave_speed * self.unit_time))


        if time_step - tau_shockwave < 0:
            receiving_flow_boundary = self.k_jam * self.area

        else:
            receiving_flow_boundary = (
                self.cumulative_outflow[time_step - tau_shockwave]
                + self.k_jam * self.area
                - self.cumulative_inflow[PREVIOUS_STEP] 
            )

        receiving_flow_max = (
            self.back_gate_width
            * self.k_critical
            * self.free_flow_speed
            * self.unit_time
        )
        receiving_flow = min(receiving_flow_boundary, receiving_flow_max)
        if receiving_flow < 0:
            print(f"Negative receiving flow detected, {receiving_flow} at {time_step}")
        receiving_flow = max(receiving_flow, 0)

        """smooth the receiving flow (Optional when we use the mitigation method)"""
        if self.receiving_flow[PREVIOUS_STEP] >= 0:
            receiving_flow = min(
                np.floor(
                    receiving_flow * 0.8 + self.receiving_flow[PREVIOUS_STEP] * 0.2
                ),
                receiving_flow,
            )

        if with_reverse:
            # For separator, we do not consider the reverse sending flow
            return max(receiving_flow, 0)
        else:
            return receiving_flow

    def operations(self):
        return "a separtor link"
    

class Regular(Link):
    """Represents a regular link in a transportation network."""

    @property
    def area(self) -> float:
        """Area of the separator link."""
        return self.length * self.width
    
    def get_density(self, time_step: int):
        """Get the density of the link.
        
        Args:
            time_step (int): time step

        Returns:
            float: density value
        """
        reverse_num_peds = 0
        if self.reverse_link is not None:
            reverse_num_peds = self.reverse_link.num_pedestrians[time_step]
        return (self.num_pedestrians[time_step] + reverse_num_peds) / self.area

    def update_link_density_flow(self, time_step: int):
        
        num_peds = self.inflow[time_step] - self.outflow[time_step]
        self.num_pedestrians[time_step] = self.num_pedestrians[time_step - 1] + num_peds
        self.density[time_step] = self.num_pedestrians[time_step] / self.area

    def update_speeds(self, time_step: int) -> None:
        """
        Update the speed of the link based on the density

        Args: 
            time_step (int): future time, that is current time step + 1.

        Returns:
            None 
        """
        k_self = self.density[time_step]
        k_opp = 0
        if self.reverse_link:
            k_opp = self.reverse_link.density[time_step]

        speed = self.speed_density_fd(
            k_self, k_opp
        )  # speed for actual link speed derived from FD, speed_eff for the sending flow calculation
        # Update travel time and speed
        self.speed[time_step] = speed
        self.travel_time[time_step] = (
            self.length / speed if speed > 0 else self.max_travel_time
        )  # avoid infinite travel time
        self.link_flow[time_step] = cal_link_flow_kv(
            self.density[time_step], self.speed[time_step]
        )

        self._travel_time_running_sum += self.travel_time[time_step]
        if time_step >= self.avg_travel_time_window:
            self._travel_time_running_sum -= self.travel_time[
                time_step - self.avg_travel_time_window
            ]
            self.avg_travel_time[time_step] = (
                self._travel_time_running_sum / self.avg_travel_time_window
            )
        # update the back_gate_width_data
        self.back_gate_width_data[time_step] = self.back_gate_width
        self.front_gate_width_data[time_step] = self.front_gate_width

    def calculate_receiving_flow(self, time_step: int, with_reverse: bool = False) -> float:
        """Calculate the receiving flow of the link at a given time step.

        Args:
            time_step (int): current time step
            with_reverse (bool): whether to consider reverse link interaction. Default is False.

        Returns:
            float: recieving flow
        """
        PREVIOUS_STEP = time_step - 1
        # TODO: is using length the correct way to calculate receiving flow?
        tau_shockwave = round(self.length / (self.shockwave_speed * self.unit_time))
        reverse_peds = self.reverse_link.num_pedestrians[PREVIOUS_STEP]
        reverse_peds_rand = np.random.binomial(n=reverse_peds, p=0.9)

        if time_step - tau_shockwave < 0:
            receiving_flow_boundary = self.k_jam * self.area - reverse_peds_rand
        else:
            receiving_flow_boundary = max(
                0,
                self.cumulative_outflow[time_step - tau_shockwave]
                + self.k_jam * self.area
                - reverse_peds_rand
                - self.cumulative_inflow[PREVIOUS_STEP],
            )

        receiving_flow_max = (
            self.back_gate_width
            * self.k_critical
            * self.free_flow_speed
            * self.unit_time
        )
        receiving_flow = min(receiving_flow_boundary, receiving_flow_max)
        if receiving_flow < 0:
            print(f"Negative receiving flow detected, {receiving_flow} at {time_step}")
        receiving_flow = max(receiving_flow, 0)

        """smooth the receiving flow (Optional when we use the mitigation method)"""
        if self.receiving_flow[PREVIOUS_STEP] >= 0:
            receiving_flow = min(
                np.floor(
                    receiving_flow * 0.8 + self.receiving_flow[PREVIOUS_STEP] * 0.2
                ),
                receiving_flow,
            )

        if with_reverse:
            reverse_sending_flow = self.reverse_link.sending_flow[PREVIOUS_STEP].copy()

            receiving_flow = receiving_flow - reverse_sending_flow
            return max(receiving_flow, 0)
        else:
            return receiving_flow


    def operations(self):
        return "a regular link"
    

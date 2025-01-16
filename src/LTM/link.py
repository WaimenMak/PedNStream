import numpy as np
from src.utils.functions import cal_travel_speed, cal_free_flow_speed

class BaseLink:
    """Base class for all link types"""
    def __init__(self, link_id, start_node, end_node, simulation_steps):
        self.link_id = link_id
        self.start_node = start_node
        self.end_node = end_node
        
        # Common dynamic attributes
        self.inflow = np.zeros(simulation_steps)
        self.outflow = np.zeros(simulation_steps)
        self.cumulative_inflow = np.zeros(simulation_steps)
        self.cumulative_outflow = np.zeros(simulation_steps)

    def update_cum_outflow(self, q_j: float, time_step: int):
        self.outflow[time_step] = q_j
        self.cumulative_outflow[time_step] = self.cumulative_outflow[time_step - 1] + q_j

    def update_cum_inflow(self, q_i: float, time_step: int):
        self.inflow[time_step] = q_i
        self.cumulative_inflow[time_step] = self.cumulative_inflow[time_step - 1] + q_i

    def update_speeds(self, time_step: int):
        return

class Link(BaseLink):
    """Physical link with full traffic dynamics"""
    def __init__(self, link_id, start_node, end_node, length, width, 
                 free_flow_speed, k_critical, k_jam, unit_time, simulation_steps):
        super().__init__(link_id, start_node, end_node, simulation_steps)
        
        # Physical attributes
        self.length = length
        self.width = width
        self.area = length * width
        self.free_flow_speed = free_flow_speed
        self.capacity = self.free_flow_speed * k_critical
        self.k_jam = k_jam
        self.k_critical = k_critical
        self.shockwave_speed = self.capacity / (self.k_jam - self.k_critical)
        self.current_speed = free_flow_speed
        self.travel_time = length / free_flow_speed
        self.unit_time = unit_time
        self.free_flow_tau = round(self.travel_time / unit_time)

        # Additional dynamic attributes
        self.num_pedestrians = np.zeros(simulation_steps)
        self.density = np.zeros(simulation_steps)
        self.speed = np.zeros(simulation_steps)
        self.outflow_buffer = dict()
        self.gamma = 2e-2
        self.receiving_flow = []
        self.reverse_link = None

    def update_density(self, time_step: int):
        num_peds = self.inflow[time_step] - self.outflow[time_step]
        self.num_pedestrians[time_step] = self.num_pedestrians[time_step - 1] + num_peds
        self.density[time_step] = self.num_pedestrians[time_step] / self.area

    def update_speeds(self, time_step: int):
        """
        Update the speed of the link based on the density
        :param time_step:
        :return:
        """

        # Get reverse link density
        reverse_density = 0
        if self.reverse_link is not None:
            reverse_density = self.reverse_link.density[time_step]

        # Calculate new speed using density ratio formula
        density = self.density[time_step] + reverse_density
        speed = cal_travel_speed(density, self.free_flow_speed, k_critical=self.k_critical, k_jam=self.k_jam)
        # speed = cal_free_flow_speed(
        #     density_i=self.density[time_step],
        #     density_j=reverse_density,
        #     v_f=self.free_flow_speed
        # )

        # Update travel time and speed
        self.speed[time_step] = speed
        self.travel_time = self.length / speed if speed > 0 else float('inf')

    def cal_sending_flow(self, time_step: int) -> float:
        if self.travel_time == float('inf'):
            return 0

        tau = round(self.travel_time / self.unit_time)
        if time_step - tau < 0:
            sending_flow = 0
        else:
            sending_flow_boundary = self.cumulative_inflow[time_step - tau] - self.cumulative_outflow[time_step - 1]
            sending_flow_max = self.k_critical * self.free_flow_speed * self.unit_time
            sending_flow = min(sending_flow_boundary, sending_flow_max)

        return max(0, sending_flow)

    def cal_receiving_flow(self, time_step: int) -> float:
        if time_step - round(self.length/(self.shockwave_speed * self.unit_time)) < 0:
            receiving_flow_boundary = self.k_jam * self.length
        else:
            receiving_flow_boundary = (self.cumulative_outflow[time_step - round(self.length/(self.shockwave_speed * self.unit_time))]
                              + self.k_jam * self.length - self.cumulative_inflow[time_step - 1])

        receiving_flow_max = self.k_critical * self.free_flow_speed * self.unit_time
        receiving_flow = min(receiving_flow_boundary, receiving_flow_max)
        # if receiving_flow < 30:
        #     print(receiving_flow, receiving_flow_boundary, receiving_flow_max, time_step)

        self.receiving_flow.append(receiving_flow)
        return receiving_flow
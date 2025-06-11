import numpy as np
from src.utils.functions import cal_travel_speed, cal_free_flow_speed, cal_link_flow_fd, cal_link_flow_kv

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
        self.sending_flow = 0
        self.receiving_flow = 0

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
    def __init__(self, link_id, start_node, end_node, simulation_steps, unit_time, **kwargs):
        """
        Initialize a Link with parameters from kwargs
        :param link_id: ID of the link
        :param start_node: Starting node
        :param end_node: Ending node
        :param kwargs: Keyword arguments including:
            - length: Length of the link
            - width: Width of the link
            - free_flow_speed: Free flow speed
            - k_critical: Critical density
            - k_jam: Jam density
            - unit_time: Time step size
            - simulation_steps: Number of simulation steps
            - gamma: Optional, default 2e-2
        """
        super().__init__(link_id, start_node, end_node, simulation_steps)
        
        # Physical attributes
        self.length = kwargs['length']
        self.width = kwargs['width']
        self.area = self.length * self.width
        self.free_flow_speed = kwargs['free_flow_speed']
        self.capacity = self.free_flow_speed * kwargs['k_critical']
        self.k_jam = kwargs['k_jam']
        self.k_critical = kwargs['k_critical']
        self.shockwave_speed = self.capacity / (self.k_jam - self.k_critical)
        self.current_speed = self.free_flow_speed
        self.travel_time = self.length / self.free_flow_speed
        self.unit_time = unit_time
        self.free_flow_tau = round(self.travel_time / self.unit_time)
        # self.congestion_tau = round(self.length / (0.8 * self.unit_time)) # assume the average congestion speed is 0.8 m/s

        # Additional dynamic attributes
        self.num_pedestrians = np.zeros(simulation_steps)
        self.density = np.zeros(simulation_steps)
        self.speed = np.zeros(simulation_steps)
        self.link_flow = np.zeros(simulation_steps)
        self.gamma = kwargs.get('gamma', 2e-3)  # Default value if not provided, diffusion coefficient
        # self.receiving_flow = []
        self.reverse_link = None

    def update_link_density_flow(self, time_step: int):
        num_peds = self.inflow[time_step] - self.outflow[time_step]
        self.num_pedestrians[time_step] = self.num_pedestrians[time_step - 1] + num_peds
        self.density[time_step] = self.num_pedestrians[time_step] / self.area
        # self.link_flow[time_step] = cal_link_flow_fd(self.density[time_step], self.free_flow_speed,
        #                                            self.k_jam, self.k_critical, self.shockwave_speed)
        # self.link_flow[time_step] = cal_link_flow_kv(self.density[time_step], self.speed[time_step])

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
        self.link_flow[time_step] = cal_link_flow_kv(self.density[time_step], self.speed[time_step])

    def get_outflow(self, time_step: int, tau: int) -> int:
        """
        Get outflow with diffusion behavior
        """
        # tau = self.congestion_tau if self.speed[time_step] < self.free_flow_speed else self.free_flow_tau
        # tau = self.congestion_tau
        travel_time = self.length / self.speed[time_step-1]
        F = 1/ (1 + self.gamma * travel_time)
        sending_flow = (F * self.inflow[time_step - tau] + F * (1 - F) * self.inflow[time_step - tau - 1] +
                        F * (1 - F) ** 2 * self.inflow[time_step - tau - 2] +
                        F * (1 - F) ** 3 * self.inflow[time_step - tau - 3])
        return np.ceil(sending_flow)

    def cal_sending_flow(self, time_step: int) -> float:
        # if self.link_id == '2_3':
        #     print(self.density[time_step-1])
        if self.travel_time == float('inf'):
            if self.density[time_step - 1] > self.k_critical: # for fully jam stage
                self.sending_flow = np.random.randint(0, 10)
            else:
                self.sending_flow = 0
            return self.sending_flow

        tau = round(self.travel_time / self.unit_time)
        # if time_step - tau < 0:
        if time_step < self.free_flow_tau:  # for the initial stage
            self.sending_flow = 0
            return self.sending_flow
        elif time_step - tau < 0:
            # for the highly congested stage
            # if self.density[time_step - 1] > self.k_critical:
            if self.density[time_step - 1] > self.k_jam - 2:
                self.sending_flow = max(0, np.floor(self.link_flow[time_step - 1] * self.unit_time)) # ensure when congested, the number of peds going through is at least 1
                # if self.link_id == '1_0':
                #     print(self.num_pedestrians[time_step-1])
            else:
                self.sending_flow = 0

            return self.sending_flow
        else:                  # for the normal stage and the congestion stage
            sending_flow_boundary = self.cumulative_inflow[time_step - tau] - self.cumulative_outflow[time_step - 1]
            sending_flow_max = self.k_critical * self.free_flow_speed * self.unit_time
            self.sending_flow = min(sending_flow_boundary, sending_flow_max)
            # TODO: add diffusion flow to the sending flow
            # TODO: fix the flow release logic
            # if (self.sending_flow < 0) and (self.speed[time_step - 1] < 0.2):
            if (self.sending_flow < 0) and (self.density[time_step - 1] > 4.5):
                # it means the link is a bit congested, 0.2 m/s is a threshold
                self.sending_flow = max(0, np.floor(self.link_flow[time_step - 1] * self.unit_time))
                # if self.link_id == '6_7':
                #     print(self.link_flow[time_step - 1], self.density[time_step - 1], self.speed[time_step - 1])
            # elif self.sending_flow > 0 and self.speed[time_step - 1] > 1.2:
            elif self.sending_flow > 0 and self.inflow[time_step - 1] > 0 and self.speed[time_step - 1] >= self.free_flow_speed:
                # if the sending flow is positive, then use diffusion flow. outcome: the flow is propagated more slowly
                self.sending_flow = min(self.get_outflow(time_step, tau), self.sending_flow)
        return max(0, self.sending_flow)

    def cal_receiving_flow(self, time_step: int) -> float:
        if time_step - round(self.length/(self.shockwave_speed * self.unit_time)) < 0:
            receiving_flow_boundary = self.k_jam * self.length
        else:
            receiving_flow_boundary = (self.cumulative_outflow[time_step - round(self.length/(self.shockwave_speed * self.unit_time))]
                              + self.k_jam * self.length - self.cumulative_inflow[time_step - 1])

        receiving_flow_max = self.k_critical * self.free_flow_speed * self.unit_time
        self.receiving_flow = min(receiving_flow_boundary, receiving_flow_max)
        # if receiving_flow < 30:
        #     print(receiving_flow, receiving_flow_boundary, receiving_flow_max, time_step)

        return self.receiving_flow
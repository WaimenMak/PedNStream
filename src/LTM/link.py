import numpy as np
from src.utils.functions import SpeedDensityFd, cal_free_flow_speed, cal_link_flow_fd, cal_link_flow_kv

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
        self.sending_flow = -1 * np.ones(simulation_steps)
        self.receiving_flow = -1 * np.ones(simulation_steps)

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
        self._width = kwargs['width']  # width of the link
        self._front_gate_width = self.width  # width of the gate, for gate control in the head
        self._back_gate_width = self.width  # width of the gate, for gate control in the tail
        self.free_flow_speed = kwargs['free_flow_speed']
        self.capacity = self.free_flow_speed * kwargs['k_critical']
        self.k_jam = kwargs['k_jam']
        self.k_critical = kwargs['k_critical']
        self.shockwave_speed = self.capacity / (self.k_jam - self.k_critical)
        self.current_speed = self.free_flow_speed
        self.max_travel_time = self.length / 0.01  # Jam threshold, equivalent to speed of 0.01 m/s
        self.speed_density_fd = SpeedDensityFd(
            v_f=self.free_flow_speed,
            k_critical=self.k_critical,
            k_jam=self.k_jam,
            model_type=kwargs.get('fd_type', 'greenshields'),
            noise_std=kwargs.get('speed_noise_std', 0)
        )

        self.travel_time = np.zeros(simulation_steps, dtype=np.float32)
        self.travel_time[0] = min(self.length / self.free_flow_speed, self.max_travel_time)
        self._travel_time_running_sum = self.travel_time[0]
        self.unit_time = unit_time
        self.free_flow_tau = round(self.travel_time[0] / self.unit_time)

        # For efficient moving average calculation
        self.avg_travel_time_window = round(40 / self.unit_time)
        self.avg_travel_time = np.zeros(simulation_steps, dtype=np.float32)
        self.avg_travel_time[:self.avg_travel_time_window] = self.travel_time[0] # initialize the first window

        # Additional dynamic attributes
        self.num_pedestrians = np.zeros(simulation_steps, dtype=np.float32)
        self.density = np.zeros(simulation_steps, dtype=np.float32) 
        self.speed = np.zeros(simulation_steps, dtype=np.float32)
        self.link_flow = np.zeros(simulation_steps, dtype=np.float32)
        self.gamma = kwargs.get('gamma', 2e-3)  # Default value if not provided, diffusion coefficient
        self.reverse_link = None
        self.activity_probability = kwargs.get('activity_probability', 0.0)

    @property
    def width(self):
        return self._width

    @property
    def front_gate_width(self):
        return self._front_gate_width

    @front_gate_width.setter
    def front_gate_width(self, value: float):
        self._front_gate_width = value
        if self.reverse_link:
            # Set the private attribute on the reverse link to avoid recursion
            self.reverse_link._back_gate_width = value

    @property
    def back_gate_width(self):
        return self._back_gate_width

    @back_gate_width.setter
    def back_gate_width(self, value: float):
        self._back_gate_width = value
        if self.reverse_link:
            # Set the private attribute on the reverse link to avoid recursion
            self.reverse_link._front_gate_width = value

    @property
    def area(self):
        """Calculates the area of the link dynamically based on its width and length."""
        return self.length * self.width

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
        :param time_step: current time step + 1, is the future time step
        :return:
        """
        # if time_step > 80 and self.link_id == '2_3':
        #     pass
        # Get reverse link density
        # reverse_num_peds = 0
        # if self.reverse_link is not None:
        #     reverse_num_peds = self.reverse_link.num_pedestrians[time_step]
            # reverse_area = self.reverse_link.area

        # Calculate new speed using density ratio formula
        # density = (self.num_pedestrians[time_step] + reverse_num_peds) / self.area  # this link area is the same as the reverse link, they share the same area
        # speed = cal_travel_speed(density, self.free_flow_speed, k_critical=self.k_critical, k_jam=self.k_jam)
        density = self.get_density(time_step) # density and num_pedestrians of all incoming and outgoing links are already updated before updating the speed
        speed = self.speed_density_fd(density)
        # speed = cal_free_flow_speed(
        #     density_i=self.density[time_step],
        #     density_j=reverse_density,
        #     v_f=self.free_flow_speed
        # )

        # Update travel time and speed
        self.speed[time_step] = speed
        self.travel_time[time_step] = self.length / speed if speed > 0 else self.max_travel_time
        # self.travel_time[time_step] = self.length / speed if speed > 0 else float('inf')

        self.link_flow[time_step] = cal_link_flow_kv(self.density[time_step], self.speed[time_step])

        self._travel_time_running_sum += self.travel_time[time_step]
        if time_step >= self.avg_travel_time_window:
            self._travel_time_running_sum -= self.travel_time[time_step - self.avg_travel_time_window]
            self.avg_travel_time[time_step] = self._travel_time_running_sum / self.avg_travel_time_window

    def get_density(self, time_step: int):
        """
        Get the density of the link
        """
        reverse_num_peds = 0
        if self.reverse_link is not None:
            reverse_num_peds = self.reverse_link.num_pedestrians[time_step]
        return (self.num_pedestrians[time_step] + reverse_num_peds) / self.area

    def get_outflow(self, time_step: int, tau: int) -> int:
        """
        Get outflow with diffusion behavior
        """
        # tau = self.congestion_tau if self.speed[time_step] < self.free_flow_speed else self.free_flow_tau
        # tau = self.congestion_tau
        travel_time = self.avg_travel_time[time_step] # use average travel time to calculate
        # travel_time = self.travel_time[time_step] # use real-time travel time to calculate
        F = 1 / (1 + self.gamma * travel_time)
        sending_flow = (F * self.inflow[time_step - tau] + F * (1 - F) * self.inflow[time_step - tau - 1] +
                        F * (1 - F) ** 2 * self.inflow[time_step - tau - 2] +
                        F * (1 - F) ** 3 * self.inflow[time_step - tau - 3])

        return max(np.ceil(sending_flow), 0)

    def cal_sending_flow(self, time_step: int) -> float:
        """
        Calculate the sending flow of the link at a given time step
        :param time_step: Current time step (t - 1)
        """
        # if time_step == 100 and self.link_id == '2_3':
        #     pass

        # get the total density
        density = self.get_density(time_step)
        # reverse_num_peds = 0
        # if self.reverse_link is not None:
        #     reverse_num_peds = self.reverse_link.num_pedestrians[time_step]
        # density = (self.num_pedestrians[time_step] + reverse_num_peds) / self.area

        ''' for the jam stage '''
        # if self.travel_time[time_step] == float('inf'): # speed is 0
        if self.travel_time[time_step] >= self.max_travel_time:
            # if self.density[time_step - 1] > self.k_critical: # for fully jam stage and link flow is 0
            # if density > self.k_critical:
            # self.sending_flow = np.random.randint(0, 10)
            # extrusion people using binomial distribution
            # if self.density[time_step] > 1:
            #     sending_flow_boundary = self.num_pedestrians[time_step]
            # else:
            #     sending_flow_boundary = self.link_flow[time_step] * self.unit_time * self.front_gate_width  # use link flow to calculate sending flow, this is the case when the link is not congested
            sending_flow_boundary = self.num_pedestrians[time_step]
            sending_flow_max = self.front_gate_width * self.k_critical * self.free_flow_speed * self.unit_time
            sending_flow = min(sending_flow_boundary, sending_flow_max)
            num_peds = int(np.floor(sending_flow))
            releasing_factor = np.clip(self.density[time_step] / self.k_jam, 0, 1)
            releasing_prob = 0 + (0.8 - 0) * releasing_factor ** 0.5
            # if self.link_id == '2_3':
            #     print(releasing_prob)
            sending_flow = np.random.binomial(n=num_peds, p=releasing_prob) # 10% of the people will leave
            # sending_flow = num_peds - num_stay
            # self.sending_flow[time_step] = num_peds - num_stay
            # self.sending_flow[time_step] = num_peds
            # else:
            #     self.sending_flow = 0
            self.sending_flow[time_step] = min(np.floor(0.8 * sending_flow + 0.2 * self.sending_flow[time_step - 1]), sending_flow)
            return self.sending_flow[time_step]

        tau = round(self.avg_travel_time[time_step] / self.unit_time) # use average travel time to calculate tau
        # tau = round(self.travel_time[time_step] / self.unit_time)  # use real-time travel time to calculate tau

        ''' for the initial stage '''
        # if time_step - tau < 0:
        if time_step < self.free_flow_tau:
            self.sending_flow[time_step] = 0
            return self.sending_flow[time_step]

        else:   
            # if time_step - tau + 1 < 0: # congestion stage
            ''' for the normal stage or the congestion stage '''
            idx = max(0, time_step + 1 - tau)
            # sending_flow_boundary = max(0, self.cumulative_inflow[idx] - self.cumulative_outflow[time_step])
            # if density > self.k_critical and self.density[time_step] > 3:  # congestion stage
            #     sending_flow_boundary = self.num_pedestrians[time_step]
            # else:  # free flow stage
            #     sending_flow_boundary = max(0, self.cumulative_inflow[idx] - self.cumulative_outflow[time_step]) # +1 is the delta t

            congestion_factor = np.clip((density - self.k_critical) / (self.k_jam - self.k_critical), 0, 1)

            boundary_congestion = self.num_pedestrians[time_step]
            boundary_freeflow = max(0, self.cumulative_inflow[idx] - self.cumulative_outflow[time_step])

            sending_flow_boundary = (congestion_factor * boundary_congestion +
                                     (1 - congestion_factor) * boundary_freeflow)

            ''' original method from the paper '''
            # sending_flow_boundary = max(0, self.cumulative_inflow[time_step + 1 - self.free_flow_tau] - self.cumulative_outflow[time_step]) # +1 is the delta t

            # sending_flow_boundary = self.num_pedestrians[time_step]
            # sending_flow_max = self.k_critical * self.free_flow_speed * self.unit_time
            sending_flow_max = self.front_gate_width * self.k_critical * self.free_flow_speed * self.unit_time
            sending_flow = min(sending_flow_boundary, sending_flow_max)
            # TODO: fix the flow release logic: if sending flow >0, then use diffusion flow
            # if (self.sending_flow < 0) and (self.speed[time_step - 1] < 0.2):
            # if (self.sending_flow < 0) and (self.density[time_step - 1] > 4.5):
            # if self.sending_flow < 0:
                # it means the link is a bit congested, 0.2 m/s is a threshold
                # self.sending_flow = max(0, np.floor(self.link_flow[time_step] * self.unit_time))
                # if self.link_id == '6_7':
                #     print(self.link_flow[time_step - 1], self.density[time_step - 1], self.speed[time_step - 1])
            # elif self.sending_flow > 0 and self.speed[time_step - 1] > 1.2:
            # elif self.sending_flow > 0 and self.inflow[time_step - 1] > 0 and self.speed[time_step - 1] >= self.free_flow_speed:
        
        ''' The purpose is to mitigate the maximum sending flow to avoid unrealistic high flow (Added)'''
        original_sending_flow = sending_flow
        if sending_flow > 0:
            releasing_factor = np.clip((density - self.k_critical) / (self.k_jam - self.k_critical), 0, 1)
            releasing_prob = 0.7 + (0.9 - 0.7) * releasing_factor ** 0.5
            # if the sending flow is positive, then use diffusion flow. outcome: the flow is propagated more slowly
            # last_flow = self.sending_flow[time_step - 1] if time_step - 1 >= 0 else 0
            # free flow stage
            if density <= self.k_critical:
                diffusion_flow = self.get_outflow(time_step, tau)
            #     # If diffusion flow is active, it represents the arrival of a platoon.
                if diffusion_flow > 0:
                    # sending_flow = min(diffusion_flow, sending_flow) # this min is necessary because there might be inflow but actual num peds is less than it.
                    # weight = np.clip( density / self.k_critical, 0.5, 0.8)
                    weight = 0.8
                    sending_flow = int(np.floor(min(weight * diffusion_flow + (1 - weight) * sending_flow, sending_flow)))
            #         # self.sending_flow[time_step] = np.floor(0.2 * diffusion_flow + 0.8 * self.sending_flow[time_step])
                else: # Prevent maximum flow when the inflow is 0, therefore duffusion flow is 0
            #         # smooth current sending flow with the sending flow of the previous time step
            #         # smooth_factor = density / self.k_critical
            #         # smooth_factor = 0.5
                    num_passing_peds = int(np.floor(sending_flow))
                    num_leave = np.random.binomial(n=num_passing_peds, p=releasing_prob)  # 70% of the people will leave
                    sending_flow = num_leave
            #         # self.sending_flow[time_step] = smooth_factor * num_leave + (1 - smooth_factor) * last_flow
            else:
            #     # smooth_factor = np.clip((density - self.k_critical) / (self.k_jam - self.k_critical), 0.5, 1) # smooth factor is between 0.5 and 1
                num_passing_peds = int(np.floor(sending_flow))
                num_leave = np.random.binomial(n=num_passing_peds, p=releasing_prob) # 70% of the people will leave
                sending_flow = num_leave
            if sending_flow < 0:
                raise ValueError(f"Negative sending flow detected, {sending_flow} at {time_step}")



        ''' Stochastic model for pedestrians performing activities (Added)'''
        if self.activity_probability > 0 and sending_flow > 1:
            # Number of pedestrians who could potentially leave
            potential_leavers = int(np.floor(sending_flow))
            # Each pedestrian has an independent probability of staying for an "activity".
            # We use a binomial distribution to find out how many stay.
            num_staying = np.random.binomial(n=potential_leavers, p=self.activity_probability)
            # Reduce the sending flow by the number of people who decided to stay
            sending_flow -= num_staying

        # if self.link_id == '1_3' and self.density[time_step] > self.k_critical and sending_flow == 0:
        #     pass
        ''' Smooth the sending flow to avoid unrealistic high flow (Added) '''
        sending_flow = max(0, sending_flow)
        sending_flow = min(np.floor(0.8 * sending_flow + 0.2 * self.sending_flow[time_step - 1]), original_sending_flow)
        if sending_flow < 0:
            raise ValueError('Negative sending flow detected, sending flow more than original flow')
        self.sending_flow[time_step] = sending_flow
        # self.sending_flow[time_step] = max(0, sending_flow)
        # self.sending_flow[time_step] = min(max(0, sending_flow), self.num_pedestrians[time_step])
        return self.sending_flow[time_step]

    def cal_receiving_flow(self, time_step: int) -> float:
        """
        Calculate the receiving flow of the link at a given time step
        :param time_step: Current time step - 1
        """
        # if self.link_id == '2_3' and time_step == 110:
        #     pass
        #TODO: is using length the correct way to calculate receiving flow?
        tau_shockwave = round(self.length/(self.shockwave_speed * self.unit_time))
        if time_step + 1 - tau_shockwave < 0:
            receiving_flow_boundary = self.k_jam * self.area
        else:
            receiving_flow_boundary = (self.cumulative_outflow[time_step + 1 - tau_shockwave]
                              + self.k_jam * self.area - self.cumulative_inflow[time_step])

        # receiving_flow_max = self.k_critical * self.free_flow_speed * self.unit_time
        receiving_flow_max = self.back_gate_width * self.k_critical * self.free_flow_speed * self.unit_time
        receiving_flow = min(receiving_flow_boundary, receiving_flow_max)
        if receiving_flow < 0:
            print(f"Negative receiving flow detected, {receiving_flow} at {time_step}")
        receiving_flow = max(receiving_flow, 0)

        '''smooth the receiving flow (Optional when we use the mitigation method)'''
        if self.receiving_flow[time_step-1] >=0:
            receiving_flow = min(np.floor(receiving_flow * 0.8 + self.receiving_flow[time_step-1] * 0.2), receiving_flow)

        # self.receiving_flow[time_step] = receiving_flow

        return receiving_flow
    
    def cal_receiving_flow_with_reverse(self, time_step: int, reverse_sending_flow: float) -> float:
        """Calculate receiving flow considering reverse link interaction"""
        forward_receiving_flow = self.cal_receiving_flow(time_step)
        
        # Simulate people squeeze in and out of the link
        # if forward_receiving_flow <= reverse_sending_flow:
        #     reverse_sending_flow -= np.random.binomial(n=int(reverse_sending_flow), p=0.2)

        receiving_flow = forward_receiving_flow - reverse_sending_flow
        return max(receiving_flow, 0)

class Separator(Link):
    """Separator: control object in the network, it adjust the width of the bidirection link"""
    def __init__(self, link_id, start_node, end_node, simulation_steps, unit_time, **kwargs):
        super().__init__(link_id, start_node, end_node, simulation_steps, unit_time, **kwargs)
        self._separator_width = self._width / 2
        self._front_gate_width = self._width / 2
        self._back_gate_width = self._width / 2

    def get_density(self, time_step: int):
        return self.density[time_step]
    
    @property
    def area(self):
        return self.length * self._separator_width

    @property
    def separator_width(self):
        return self._separator_width

    @separator_width.setter
    def separator_width(self, value):
        """
        Sets the width of this link and dynamically adjusts the reverse link's width
        to maintain a constant total corridor width.
        value: float, the width of the separator, has minimum and maximum value
        """
        self._separator_width = value
        # Update gate widths to match separator width
        self._front_gate_width = value
        self._back_gate_width = value
        
        if self.reverse_link:
            self.reverse_link._separator_width = self._width - value
            # Also update the reverse link's gate widths
            self.reverse_link._front_gate_width = self._width - value
            self.reverse_link._back_gate_width = self._width - value

    def cal_receiving_flow_with_reverse(self, time_step: int, reverse_sending_flow: float) -> float:
        """Calculate receiving flow for separator (no reverse link interaction)"""
        forward_receiving_flow = self.cal_receiving_flow(time_step)
        return max(forward_receiving_flow, 0)



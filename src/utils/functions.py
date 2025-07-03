import numpy as np

class SpeedDensityFd:
    """
    A callable class to model travel speed based on traffic density.
    This encapsulates the fundamental diagram parameters.
    """
    def __init__(self, v_f, k_critical, k_jam, model_type=1, add_noise=False):
        """
        Initializes the travel speed model with hyperparameters.
        :param v_f: Free-flow speed.
        :param k_critical: Critical density.
        :param k_jam: Jam density.
        :param model_type: The type of fundamental diagram model to use.
        """
        if k_jam <= k_critical:
            raise ValueError("k_jam must be greater than k_critical")
        self.v_f = v_f
        self.k_critical = k_critical
        self.k_jam = k_jam
        self.u0 = 1.5
        self.gamma = self.u0 * self.k_critical 
        self.model_type = model_type
        self.add_noise = add_noise
    def __call__(self, density: float) -> float:
        """Calculate the travel speed for a given density."""
        # type1: linear density speed fundamental diagram (Greenshields)
        if self.model_type == 'greenshields':
            if density <= self.k_critical:
                v = self.v_f
            elif self.k_critical < density:
                v = max(0, -self.v_f * (density - self.k_jam) / (self.k_jam - self.k_critical))

        # type2: triangular density flow fundamental diagram (Yperman's LTM)
        elif self.model_type == 'yperman':
            if density <= self.k_critical:
                v = self.v_f
            elif self.k_critical < density:
                v = max(0, (self.k_critical * self.v_f) / (self.k_jam - self.k_critical) * (self.k_jam / density - 1))

        # type3: Smulders’ fundamental diagram vehiclar
        elif self.model_type == 'smulders':
            if density <= self.k_critical:
                v = self.u0 * (1 - density / self.k_jam)
            elif self.k_critical < density:
                v = max(0, self.gamma * (1 / density - 1 / self.k_jam))

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        if self.add_noise:
            v += np.random.normal(0, 0.01)

        return max(0, v)


def travel_cost(link, inflow, outflow):
    """Calculate the travel time of a link"""
    return link.length / (link.free_flow_speed * (1 - (inflow + outflow) / link.capacity))

def cal_travel_speed(density, v_f, k_critical, k_jam):
    """Calculate the travel speed of a link based on density"""
    if density <= k_critical:
        return v_f
    elif k_critical < density:
        noise = 0
        # if np.random.rand() < 0.5:
        #     noise = np.random.normal(0, 0.1)
        # return max(0, v_f * (1 - density / k_jam) + noise)
    return max(0, -v_f * (density - k_jam) / (k_jam - k_critical))

def cal_free_flow_speed(density_i, density_j, v_f):
    """Calculate the travel speed based on densities"""
    if density_i + density_j == 0:
        return v_f
    
    rho = density_i / (density_i + density_j)
    v_f = v_f/np.exp(1 - rho)
    return v_f

def cal_travel_time(link_length, density, v_max, k_critical, k_jam):
    """Calculate the travel time of a link"""
    speed = cal_travel_speed(density, v_max, k_critical, k_jam)
    if speed == 0:
        return float('inf')
    return link_length / speed 

def cal_link_flow_fd(density, v_f, k_jam, k_critical, shock_wave):
    """Triangular fundamental diagram, return link flow
    """
    if density <= k_critical:
        return v_f * density
    else:
        return shock_wave * (k_jam - density)
    
def cal_link_flow_kv(density, v):
    """
    Link flow q = v * density
    """
    return v * density
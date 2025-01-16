import numpy as np

def travel_cost(link, inflow, outflow):
    """Calculate the travel time of a link"""
    return link.length / (link.free_flow_speed * (1 - (inflow + outflow) / link.capacity))

def cal_travel_speed(density, v_f, k_critical, k_jam):
    """Calculate the travel speed of a link based on density"""
    if density <= k_critical:
        return v_f
    elif k_critical < density:
        return max(0, v_f * (1 - density / k_jam))

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
import sys
sys.path.append("/Users/mmai/Devs/Crowd-Control")
from rl.pz_pednet_env import PedNetParallelEnv

env = PedNetParallelEnv(dataset="butterfly_scF", obs_mode="option4")
print("Agents:", env.possible_agents)
for agent_id in env.possible_agents:
    print(agent_id, "obs_dim:", env.observation_space(agent_id).shape[0], "act_dim:", env.action_space(agent_id).shape[0])

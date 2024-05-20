import networkx as nx
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.dqn import DQN
import matplotlib.pyplot as plt
import csv

# Load NSFNET topology
def load_topology():
    graph = nx.read_gml('nsfnet.gml')
    nx.set_edge_attributes(graph, 10, 'capacity')
    return graph

# Request class
class Request:
    def __init__(self, s, t, ht):
        self.s = s
        self.t = t
        self.ht = ht

# EdgeStats class
class EdgeStats:
    def __init__(self, u, v, cap):
        self.id = (u, v)
        self.cap = cap
        self.slots = [None] * cap
        self.hts = [0] * cap

    def add_request(self, req, color):
        self.slots[color] = req
        self.hts[color] = req.ht

    def remove_requests(self):
        for i in range(self.cap):
            if self.hts[i] > 0:
                self.hts[i] -= 1
                if self.hts[i] == 0:
                    self.slots[i] = None

    def get_available_colors(self):
        return [i for i in range(self.cap) if self.slots[i] is None]

# Generate requests
def generate_requests(num_reqs, nodes, case):
    requests = []
    if case == 'case1':
        s, t = 'San Diego Supercomputer Center', 'Jon Von Neumann Center, Princeton, NJ'
        for _ in range(num_reqs):
            ht = np.random.randint(10, 20)
            requests.append(Request(s, t, ht))
    elif case == 'case2':
        for _ in range(num_reqs):
            s, d = np.random.choice(nodes, 2, replace=False)
            ht = np.random.randint(10, 20)
            requests.append(Request(s, d, ht))
    return requests

# Optical Network Environment class
class OpticalNetworkEnv(gym.Env):
    def __init__(self, config):
        super(OpticalNetworkEnv, self).__init__()
        self.graph = config['graph']
        self.num_requests = config['num_requests']
        self.action_space = spaces.Discrete(len(self.graph.edges))
        self.observation_space = spaces.Box(0, 1, shape=(len(self.graph.nodes),), dtype=np.float32)
        self.current_request_index = 0
        self.requests = generate_requests(self.num_requests, list(self.graph.nodes), config['case'])
        self.estats = {e: EdgeStats(*e, self.graph[e[0]][e[1]]['capacity']) for e in self.graph.edges}

    def step(self, action):
        request = self.requests[self.current_request_index]
        edge = list(self.graph.edges)[action]

        done = False
        reward = -1
        if self._is_action_valid(action, request):
            reward = 1
            self._allocate_spectrum(edge, request)
            print(f"Action {action} valid for request {self.current_request_index}. Edge: {edge}")
        else:
            reward = -1
            print(f"Action {action} invalid for request {self.current_request_index}. Edge: {edge}")

        self.current_request_index += 1
        if self.current_request_index >= self.num_requests:
            done = True

        return self._get_observation(), reward, done, {}

    def reset(self):
        self.current_request_index = 0
        self.requests = generate_requests(self.num_requests, list(self.graph.nodes), 'case2')
        return self._get_observation()

    def render(self, mode='human'):
        pass

    def _get_observation(self):
        obs = np.zeros(len(self.graph.nodes))
        for edge in self.graph.edges:
            if self.graph.edges[edge].get('allocated', False):
                obs[list(self.graph.nodes).index(edge[0])] = 1
                obs[list(self.graph.nodes).index(edge[1])] = 1
        return obs

    def _is_action_valid(self, action, request):
        edge = list(self.graph.edges)[action]
        path = nx.shortest_path(self.graph, source=request.s, target=request.t)
        valid = edge in zip(path[:-1], path[1:]) or edge[::-1] in zip(path[:-1], path[1:])
        return valid

    def _allocate_spectrum(self, edge, request):
        estats = self.estats.get(edge) or self.estats.get(edge[::-1])
        available_colors = estats.get_available_colors()
        if available_colors:
            estats.add_request(request, available_colors[0])
            self.graph.edges[edge]['allocated'] = True
            self.estats[edge] = estats
            print(f"Allocated request {request} on edge {edge} with color {available_colors[0]}")

# Train RL
def train_rl(algorithm, graph, num_requests, case, csv_filename):
    config = {
        "env": OpticalNetworkEnv,
        "env_config": {
            "graph": graph,
            "num_requests": num_requests,
            "case": case,
        },
        "num_workers": 1,
    }
    trainer = algorithm(config=config)
    rewards = []

    # Open CSV file for writing
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Reward"])  # Write the header row

        for i in range(100):
            result = trainer.train()
            reward = result['episode_reward_mean']
            rewards.append(reward)
            writer.writerow([i, reward])  # Write the episode number and reward to the CSV file
            print(f"Episode {i}: reward={reward}")

    return trainer, rewards

# Spectrum allocation
def spectrum_allocation(graph, estats, req):
    path = nx.shortest_path(graph, source=req.s, target=req.t)
    path_edges = list(zip(path[:-1], path[1:]))
    available_colors = [estats[edge].get_available_colors() if edge in estats else estats[edge[::-1]].get_available_colors() for edge in path_edges]
    min_length = min(len(colors) for colors in available_colors)
    color = next((c for c in range(min_length) if all(c in colors for colors in available_colors)), None)
    if color is not None:
        for edge in path_edges:
            estats[edge].add_request(req, color) if edge in estats else estats[edge[::-1]].add_request(req, color)
    else:
        print(f"Request {req} blocked.")

# Run simulation
def run_simulation(graph, num_requests, case):
    nodes = list(graph.nodes)
    requests = generate_requests(num_requests, nodes, case)
    estats = {e: EdgeStats(*e, graph[e[0]][e[1]]['capacity']) for e in graph.edges}
    env = OpticalNetworkEnv({"graph": graph, "num_requests": num_requests, "case": case})
    ppo_trainer, ppo_rewards = train_rl(PPO, graph, num_requests, case, 'ppo_rewards.csv')
    dqn_trainer, dqn_rewards = train_rl(DQN, graph, num_requests, case, 'dqn_rewards.csv')

    for req in requests:
        spectrum_allocation(graph, estats, req)
        for es in estats.values():
            es.remove_requests()

    return estats, ppo_rewards, dqn_rewards

# Plot results
def plot_results(rewards, title):
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(title)
    plt.show()

# Main execution
if __name__ == "__main__":
    graph = load_topology()
    estats, ppo_rewards, dqn_rewards = run_simulation(graph, 100, 'case1')

    plot_results(ppo_rewards, "PPO Learning Curve - Case I")
    plot_results(dqn_rewards, "DQN Learning Curve - Case I")

    graph1 = load_topology()
    estats, ppo_rewards1, dqn_rewards1 = run_simulation(graph1, 100, 'case2')
    plot_results(ppo_rewards1, "PPO Learning Curve - Case II")
    plot_results(dqn_rewards1, "DQN Learning Curve - Case II")

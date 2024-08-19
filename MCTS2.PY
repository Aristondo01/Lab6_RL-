import gymnasium as gym

# Crear el entorno FrozenLake-v1
env = gym.make('FrozenLake-v1', is_slippery=False, render_mode="human")  # is_slippery=False para una dinámica determinista

import math
import numpy as np
import time

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self, actions):
        return len(self.children) == len(actions)

    def add_child(self, action, state):
        child = MCTSNode(state, parent=self)
        self.children[action] = child
        return child

    def update(self, reward):
        self.visits += 1
        self.value += reward

    def best_child(self, exploration_weight=1.0):
        choices_weights = [
            (child.value / child.visits) + exploration_weight * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children.values()
        ]
        return list(self.children.values())[np.argmax(choices_weights)]

def selection(node, env):
    actions = list(range(env.action_space.n))
    while not node.is_fully_expanded(actions):
        if not node.children:
            return node
        node = node.best_child()
    return node

def expansion(node, env):
    actions = list(range(env.action_space.n))
    for action in actions:
        if action not in node.children:
            state, _, done, _, _ = env.step(action)
            return node.add_child(action, state)
    return node

def simulation(env, state, max_steps=100):
    total_reward = 0
    env.reset()
    env.env.s = state
    for _ in range(max_steps):
        action = env.action_space.sample()
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward

def backpropagation(node, reward):
    while node is not None:
        node.update(reward)
        node = node.parent

def mcts(env, num_iterations=1000, exploration_weight=1.0):
    root = MCTSNode(env.reset())
    for _ in range(num_iterations):
        node = selection(root, env)
        if not node.is_fully_expanded(list(range(env.action_space.n))):
            node = expansion(node, env)
        reward = simulation(env, node.state)
        backpropagation(node, reward)
    return root.best_child(exploration_weight).state

state = env.reset()
done = False

while not done:
    action = mcts(env)
    state, reward, done, _, _ = env.step(action)
    env.render()
    
    if done and reward == 0:
        print("El agente ha caído en un agujero.")
        done = False
        state = env.reset()
        time.sleep(0.1)
    else:
        print(f"Llegada a la meta. Recompensa: {reward}")

print(f"Recompensa total: {reward}")

import gym
import numpy as np
import math
import random
import matplotlib.pyplot as plt

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward = 0
        self.is_terminal = False
        self.action = action
        self.untried_actions = list(range(env.action_space.n))

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, exploration_weight=1.41):
        return max(self.children, key=lambda node: node.ucb1(exploration_weight))

    def ucb1(self, exploration_weight):
        if self.visits == 0:
            return float('inf')
        return (self.reward / self.visits) + exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)

class MCTS:
    def __init__(self, env, iterations=1000, exploration_weight=1.41):
        self.envi = env
        self.iterations = iterations
        self.exploration_weight = exploration_weight
        self.rewards = []

    def search(self, initial_state):
        root = Node(state=initial_state)

        for _ in range(self.iterations):
            node = self._select(root)
            if node.is_terminal:
                continue
            reward = self._simulate(node)
            self._backpropagate(node, reward)
            self.rewards.append(reward)  # Guardar recompensa después de cada simulación

        return root.best_child(exploration_weight=0).action

    def _select(self, node):
        while not node.is_terminal:
            if not node.is_fully_expanded():
                return self._expand(node)
            else:
                node = node.best_child(self.exploration_weight)
        return node

    def _expand(self, node):
        if not node.untried_actions:
            return node

        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)  # Elimina la acción una vez que se ha probado

        next_state, reward, done, _, _ = self.envi.step(action)
        child_node = Node(state=next_state, parent=node, action=action)
        child_node.is_terminal = done
        node.children.append(child_node)
        return child_node

    def _simulate(self, node):
        current_state = node.state
        total_reward = 0
        done = False

        self.envi.env.s = current_state  # Establecer el estado inicial para la simulación

        while not done:
            action = self.envi.action_space.sample()
            next_state, reward, done, _, _ = self.envi.step(action)
            total_reward += reward
            current_state = next_state  # Actualizar el estado real después de cada acción

        print(f"Simulación finalizada con recompensa: {total_reward}")
        return total_reward

    def _backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.reward += reward
            node = node.parent

env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='human')
mcts = MCTS(env, iterations=1000)

state, _ = env.reset()
done = False

while not done:
    action = mcts.search(state)
    state, reward, done, _, _ = env.step(action)
    env.render()

    if done and reward > 0:
        print("Episodio terminado.")
        state, _ = env.reset()
    else:
        done = False
        state, _ = env.reset()

env.close()

# Graficar las recompensas
plt.plot(mcts.rewards)
plt.xlabel('Iteración')
plt.ylabel('Recompensa')
plt.title('Recompensas acumuladas en MCTS')
plt.show()

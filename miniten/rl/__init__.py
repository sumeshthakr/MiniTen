"""
Reinforcement Learning Basics

Basic reinforcement learning algorithms for edge devices.
Optimized for low-memory and fast inference.

Features:
- Q-Learning (tabular and function approximation)
- Policy Gradient (REINFORCE)
- Deep Q-Network (DQN) - lightweight version
- Environment interface
- Replay buffer
- Minimal dependencies
"""

import math
import random
from typing import List, Tuple, Dict, Optional, Callable, Any
from abc import ABC, abstractmethod


# ============================================================================
# Environment Interface
# ============================================================================

class Environment(ABC):
    """
    Abstract base class for RL environments.
    """
    
    @abstractmethod
    def reset(self) -> Any:
        """Reset environment and return initial state."""
        pass
    
    @abstractmethod
    def step(self, action: int) -> Tuple[Any, float, bool, Dict]:
        """
        Take action in environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        pass
    
    @property
    @abstractmethod
    def action_space(self) -> int:
        """Number of possible actions."""
        pass
    
    @property
    @abstractmethod
    def observation_space(self) -> Tuple[int, ...]:
        """Shape of observation space."""
        pass


class DiscreteEnvironment(Environment):
    """
    Simple discrete environment wrapper.
    """
    
    def __init__(self, n_states: int, n_actions: int,
                 transition_fn: Callable[[int, int], Tuple[int, float, bool]]):
        self._n_states = n_states
        self._n_actions = n_actions
        self._transition_fn = transition_fn
        self._current_state = 0
    
    def reset(self) -> int:
        self._current_state = 0
        return self._current_state
    
    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        next_state, reward, done = self._transition_fn(self._current_state, action)
        self._current_state = next_state
        return next_state, reward, done, {}
    
    @property
    def action_space(self) -> int:
        return self._n_actions
    
    @property
    def observation_space(self) -> Tuple[int, ...]:
        return (self._n_states,)


# ============================================================================
# Built-in Environments
# ============================================================================

class GridWorld(Environment):
    """
    Simple grid world environment.
    Agent navigates grid to reach goal.
    """
    
    def __init__(self, size: int = 5, goal: Tuple[int, int] = None):
        self.size = size
        self.goal = goal or (size - 1, size - 1)
        self.agent_pos = [0, 0]
    
    def reset(self) -> List[int]:
        self.agent_pos = [0, 0]
        return list(self.agent_pos)
    
    def step(self, action: int) -> Tuple[List[int], float, bool, Dict]:
        # Actions: 0=up, 1=down, 2=left, 3=right
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        
        # Move
        new_x = max(0, min(self.size - 1, self.agent_pos[0] + dx))
        new_y = max(0, min(self.size - 1, self.agent_pos[1] + dy))
        self.agent_pos = [new_x, new_y]
        
        # Check goal
        done = tuple(self.agent_pos) == self.goal
        reward = 1.0 if done else -0.01
        
        return list(self.agent_pos), reward, done, {}
    
    @property
    def action_space(self) -> int:
        return 4
    
    @property
    def observation_space(self) -> Tuple[int, ...]:
        return (2,)


class CartPoleEnv(Environment):
    """
    Simplified CartPole environment.
    Balance a pole on a cart.
    """
    
    def __init__(self):
        self.gravity = 9.8
        self.cart_mass = 1.0
        self.pole_mass = 0.1
        self.pole_length = 0.5
        self.force_mag = 10.0
        self.dt = 0.02
        
        self.state = [0.0, 0.0, 0.0, 0.0]  # x, x_dot, theta, theta_dot
        self.max_steps = 200
        self.step_count = 0
    
    def reset(self) -> List[float]:
        self.state = [
            random.uniform(-0.05, 0.05),
            random.uniform(-0.05, 0.05),
            random.uniform(-0.05, 0.05),
            random.uniform(-0.05, 0.05)
        ]
        self.step_count = 0
        return list(self.state)
    
    def step(self, action: int) -> Tuple[List[float], float, bool, Dict]:
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        
        # Physics simulation
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        
        total_mass = self.cart_mass + self.pole_mass
        pole_mass_length = self.pole_mass * self.pole_length
        
        temp = (force + pole_mass_length * theta_dot**2 * sin_theta) / total_mass
        theta_acc = (self.gravity * sin_theta - cos_theta * temp) / (
            self.pole_length * (4/3 - self.pole_mass * cos_theta**2 / total_mass)
        )
        x_acc = temp - pole_mass_length * theta_acc * cos_theta / total_mass
        
        # Update state
        x = x + self.dt * x_dot
        x_dot = x_dot + self.dt * x_acc
        theta = theta + self.dt * theta_dot
        theta_dot = theta_dot + self.dt * theta_acc
        
        self.state = [x, x_dot, theta, theta_dot]
        self.step_count += 1
        
        # Check termination
        done = (
            abs(x) > 2.4 or
            abs(theta) > 0.21 or
            self.step_count >= self.max_steps
        )
        
        reward = 1.0 if not done else 0.0
        
        return list(self.state), reward, done, {}
    
    @property
    def action_space(self) -> int:
        return 2
    
    @property
    def observation_space(self) -> Tuple[int, ...]:
        return (4,)


# ============================================================================
# Replay Buffer
# ============================================================================

class ReplayBuffer:
    """
    Experience replay buffer for off-policy learning.
    """
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: List[Tuple] = []
        self.position = 0
    
    def push(self, state, action: int, reward: float, 
             next_state, done: bool):
        """Add experience to buffer."""
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample batch from buffer."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)


# ============================================================================
# Q-Learning (Tabular)
# ============================================================================

class TabularQLearning:
    """
    Tabular Q-Learning algorithm.
    For discrete state and action spaces.
    """
    
    def __init__(self, n_states: int, n_actions: int,
                 learning_rate: float = 0.1,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table
        self.q_table = [[0.0] * n_actions for _ in range(n_states)]
    
    def _state_to_index(self, state) -> int:
        """Convert state to index."""
        if isinstance(state, int):
            return state
        elif isinstance(state, (list, tuple)):
            # Simple hashing for small grids
            return hash(tuple(state)) % self.n_states
        return 0
    
    def select_action(self, state, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        
        state_idx = self._state_to_index(state)
        q_values = self.q_table[state_idx]
        return q_values.index(max(q_values))
    
    def update(self, state, action: int, reward: float, 
               next_state, done: bool):
        """Update Q-value using temporal difference."""
        state_idx = self._state_to_index(state)
        next_state_idx = self._state_to_index(next_state)
        
        # Q-learning update
        current_q = self.q_table[state_idx][action]
        
        if done:
            target = reward
        else:
            max_next_q = max(self.q_table[next_state_idx])
            target = reward + self.gamma * max_next_q
        
        self.q_table[state_idx][action] += self.lr * (target - current_q)
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train_episode(self, env: Environment) -> float:
        """Train for one episode."""
        state = env.reset()
        total_reward = 0.0
        done = False
        
        while not done:
            action = self.select_action(state, training=True)
            next_state, reward, done, _ = env.step(action)
            
            self.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
        
        self.decay_epsilon()
        return total_reward


# ============================================================================
# Deep Q-Network (Lightweight)
# ============================================================================

class DQN:
    """
    Lightweight Deep Q-Network for continuous state spaces.
    Uses simple neural network without external dependencies.
    """
    
    def __init__(self, state_dim: int, n_actions: int,
                 hidden_sizes: List[int] = None,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        if hidden_sizes is None:
            hidden_sizes = [64, 64]
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Build network
        self.layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_sizes:
            self.layers.append({
                'weights': self._init_weights(prev_dim, hidden_dim),
                'bias': [0.0] * hidden_dim
            })
            prev_dim = hidden_dim
        
        # Output layer
        self.layers.append({
            'weights': self._init_weights(prev_dim, n_actions),
            'bias': [0.0] * n_actions
        })
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(10000)
    
    def _init_weights(self, in_dim: int, out_dim: int) -> List[List[float]]:
        """Initialize weights using Xavier initialization."""
        scale = math.sqrt(6.0 / (in_dim + out_dim))
        return [[random.uniform(-scale, scale) for _ in range(out_dim)]
                for _ in range(in_dim)]
    
    def _relu(self, x: List[float]) -> List[float]:
        """ReLU activation."""
        return [max(0, v) for v in x]
    
    def _forward(self, state: List[float]) -> Tuple[List[float], List[List[float]]]:
        """Forward pass, returns Q-values and activations for backprop."""
        activations = [state]
        x = state
        
        for i, layer in enumerate(self.layers):
            # Linear transform
            out = list(layer['bias'])
            for j in range(len(out)):
                for k in range(len(x)):
                    out[j] += x[k] * layer['weights'][k][j]
            
            # Activation (ReLU for hidden, none for output)
            if i < len(self.layers) - 1:
                out = self._relu(out)
            
            activations.append(out)
            x = out
        
        return x, activations
    
    def get_q_values(self, state: List[float]) -> List[float]:
        """Get Q-values for state."""
        q_values, _ = self._forward(state)
        return q_values
    
    def select_action(self, state: List[float], training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        
        q_values = self.get_q_values(state)
        return q_values.index(max(q_values))
    
    def _backward(self, state: List[float], action: int, 
                  target: float, activations: List[List[float]]):
        """Backpropagation to update weights."""
        # Compute output gradient
        q_values = activations[-1]
        output_grad = [0.0] * self.n_actions
        output_grad[action] = q_values[action] - target
        
        # Backpropagate
        grad = output_grad
        
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            prev_activation = activations[i]
            
            # Update weights and bias
            for j in range(len(grad)):
                layer['bias'][j] -= self.lr * grad[j]
                for k in range(len(prev_activation)):
                    layer['weights'][k][j] -= self.lr * grad[j] * prev_activation[k]
            
            # Compute gradient for previous layer
            if i > 0:
                new_grad = [0.0] * len(prev_activation)
                for k in range(len(prev_activation)):
                    for j in range(len(grad)):
                        new_grad[k] += grad[j] * layer['weights'][k][j]
                
                # ReLU derivative
                new_grad = [g if prev_activation[k] > 0 else 0 
                           for k, g in enumerate(new_grad)]
                grad = new_grad
    
    def update(self, batch_size: int = 32):
        """Update network from replay buffer."""
        if len(self.replay_buffer) < batch_size:
            return
        
        batch = self.replay_buffer.sample(batch_size)
        
        for state, action, reward, next_state, done in batch:
            _, activations = self._forward(state)
            
            if done:
                target = reward
            else:
                next_q = self.get_q_values(next_state)
                target = reward + self.gamma * max(next_q)
            
            self._backward(state, action, target, activations)
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train_episode(self, env: Environment, batch_size: int = 32) -> float:
        """Train for one episode."""
        state = list(env.reset())
        total_reward = 0.0
        done = False
        
        while not done:
            action = self.select_action(state, training=True)
            next_state, reward, done, _ = env.step(action)
            next_state = list(next_state)
            
            self.replay_buffer.push(state, action, reward, next_state, done)
            self.update(batch_size)
            
            state = next_state
            total_reward += reward
        
        self.decay_epsilon()
        return total_reward


# ============================================================================
# Policy Gradient (REINFORCE)
# ============================================================================

class REINFORCE:
    """
    REINFORCE policy gradient algorithm.
    Monte Carlo policy gradient for discrete actions.
    """
    
    def __init__(self, state_dim: int, n_actions: int,
                 hidden_sizes: List[int] = None,
                 learning_rate: float = 0.01,
                 gamma: float = 0.99):
        if hidden_sizes is None:
            hidden_sizes = [64]
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        
        # Build policy network
        self.layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_sizes:
            self.layers.append({
                'weights': self._init_weights(prev_dim, hidden_dim),
                'bias': [0.0] * hidden_dim
            })
            prev_dim = hidden_dim
        
        # Output layer (logits)
        self.layers.append({
            'weights': self._init_weights(prev_dim, n_actions),
            'bias': [0.0] * n_actions
        })
        
        # Episode memory
        self.saved_log_probs: List[Tuple[int, List[float]]] = []
        self.rewards: List[float] = []
    
    def _init_weights(self, in_dim: int, out_dim: int) -> List[List[float]]:
        """Initialize weights."""
        scale = math.sqrt(6.0 / (in_dim + out_dim))
        return [[random.uniform(-scale, scale) for _ in range(out_dim)]
                for _ in range(in_dim)]
    
    def _relu(self, x: List[float]) -> List[float]:
        """ReLU activation."""
        return [max(0, v) for v in x]
    
    def _softmax(self, x: List[float]) -> List[float]:
        """Softmax function."""
        max_x = max(x)
        exp_x = [math.exp(v - max_x) for v in x]
        sum_exp = sum(exp_x)
        return [e / sum_exp for e in exp_x]
    
    def _forward(self, state: List[float]) -> Tuple[List[float], List[List[float]]]:
        """Forward pass, returns action probabilities and activations."""
        activations = [state]
        x = state
        
        for i, layer in enumerate(self.layers):
            # Linear transform
            out = list(layer['bias'])
            for j in range(len(out)):
                for k in range(len(x)):
                    out[j] += x[k] * layer['weights'][k][j]
            
            # Activation
            if i < len(self.layers) - 1:
                out = self._relu(out)
            
            activations.append(out)
            x = out
        
        # Softmax for probabilities
        probs = self._softmax(x)
        return probs, activations
    
    def select_action(self, state: List[float], training: bool = True) -> int:
        """Select action by sampling from policy."""
        probs, _ = self._forward(state)
        
        # Sample from distribution
        r = random.random()
        cumsum = 0.0
        for i, p in enumerate(probs):
            cumsum += p
            if r < cumsum:
                if training:
                    self.saved_log_probs.append((i, probs))
                return i
        
        if training:
            self.saved_log_probs.append((len(probs) - 1, probs))
        return len(probs) - 1
    
    def store_reward(self, reward: float):
        """Store reward for current step."""
        self.rewards.append(reward)
    
    def _compute_returns(self) -> List[float]:
        """Compute discounted returns."""
        returns = []
        R = 0.0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        # Normalize returns
        mean_return = sum(returns) / len(returns) if returns else 0
        std_return = math.sqrt(sum((r - mean_return)**2 for r in returns) / len(returns)) if returns else 1
        std_return = max(std_return, 1e-8)
        
        return [(r - mean_return) / std_return for r in returns]
    
    def update(self):
        """Update policy using collected episode."""
        if not self.saved_log_probs:
            return
        
        returns = self._compute_returns()
        
        for (action, probs), G in zip(self.saved_log_probs, returns):
            # Simple gradient update - approximate policy gradient
            for layer in self.layers:
                for j in range(len(layer['bias'])):
                    layer['bias'][j] += self.lr * G * (1 if j == action else 0)
        
        # Clear episode memory
        self.saved_log_probs = []
        self.rewards = []
    
    def train_episode(self, env: Environment) -> float:
        """Train for one episode."""
        state = list(env.reset())
        total_reward = 0.0
        done = False
        
        while not done:
            action = self.select_action(state, training=True)
            next_state, reward, done, _ = env.step(action)
            
            self.store_reward(reward)
            
            state = list(next_state)
            total_reward += reward
        
        self.update()
        return total_reward


# ============================================================================
# Training Utilities
# ============================================================================

def train_agent(agent, env: Environment, n_episodes: int = 1000,
                log_interval: int = 100, verbose: bool = True) -> List[float]:
    """
    Train agent for specified number of episodes.
    
    Args:
        agent: RL agent (TabularQLearning, DQN, or REINFORCE)
        env: Environment
        n_episodes: Number of training episodes
        log_interval: Interval for logging progress
        verbose: Whether to print progress
        
    Returns:
        List of episode rewards
    """
    rewards = []
    
    for episode in range(n_episodes):
        reward = agent.train_episode(env)
        rewards.append(reward)
        
        if verbose and (episode + 1) % log_interval == 0:
            avg_reward = sum(rewards[-log_interval:]) / log_interval
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}")
    
    return rewards


def evaluate_agent(agent, env: Environment, n_episodes: int = 10) -> float:
    """
    Evaluate agent performance.
    
    Args:
        agent: Trained agent
        env: Environment
        n_episodes: Number of evaluation episodes
        
    Returns:
        Average reward
    """
    total_reward = 0.0
    
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        
        while not done:
            if isinstance(state, list):
                action = agent.select_action(state, training=False)
            else:
                action = agent.select_action(state, training=False)
            
            state, reward, done, _ = env.step(action)
            total_reward += reward
    
    return total_reward / n_episodes

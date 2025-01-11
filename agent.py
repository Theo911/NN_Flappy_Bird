import gymnasium as gym
import numpy as np
import random
import torch
from torch import nn
import yaml

from experience_replay import ReplayMemory
from dqn import DQN

from datetime import datetime
import argparse
import itertools

import flappy_bird_gymnasium
import os
import signal

DATE_FORMAT = "%m-%d %H:%M:%S"

RUNS_DIR = os.path.join(os.path.dirname(__file__), 'runs')
os.makedirs(RUNS_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# device = 'cpu'

class Agent():

    def __init__(self, hyperparameter_set):
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, 'hyperparameters.yml')
        with open(file_path, 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
            # print(hyperparameters)

        self.hyperparameter_set = hyperparameter_set

        self.env_id = hyperparameters['env_id']
        self.learning_rate_a = hyperparameters['learning_rate_a']
        self.discount_factor_g = hyperparameters['discount_factor_g']
        self.network_sync_rate = hyperparameters[
            'network_sync_rate']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters[
            'mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.stop_on_reward = hyperparameters['stop_on_reward']
        self.fc1_nodes = hyperparameters['fc1_nodes']
        self.env_make_params = hyperparameters.get('env_make_params',
                                                   {})
        self.enable_double_dqn = hyperparameters['enable_double_dqn']  # double dqn on/off flag
        self.best_reward = -9999999
        self.epsilon = self.epsilon_init

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

    def save_agent_state(self, policy_dqn, best_reward, epsilon):
        """
        Save the agent's state, including the model, optimizer state,
        best reward, and epsilon for future training continuation.
        """
        state_dict = {
            'model': policy_dqn.state_dict(),
            'best_reward': best_reward,
            'epsilon': epsilon,
            'optimizer': self.optimizer.state_dict() if self.optimizer else None
        }
        torch.save(state_dict, self.MODEL_FILE)
        print(f"Saved agent state: {self.MODEL_FILE}")

    def load_agent_state(self, policy_dqn):
        """
        Load the agent's state, including the model, best reward,
        and optimizer state for continuation of training.
        If the checkpoint file doesn't exist, initialize a new model.
        """
        if os.path.exists(self.MODEL_FILE):
            try:
                checkpoint = torch.load(self.MODEL_FILE, map_location=torch.device(device))

                policy_dqn.load_state_dict(checkpoint['model'])

                self.best_reward = checkpoint['best_reward']
                self.epsilon = checkpoint['epsilon']

                if checkpoint['optimizer'] is not None and self.optimizer:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                print(f"Loaded agent state from {self.MODEL_FILE}")
            except KeyError:
                print(f"Checkpoint file is missing 'model' key. Initializing a new model.")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
        else:
            print(f"No checkpoint found. Initializing a new model.")

    def run(self, is_training=True, render=False):
        if is_training:
            start_time = datetime.now()
            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        env = gym.make(self.env_id, render_mode='human' if render else None, **self.env_make_params)

        num_actions = env.action_space.n

        num_states = env.observation_space.shape[0]

        rewards_per_episode = []

        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_double_dqn).to(device)

        if is_training:
            self.epsilon = self.epsilon_init
            memory = ReplayMemory(self.replay_memory_size)
            target_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_double_dqn).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
            epsilon_history = []
            step_count = 0

            if os.path.exists(self.MODEL_FILE):
                self.load_agent_state(policy_dqn)
                best_reward = self.best_reward
                self.epsilon = self.epsilon
                print(f"Resuming training from episode with best reward: {best_reward}")
            else:
                print("Starting fresh training session with new model.")

        else:
            # policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            checkpoint = torch.load(self.MODEL_FILE, map_location=torch.device(device))
            policy_dqn.load_state_dict(checkpoint['model'])
            policy_dqn.eval()

        def save_and_exit(signal, frame):
            print("Interrupt received, saving model...")
            self.save_agent_state(policy_dqn, best_reward, self.epsilon)
            exit(0)

        signal.signal(signal.SIGINT, save_and_exit)

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)
            terminated = False
            episode_reward = 0.0

            while not terminated and episode_reward < self.stop_on_reward:
                # Epsilon-greedy action selection
                if is_training and random.random() < self.epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                new_state, reward, terminated, truncated, info = env.step(action.item())
                episode_reward += reward
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    memory.append((state, action, new_state, reward, terminated))
                    step_count += 1

                state = new_state

            rewards_per_episode.append(episode_reward)

            if is_training:
                if episode_reward > self.best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} at episode {episode}"
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    self.save_agent_state(policy_dqn, episode_reward, self.epsilon)
                    self.best_reward = episode_reward

                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(self.epsilon)

                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0

    def optimize(self, mini_batch, policy_dqn, target_dqn):

        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)

        actions = torch.stack(actions)

        new_states = torch.stack(new_states)

        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            if self.enable_double_dqn:
                best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)

                target_q = rewards + (1 - terminations) * self.discount_factor_g * \
                           target_dqn(new_states).gather(dim=1,
                                                         index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
            else:
                target_q = rewards + (1 - terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]

        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)

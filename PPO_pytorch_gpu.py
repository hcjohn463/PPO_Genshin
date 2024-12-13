import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
import time
from torch.utils.tensorboard import SummaryWriter

# Hyperparameters
GAMMA = 0.99  # Discount factor
LAMBDA = 0.95  # GAE lambda
CLIP_EPSILON = 0.2  # PPO clipping epsilon
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.0002
ENTROPY_COEF = 0.01  # Entropy regularization coefficient
BATCH_SIZE = 64
EPOCHS = 10
REPLAY_SIZE = 2000
WIDTH = 84
HEIGHT = 84

# Optimized hyperparameters
WIDTH = 84  # Reduced from 160
HEIGHT = 84  # Reduced from 160

class ActorCriticNetwork(nn.Module):
    def __init__(self, state_h, state_w, action_dim):
        super().__init__()
        # Improved CNN architecture
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # Reduced kernel size, increased stride
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # Add batch normalization
        self.batch_norm = nn.BatchNorm2d(32)
        
        # Calculate flattened size
        self._to_linear = self._get_conv_output_size((1, state_h, state_w))
        
        # Improved shared layers
        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._to_linear, 512),  # Increased units
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )
        
        # Actor head with smaller final layer
        self.actor_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # Critic head with smaller final layer
        self.critic_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def _get_conv_output_size(self, shape):
        with torch.no_grad():
            input = torch.zeros(1, *shape)
            output = self.conv_layers(input)
            return int(np.prod(output.size()[1:]))

    def forward(self, x, mode='both'):
        x = self.conv_layers(x)
        x = self.batch_norm(x)
        x = self.shared_fc(x)
        
        if mode == 'actor':
            return torch.softmax(self.actor_head(x), dim=-1)
        elif mode == 'critic':
            return self.critic_head(x)
        else:  # both
            return (torch.softmax(self.actor_head(x), dim=-1), 
                    self.critic_head(x))


class PPO:
    def __init__(self, observation_width, observation_height, action_space, model_file, log_file):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_w = observation_width
        self.state_h = observation_height
        self.action_dim = action_space
        self.replay_buffer = deque(maxlen=REPLAY_SIZE)
        self.model_file = model_file
        self.log_file = log_file

        # Create networks
        self.actor, self.critic = self.create_actor_critic_networks()

        # Move models to device
        self.actor = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), 
                                          lr=ACTOR_LEARNING_RATE, 
                                          betas=(0.9, 0.999))
        self.critic_optimizer = optim.Adam(self.critic.parameters(), 
                                           lr=CRITIC_LEARNING_RATE, 
                                           betas=(0.9, 0.999))

        # Learning rate schedulers
        self.actor_scheduler = optim.lr_scheduler.ExponentialLR(
            self.actor_optimizer, gamma=0.9
        )
        self.critic_scheduler = optim.lr_scheduler.ExponentialLR(
            self.critic_optimizer, gamma=0.9
        )
        self.writer = SummaryWriter('./runs')
        self.global_step = 0  # 新增這一行

         # Use full file path and ensure directory exists
        self.log_file_path = log_file
        os.makedirs(os.path.dirname(self.log_file_path) or '.', exist_ok=True)
        
        try:
            # Open log file with unique filename to prevent conflicts
            unique_log_file = f"{self.log_file_path}_{int(time.time())}.log"
            self.log_file = open(unique_log_file, 'w')
        except PermissionError:
            # Fallback to using a default log path in the current directory
            default_log_file = f"ppo_log_{int(time.time())}.log"
            self.log_file = open(default_log_file, 'w')
            print(f"Warning: Could not create log file at {self.log_file_path}. Using {default_log_file} instead.")

    def create_actor_critic_networks(self):
        """
        Create and return actor and critic networks.
        
        Returns:
            tuple: (actor_network, critic_network)
        """
        actor = ActorCriticNetwork(self.state_h, self.state_w, self.action_dim)
        critic = ActorCriticNetwork(self.state_h, self.state_w, self.action_dim)
        
        return actor, critic

    def generalized_advantage_estimation(self, rewards, values, dones, gamma=GAMMA, lam=LAMBDA):
        """Generalized Advantage Estimation implementation"""
        advantages = []
        gae = 0
        
        # Reverse calculation of advantages
        for i in reversed(range(len(rewards))):
            # Reset GAE if terminal state
            delta = rewards[i] + (1 - dones[i]) * gamma * (values[i+1] if i+1 < len(values) else 0) - values[i]
            gae = delta + (1 - dones[i]) * gamma * lam * gae
            advantages.insert(0, gae)
        
        # Standardize advantages
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Calculate target values
        target_values = advantages + np.array(values[:-1])
        return advantages, target_values

    def train(self):
        if len(self.replay_buffer) < BATCH_SIZE:
                return

        # Sample from replay buffer
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        
        # Convert lists to numpy arrays first
        states = np.array([item[0] for item in minibatch])
        actions = np.array([item[1] for item in minibatch])
        rewards = np.array([item[2] for item in minibatch])
        next_states = np.array([item[3] for item in minibatch])
        dones = np.array([item[4] for item in minibatch])
        old_log_probs = np.array([item[5] for item in minibatch])

        # Reshape states and next_states to have correct dimensions [batch_size, channels, height, width]
        states = torch.FloatTensor(states).to(self.device)
        states = states.view(BATCH_SIZE, 1, HEIGHT, WIDTH)  # Changed from unsqueeze
        
        next_states = torch.FloatTensor(next_states).to(self.device)
        next_states = next_states.view(BATCH_SIZE, 1, HEIGHT, WIDTH)  # Changed from unsqueeze
        
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)

        # Compute values
        with torch.no_grad():
            # Change this line to extract the critic value from the tuple
            values = self.critic(states, mode='critic').squeeze()
            next_values = self.critic(next_states, mode='critic').squeeze()
            values = torch.cat([values, next_values[-1:]])

        # Compute advantages and target values
        advantages, target_values = self.generalized_advantage_estimation(
            rewards.cpu().numpy(), values.cpu().numpy(), dones.cpu().numpy()
        )
        advantages = torch.FloatTensor(advantages).to(self.device)
        target_values = torch.FloatTensor(target_values).to(self.device)

        # Multiple training epochs
        for epoch in range(EPOCHS):
            # Get current policy probabilities (only the actor part)
            curr_action_probs = self.actor(states, mode='actor')  
            log_probs = torch.log(curr_action_probs.gather(1, actions.unsqueeze(1)) + 1e-10).squeeze()

            # Compute policy ratio
            ratios = torch.exp(log_probs - old_log_probs)

            # Clipped surrogate objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantages
            policy_loss = -torch.mean(torch.min(surr1, surr2))

            # Entropy regularization
            entropy = -torch.mean(torch.sum(curr_action_probs * torch.log(curr_action_probs + 1e-10), dim=1))

            # Value function loss (critic part)
            critic_values = self.critic(states, mode='critic').squeeze()
            critic_loss = torch.mean((critic_values - target_values) ** 2)

            # Total loss
            actor_loss = policy_loss - ENTROPY_COEF * entropy
            total_loss = actor_loss + 0.5 * critic_loss

            # TensorBoard Logging
            self.writer.add_scalar('Loss/Actor', actor_loss.item(), self.global_step)
            self.writer.add_scalar('Loss/Critic', critic_loss.item(), self.global_step)
            self.writer.add_scalar('Loss/Total', total_loss.item(), self.global_step)
            self.writer.add_scalar('Entropy', entropy.item(), self.global_step)

            # Log weights and gradients
            for name, param in self.actor.named_parameters():
                self.writer.add_histogram(f'Actor/weights/{name}', param, self.global_step)
                if param.grad is not None:
                    self.writer.add_histogram(f'Actor/gradients/{name}', param.grad, self.global_step)
            
            for name, param in self.critic.named_parameters():
                self.writer.add_histogram(f'Critic/weights/{name}', param, self.global_step)
                if param.grad is not None:
                    self.writer.add_histogram(f'Critic/gradients/{name}', param.grad, self.global_step)

            # Backpropagation
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)

            self.actor_optimizer.step()
            self.critic_optimizer.step()

            # Log losses
            self.log_file.write(f"Epoch {epoch}: Actor Loss={actor_loss.item()}, "
                                f"Critic Loss={critic_loss.item()}, "
                                f"Total Loss={total_loss.item()}\n")
            self.log_file.flush()  # Ensure immediate writing

            # Increment global step
            self.global_step += 1

        # Step learning rate schedulers
        self.actor_scheduler.step()
        self.critic_scheduler.step()

    def choose_action(self, state):
        # Ensure state is a numpy array with correct shape
        state = np.array(state).reshape(1, HEIGHT, WIDTH)
        
        # Convert to tensor with correct shape
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(1)
        
        with torch.no_grad():
            action_probs = self.actor(state_tensor, mode='actor')
            action_probs = action_probs.cpu().numpy().squeeze()
            
            # Sample action based on probabilities
            action = np.random.choice(range(self.action_dim), p=action_probs)
            log_prob = np.log(action_probs[action] + 1e-10)
        
        return action, log_prob

    def store_data(self, state, action, reward, next_state, done, log_prob):
        # Ensure state and next_state are 2D arrays
        if isinstance(state, torch.Tensor):
            state = state.squeeze().cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.squeeze().cpu().numpy()
            
        # Ensure correct shape
        state = np.array(state).reshape(HEIGHT, WIDTH)
        next_state = np.array(next_state).reshape(HEIGHT, WIDTH)
        
        self.replay_buffer.append((state, action, reward, next_state, done, log_prob))

    def save_model(self):
        """
        Save the actor and critic model weights to the specified directory.
        """
        # Ensure the directory exists
        os.makedirs(self.model_file, exist_ok=True)
        
        # Save actor and critic
        actor_path = os.path.join(self.model_file, "actor.pth")
        critic_path = os.path.join(self.model_file, "critic.pth")
        
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        
        print(f"Actor model saved to {actor_path}")
        print(f"Critic model saved to {critic_path}")

    def load_model(self):
        """Load model weights"""
        self.actor.load_state_dict(torch.load(os.path.join(self.model_file, "actor.pth")))
        self.critic.load_state_dict(torch.load(os.path.join(self.model_file, "critic.pth")))
        print(f"Models loaded from {self.model_file}")

    def __del__(self):
        """Ensure log file is closed"""
        if hasattr(self, 'log_file') and hasattr(self.log_file, 'close'):
            self.log_file.close()
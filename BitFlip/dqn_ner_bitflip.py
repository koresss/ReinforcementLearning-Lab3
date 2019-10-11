import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
batch_size = 32

buffer_limit = 5000

class BitFlipEnv():
    
    def __init__(self, n = 8):
        '''
        Setup the environment with a init state and target state
        The init and target stae should not be equal
        '''
        self.n = n
        self.init_state = np.random.randint(2, size=n)
        self.target_state = np.random.randint(2, size=n)
        while np.array_equal(self.init_state, self.target_state):
            self.target_state =np.random.randint(2, size=n)
        self.curr_state = self.init_state.copy()
        
    def step(self, action):
        '''
        Take a step, i.e. flip the bit specified by the position action
        Return the next state and the reward 
        Reward is 0 if the target state is reached
        Otherwise reward is -1
        '''
        self.curr_state[action] = 1 - self.curr_state[action]
        if np.array_equal(self.curr_state, self.target_state):
            return self.curr_state.copy(), 0, True
        else:
            return self.curr_state.copy(), -1, False
        
    def reset(self):
        '''
        Reset the bit flip environment
        '''
        self.init_state = np.random.randint(2, size=self.n)
        self.target_state = np.random.randint(2, size=self.n)
        while np.array_equal(self.init_state, self.target_state):
            self.target_state = np.random.randint(2, size=self.n)
        self.curr_state = self.init_state.copy()


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
        
        
    # Normal Experience Replay
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)    

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self, state_size, hidden_units, no_actions):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_units)
        self.fc2 = nn.Linear(hidden_units, no_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, len(out)-1)
        else:
            return out.argmax().item()


def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

STATE_SIZE = 8
HIDDEN_SIZE = 256
ACTION_NO = 8
num_episodes = 3000
episode_length = 8
def main():
    env = BitFlipEnv()
    q = Qnet(STATE_SIZE, HIDDEN_SIZE, ACTION_NO)
    q_target = Qnet(STATE_SIZE, HIDDEN_SIZE, ACTION_NO)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 100
    score = 0.0
    success = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(num_episodes):
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))  # Linear annealing from 8% to 1%
        env.reset()
        s = env.curr_state.copy()

        for t in range(episode_length):
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, done = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r / 100.0, s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                success += 1
                break

        if memory.size() > 2000:
            train(q, q_target, memory, optimizer)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print("# of episode :{}, avg score : {:.1f}, success rate : {:.1f}%, epsilon : {:.1f}%".format(
                n_epi, score / print_interval, success / print_interval * 100, epsilon * 100))
            score = 0.0
            success = 0.0
    


if __name__ == '__main__':
    main()

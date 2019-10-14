# REFERENCE: https://github.com/seungeunrho/minimalRL "
import gym
import collections
import random

import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple
#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 50000
batch_size    = 32


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        last_transition = self.buffer[-1]
        mini_batch = random.sample(self.buffer, n-1)
        mini_batch.append(last_transition)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])      
       
        #print(goal_lst)
        #print(goal_lst)
        return torch.stack(s_lst), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 5, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(5*20*16, 256)
        self.fc2 = nn.Linear(256, 3)

    def forward(self, x):       
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 5 * 20 *16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
      
    def sample_action(self, obs, epsilon):
        #print('obs',obs)       
        out = self.forward(obs.unsqueeze(0).unsqueeze(0).float())
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,2)
        else : 
            return out.argmax().item()
            
def train(q, q_target, memory, optimizer):
    #for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)       
        s_prime=s_prime.to(dtype=torch.float64)
        #print(s_prime)
        #print(goal)
        q_out = q(s.unsqueeze(1).float())
        q_a = q_out.gather(1,a)     
        max_q_prime = q_target(s_prime.unsqueeze(1).float()).to(dtype=torch.float).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()        
        loss.backward()
        optimizer.step()

def cer_cube(n_episodes_, batch_size_, buf_size_):
    global buffer_limit
    buffer_limit = buf_size_
    global batch_size
    batch_size = batch_size_

    HindsightTransition = namedtuple('HindsightTransition', ('state', 'action', 'next_state', 'reward'))
    env = gym.make('CubeCrashSparse-v0')
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()
    n_success = 0
    print_interval = 100
    score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    succes_rate = []
   
    for n_epi in range(n_episodes_):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
        s = torch.tensor(np.mean(env.reset(),axis = 2))        
        transitions = []
       
        for t in range(600):
            a = q.sample_action(s, epsilon)      
            s_prime, r, done, info = env.step(a)
            s_prime = np.mean(s_prime,axis = 2)
            done_mask = 0.0 if done else 1.0
            
            memory.put((s,a,r/1.0,s_prime, done_mask))
            transitions.append(HindsightTransition(s, a, s_prime, r))
            s = torch.tensor(s_prime)
            score += r           
            if done:
                if r == 1:                    
                    n_success += 1                  
                break
       
            if memory.size()>2000:
                train(q, q_target, memory, optimizer)

        if n_epi%print_interval==0 and n_epi!=0:            
            q_target.load_state_dict(q.state_dict())
            succes_rate.append(n_success/print_interval)
            print('# of episode : {}, avg score : {:.1f}, success rate : {:.1f}%, buffer size : {}'.format(n_epi, score/print_interval , n_success/print_interval * 100, memory.size()))
            n_success = 0.0
            score = 0.0
            
    env.close()
    return succes_rate
    
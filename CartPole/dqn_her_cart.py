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
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst,goal_lst = [], [], [], [], [],[]
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask,goal = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])
            goal_lst.append(goal)
        #print(goal_lst)
        #print(goal_lst)
        return torch.stack(s_lst), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst), torch.stack(goal_lst)
    
    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(8, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
      
    def sample_action(self, obs, epsilon,goal):
        #print('obs',obs)
        state_goal = torch.cat((obs, goal)).to(dtype=torch.float)
        out = self.forward(state_goal)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else : 
            return out.argmax().item()
            
def train(q, q_target, memory, optimizer):
    for i in range(30):
        s,a,r,s_prime,done_mask,goal = memory.sample(batch_size)
        goal=goal.to(dtype=torch.float64)
        s_prime=s_prime.to(dtype=torch.float64)
        #print(s_prime)
        #print(goal)
        q_out = q(torch.cat((s, goal), 1).to(dtype=torch.float))
        q_a = q_out.gather(1,a)
        max_q_prime = q_target(torch.cat((s_prime, goal), 1).to(dtype=torch.float)).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def her_cart(n_episodes_, batch_size_, buf_size_):
    global buffer_limit
    buffer_limit = buf_size_
    global batch_size
    batch_size = batch_size_

    HindsightTransition = namedtuple('HindsightTransition', ('state', 'action', 'next_state', 'reward'))
    env = gym.make('CartPole-v1')
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    success = 0
    success_rate = []
    score_rate = []
    theta = 0.0001
    print_interval = 100
    score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(n_episodes_):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
        s = torch.tensor(env.reset())
        goal=torch.tensor(env.reset())
        transitions = []
        reward = 0

        for t in range(600):
            a = q.sample_action(s, epsilon,goal)      
            s_prime, r, done, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s,a,r/100.0,s_prime, done_mask,goal))
            transitions.append(HindsightTransition(s, a, s_prime, r))
            s = torch.tensor(s_prime)

            score += r
            reward += r
            if done:
                if reward == 500:
                    success += 1
                break         
        
        
        for i in range(len(transitions)):
            hindsight_goal_state = transitions[np.random.randint(i,len(transitions))].state            
            if np.linalg.norm(torch.tensor(transitions[i].next_state) - hindsight_goal_state) < theta:
                memory.put((transitions[i].state, transitions[i].action,torch.tensor([2.0]), transitions[i].next_state,0.0, hindsight_goal_state))
            else:
                memory.put((transitions[i].state, transitions[i].action,transitions[i].reward, transitions[i].next_state,0.0, hindsight_goal_state))
            
        if memory.size()>2000:
            train(q, q_target, memory, optimizer)

        if n_epi%print_interval==0 and n_epi!=0:            
            q_target.load_state_dict(q.state_dict())
            score_rate.append(score / print_interval)
            success_rate.append(success / print_interval)
            print("# of episode :{}, avg score : {:.1f}, success rate : {:.1f}%, epsilon : {:.1f}%".format(
                n_epi, score / print_interval, success / print_interval * 100, epsilon * 100))
            score = 0.0
            success = 0
    env.close()
    return success_rate, score_rate

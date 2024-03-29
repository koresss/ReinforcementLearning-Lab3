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
import matplotlib.pyplot as plt



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 20000
batch_size = 32

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
               torch.tensor(r_lst), torch.stack(s_prime_lst), \
               torch.tensor(done_mask_lst), torch.stack(goal_lst)
    
    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.conv1 = torch.nn.Conv2d(2, 5, kernel_size=3, stride=1, padding=1)
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
      
    def sample_action(self, obs, epsilon,goal):
        #print('obs',obs)
        state_goal = torch.stack((obs, goal)).unsqueeze(0).to(dtype=torch.float)
        out = self.forward(state_goal)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,2)
        else : 
            return out.argmax().item()
            
def train(q, q_target, memory, optimizer):
    for i in range(30):
        s,a,r,s_prime,done_mask,goal = memory.sample(batch_size)        
        r = r.to(device)
        a = a.to(device)
        done_mask = done_mask.to(device)
        #print(s_prime)
        #print(goal)
        q_out = q(torch.stack((s, goal),1).to(dtype=torch.float))
        q_a = q_out.gather(1,a)     
        max_q_prime = q_target(torch.stack((s_prime, goal),1).to(dtype=torch.float)).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        #print(loss)
        loss.backward()
        optimizer.step()
        
        
def get_goal(s):
    goal = s.clone()
    goal[goal<255] = 0.0   
    
    start_idx = (goal[-1,:] < 255).cpu().numpy().argmax() + random.randint(0,6)    
    goal[-1,start_idx:start_idx+3] = 85.0
    
    #plt.imshow(s.cpu())
    #plt.figure()
    #plt.imshow(goal.cpu())
    return goal

def her_cube(n_episodes_, batch_size_, buf_size_):
    global buffer_limit
    buffer_limit = buf_size_
    global batch_size
    batch_size = batch_size_
    
    HindsightTransition = namedtuple('HindsightTransition', ('state', 'action', 'next_state', 'reward', 'mask'))
    env = gym.make('CubeCrashSparse-v0')
    q = Qnet().to(device)
    q_target = Qnet().to(device)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()
    n_success = 0
    print_interval = 100
    score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    succes_rate = []
    
    
    for n_epi in range(n_episodes_):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
        s = torch.tensor(np.mean(env.reset(),axis = 2)).to(device)           
        transitions = []
        goal = get_goal(s)
       
        for t in range(600):
            a = q.sample_action(s, epsilon,goal)      
            s_prime, r, done, info = env.step(a)
            s_prime = torch.tensor(np.mean(s_prime,axis = 2)).to(device)
            done_mask = 0.0 if done else 1.0            
            
            transitions.append(HindsightTransition(s, a, s_prime, r, done_mask))
            s = s_prime.clone()
            score += r           
            if done:                  
                if r == 1:     
                    goal = s_prime.clone()
                    n_success += 1  
                    
                for i in range(len(transitions)):
                    memory.put((transitions[i].state, transitions[i].action, transitions[i].reward, transitions[i].next_state, transitions[i].mask, goal))
                    
                    #hindsight_goal_state = transitions[np.random.randint(i,len(transitions))].next_state
                    hindsight_goal_state = transitions[-1].next_state
                    if np.array_equal((transitions[i].next_state).cpu(), hindsight_goal_state.cpu()):
                        memory.put((transitions[i].state, transitions[i].action, 1.0, transitions[i].next_state, 0.0, hindsight_goal_state))
                    else:
                        memory.put((transitions[i].state, transitions[i].action, 0.0, transitions[i].next_state, 1.0, hindsight_goal_state))
                               
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

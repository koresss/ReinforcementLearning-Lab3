#ref https://github.com/viraat/hindsight-experience-replay
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

torch.uint8 = torch.bool

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BitFlipEnv():
    
    def __init__(self, n = 8):
        '''
        Setup the environment with a init state and target state
        The init and target stae should not be equal
        '''
        self.n = n
        self.init_state = torch.randint(2, size=(n,))
        self.target_state = torch.randint(2, size=(n,))
        while np.array_equal(self.init_state, self.target_state):
            self.target_state = torch.randint(2, size=(n,))
        self.curr_state = self.init_state.clone()
        
    def step(self, action):
        '''
        Take a step, i.e. flip the bit specified by the position action
        Return the next state and the reward 
        Reward is 0 if the target state is reacher
        Otherwise reward is -1
        '''
        self.curr_state[action] = 1 - self.curr_state[action]
        if np.array_equal(self.curr_state, self.target_state):
            return self.curr_state.clone(), 0
        else:
            return self.curr_state.clone(), -1
        
    def reset(self):
        '''
        Reset the bit flip environment
        '''
        self.init_state = torch.randint(2, size=(self.n,))
        self.target_state = torch.randint(2, size=(self.n,))
        while np.array_equal(self.init_state, self.target_state):
            self.target_state = torch.randint(2, size=(self.n,))
        self.curr_state = self.init_state.clone()
        
Transition = namedtuple('Transition', 
                       ('state', 'action', 'next_state', 'reward', 'goal'))

class ReplayMemory(object):
    
    def __init__(self, capacity = 1e5):
        self.capacity = capacity
        self.memory = []
    
    def push(self, *args):
        """Saves a transition which should contain:
        - current state
        - action taken
        - next state
        - reward obtained
        - goal state"""
        self.memory.append(Transition(*args))
        if len(self.memory) > self.capacity:
#             print('!!!!!memory capacity exceeded!')
            del self.memory[0]

    def sample(self, batch_size):
        """
        Returns batch_size number of samples from the replay memory
        """
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    

NUM_BITS = 8
HIDDEN_SIZE = 256
class FNN(nn.Module):
    
    def __init__(self):
        super(FNN, self).__init__()
        self.ln1 = nn.Linear(NUM_BITS*2, HIDDEN_SIZE)
        self.ln2 = nn.Linear(HIDDEN_SIZE, NUM_BITS)
        
    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = self.ln2(x)
        return x
BATCH_SIZE = 128 # batch size for training
GAMMA = 0.999 # discount factor
EPS_START = 0.95 # eps greedy parameter
EPS_END = 0.05
TARGET_UPDATE = 50 # number of epochs before target network weights are updated to policy network weights
steps_done = 0 # for decayin eps
policy_net = FNN().to(device)
target_net = FNN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(1e6)


def select_action(state, goal, greedy=False):
    '''
    Select an action given the state and goal acc to policy_net
    - use eps_greedy policy when greedy=False
    - use greedy policy when greedy=True
    Returns action taken which is from range(0, n-1)
    '''
    global steps_done
    sample = random.random()
    state_goal = torch.cat((state, goal)).to(device,dtype=torch.float)
    #print(state_goal)

    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
#     if steps_done % 1000 == 0:
#         print('Steps done: {} Epsilon threshold: {}'.format(steps_done, eps_threshold))
    if greedy:
        with torch.no_grad():
            return policy_net(state_goal).argmax().view(1,1)
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state_goal).argmax().view(1,1)
    else: 
        return torch.tensor([[random.randrange(NUM_BITS)]], device=device, dtype=torch.long)
    
    
def optimize_model():
    '''
    optimize the model, i.e. perform one step of Q-learning using BATCH_SIZE number of examples
    '''
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, 
                                           batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.stack([s for s in batch.next_state 
                                      if s is not None])
    
    # extract state, action, reward, goal from randomly sampled transitions
    state_batch = torch.stack(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    goal_batch = torch.stack(batch.goal)
    
    # concatenate state and goal for network input
    state_goal_batch = torch.cat((state_batch, goal_batch), 1).to(dtype=torch.float)
    non_final_next_states_goal = torch.cat((non_final_next_states, goal_batch), 1).to(dtype=torch.float)
    
    # get current state action values 
    state_action_values = policy_net(state_goal_batch).gather(1, action_batch)
    
    # get next state values according to target_network
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states_goal).max(1)[0].detach()
    
    # calculate expected q value of current state acc to target_network
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.float()
    
    # find huber loss using curr q-value and expected q-value
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
 
    
num_episodes = 3000    
EPS_DECAY = num_episodes * NUM_BITS * 0.05 # decay rate  
def her_bitflip():        
    HindsightTransition = namedtuple('HindsightTransition', 
                           ('state', 'action', 'next_state', 'reward'))
    CHECK_RATE = 100 # evaluate success on the last 100 episodes
    
    episode_length = 8    
    env = BitFlipEnv(NUM_BITS)
    success = 0.0
    score = 0.0
    episodes = [] # every 100 episodes
    success_rate = [] # append success rate of last 100 episodes
    # train the network
    for i_episode in range(num_episodes):
        env.reset()
        state = env.init_state
        goal = env.target_state
        transitions = []
        episode_success = False
        # for bit length
        for t in range(episode_length):
            
            if episode_success:
                continue
            
            action = select_action(state, goal)
            next_state, reward = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            
            # add transition to replay memory
            memory.push(state.to(device), action, next_state.to(device), reward, goal.to(device))
            
            # store transition without goal state for hindsight 
            transitions.append(HindsightTransition(state, action, next_state, reward))
    
            state = next_state
            score += reward.item()
            #optimize_model()
            if reward == 0:
                if episode_success:
                    continue
                else:
                    episode_success = True
                    success += 1
        
        # add hindsight transitions to the replay memory
        if not episode_success:
            # failed episode store the last visited state as new goal
            new_goal_state = state.clone()
            if not np.array_equal(new_goal_state, goal):
                for i in range(NUM_BITS):
                    # if goal state achieved
                    if np.array_equal(transitions[i].next_state, new_goal_state):
                        memory.push(transitions[i].state.to(device), transitions[i].action.to(device), transitions[i].next_state.to(device), torch.tensor([0]).to(device), new_goal_state.to(device))
                        #optimize_model()
                        break
    
                    memory.push(transitions[i].state.to(device), transitions[i].action.to(device), transitions[i].next_state.to(device), transitions[i].reward.to(device), new_goal_state.to(device))
        
        for i in range(10):
            optimize_model()
    
        # update the target networks weights 
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
    
        if i_episode % CHECK_RATE == 0:
            print('# of episode : {}, avg score : {:.1f}, success rate : {:.1f}%,'.format(i_episode, score/CHECK_RATE , success/CHECK_RATE * 100))
            success_rate.append(success/CHECK_RATE)
            episodes.append(i_episode)
            success = 0.0
            score = 0.0

    return success_rate

if __name__ == '__main__':
    her_bitflip()

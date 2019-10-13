# took a look at https://github.com/rlcode/per/blob/master/prioritized_memory.py

from SumTree import SumTree

buffer_limit = 50000
import collections
import heapq
import random
import gym
import collections
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BUDGET = 1000000
BETA = 1
K = 200
learning_rate = 0.0005
GAMMA = 0.98
epsilon = 0.05
k = 32  # batch size
alpha = 0.7  # https://arxiv.org/pdf/1511.05952.pdf


class Transition:
    def __init__(self, s, a, r, s_t, done, gamma):
        self.s = s
        self.a = a
        self.r = r
        self.s_t = s_t
        self.done = done
        self.gamma = gamma


class PriorityQueueBuffer:
    # rank priority priority buffer
    def __init__(self, capacity=20000):
        self.buffer = SumTree(capacity=capacity)
        self.capacity = capacity
        self.length = 0

    def put(self, transition, priority):
        p = (abs(priority) + epsilon) ** alpha  # alpha factor formula
        self.buffer.add(p, transition)
        if self.length < self.capacity:
            self.length += 1

    def update_indices(self, indices, td_errors):
        for i in range(k):
            err = td_errors[i].item()
            idx = indices[i]
            self.buffer.update(idx, err)

    def _decode(self, transitions):
        s_lst, a_lst, r_lst, s_t_lst, g_lst, done = [], [], [], [], [], []
        for transition in transitions:
            s_lst.append(transition.s)
            a_lst.append(transition.a)
            r_lst.append(transition.r)
            s_t_lst.append(transition.s_t)
            g_lst.append(transition.gamma)
            done.append(transition.done)

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_t_lst, dtype=torch.float), torch.tensor(g_lst,
                                                                                           dtype=torch.float), torch.tensor(
            done, dtype=torch.float)

    def get_is_weights(self, probs):
        is_weights = np.power(torch.tensor(probs) * self.length, -BETA)
        is_weights /= torch.max(is_weights)
        return is_weights

    # Prioritised Experience Replay
    def sample(self, n):
        p_total = self.buffer.total()
        probs = []
        transitions = []
        indexes = []
        # dive into n ranges
        segment = p_total / n

        for i in range(n):
            left = segment * i
            right = segment * i + segment
            sampled_number = random.uniform(left, right)
            idx, p, transition = self.buffer.get(sampled_number)
            probs.append(p)
            transitions.append(transition)
            indexes.append(idx)

        weights = self.get_is_weights(probs)
        transitions = self._decode(transitions)
        return transitions, weights, indexes


class Qnet(nn.Module):
    def __init__(self, state_size, no_actions):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, no_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item()


def train(q, q_target, memory, optimizer):
    transitions, weights, indices = memory.sample(k)

    s = transitions[0]
    a = transitions[1].view(-1, 1)
    r = transitions[2].view(-1, 1)
    s_t = transitions[3]
    g = transitions[4].view(-1, 1)
    done_mask = transitions[5].view(-1, 1)

    # next state
    max_q_prime = q_target(s_t).max(1)[0].unsqueeze(1)

    # current state
    q_out = q(s)
    q_a = q_out.gather(1, a)

    td_error = r + g * max_q_prime * done_mask - q_a

    memory.update_indices(indices, td_error)

    loss = (torch.FloatTensor(weights) * F.smooth_l1_loss(q_a, td_error)).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def policy(q, s, env, epsilon):
    if random.random() < 1 - epsilon:
        return env.action_space.sample()
    actions = q(s)
    return np.argmax(actions)


def main():
    epsilon = 0.05
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    q = Qnet(state_size, action_size)
    q_target = Qnet(state_size, action_size)
    q_target.load_state_dict(q.state_dict())
    memory = PriorityQueueBuffer()

    print_interval = 1
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    s = env.reset()
    a = q.sample_action(torch.from_numpy(s).float(), epsilon)
    priorities = list()
    priorities.append(1)
    for t in range(BUDGET):
        epsilon = max(0.01, 0.08 - 0.01 * (t / 20000))  # Linear annealing from 8% to 1%

        s_t, r, done, info = env.step(a)
        done_mask = 0. if done else 1.

        # calculating priority
        q_a = q(torch.from_numpy(s).float())
        old_val = q_a[a]

        p = r
        if not done:
            max_q_prime = q_target(torch.from_numpy(s_t).float()).max()
            p += GAMMA * max_q_prime

        p -= old_val
        p = abs(p)

        # put in memory with transition priority
        transition = Transition(s, a, r, s_t, done_mask, GAMMA)
        memory.put(transition, p)

        if (t + 1) % K == 0:
            train(q, q_target, memory, optimizer)

        score += r
        # picking the next action
        if done and print_interval >100:
            s = env.reset()

            q_target.load_state_dict(q.state_dict())
            print("# of T :{}, avg score : {:.1f}, buffer size : {}, epsilon : {:.1f}%".format(
                t, score / print_interval, memory.length, epsilon * 100))
            score = 0.0
            print_interval =0

        a = q.sample_action(torch.from_numpy(s).float(), epsilon)
        print_interval +=1

    env.close()


if __name__ == '__main__':
    main()

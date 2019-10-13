import random

from SumTree import SumTree
import numpy as np
import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

batch_size = 32
gamma = 0.98
learning_rate = 0.0005
train_interval = 10
BUFFER_SIZE = 10000


# took a look at https://github.com/rlcode/per/blob/master/prioritized_memory.py
# took a look at https://github.com/rlcode/per/blob/master/cartpole_per.py

class PriorityQueueBuffer:
    # rank priority priority buffer
    def __init__(self, capacity=BUFFER_SIZE, epsilon=1e-5, alpha=0.7, beta=1):
        self.buffer = SumTree(capacity=capacity)
        self.capacity = capacity
        self.length = 0
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta

    def put(self, transition, priority):
        p = (abs(priority) + self.epsilon) ** self.alpha  # alpha factor formula
        self.buffer.add(p, transition)
        if self.length < self.capacity:
            self.length += 1

    def update_indices(self, indices, td_errors):
        size = len(indices)
        for i in range(size):
            err = td_errors[i].item()
            idx = indices[i]
            self.buffer.update(idx, err)

    def _decode(self, transitions):
        s_lst, a_lst, r_lst, s_t_lst, done = [], [], [], [], []
        for transition in transitions:
            s_lst.append(transition[0])
            a_lst.append(transition[1])
            r_lst.append(transition[2])
            s_t_lst.append(transition[3])
            done.append(transition[4])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_t_lst, dtype=torch.float), torch.tensor(
            done, dtype=torch.float)

    def get_is_weights(self, probs):
        is_weights = torch.pow(torch.tensor(probs) * self.length, -self.beta)
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

    def size(self):
        return self.length


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
            return random.randint(0, 1)
        else:
            return out.argmax().item()


def get_priority(s, a, r, s_prime, done_mask, q, q_target):
    q_a = q(torch.torch.from_numpy(s).float())
    value = q_a[a]

    if done_mask == 1:
        q_target_max = q_target(torch.from_numpy(s_prime).float()).max()
        target = r + gamma * q_target_max
    else:
        target = r

    return abs(target - value)


def train(q, q_target, memory, optimizer):
    transitions, weights, indices = memory.sample(batch_size)
    s, a, r, s_prime, done_mask = transitions

    q_out = q(s)
    q_a = q_out.gather(1, a.view(-1, 1)).view(-1, 1)

    q_a_prime = q(s_prime)
    indices = q_a_prime.max(1)[1].view(-1, 1)

    max_q_prime = q_target(s_prime).gather(1, indices)
    target = r.view(-1, 1) + gamma * max_q_prime * done_mask.view(-1, 1)

    # might have a problem
    loss = (torch.FloatTensor(weights) * F.mse_loss(q_a, target)).mean()

    memory.update_indices(indices, torch.abs(target - q_a))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def priority_cart():
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    q = Qnet(state_size, 256, action_size).cuda()
    q_target = Qnet(state_size, 256, action_size).cuda()
    q_target.load_state_dict(q.state_dict())
    memory = PriorityQueueBuffer()

    success = 0
    success_rate = []
    score_rate = []
    print_interval = 100
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(10000):
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))  # Linear annealing from 8% to 1%
        s = env.reset()
        reward = 0

        for t in range(600):
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, done, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            p = get_priority(s, a, r, s_prime, done_mask, q, q_target)
            memory.put((s, a, r / 100.0, s_prime, done_mask), p)
            s = s_prime

            score += r
            reward += r

            # if train_counter % train_interval == 0:
            #     train_counter += 1

            if done:
                if reward == 499:
                    success += 1
                break
            if memory.size() >= BUFFER_SIZE:
                train(q, q_target, memory, optimizer)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            score_rate.append(score / print_interval)
            success_rate.append(success / print_interval)
            print("# of episode :{}, avg score : {:.1f}, success rate : {:.1f}%, epsilon : {:.1f}%".format(
                n_epi, score / print_interval, success / print_interval * 100, epsilon * 100))
            score = 0.0
            success = 0

    env.close()
    return success_rate, score_rate


if __name__ == '__main__':
    priority_cart()

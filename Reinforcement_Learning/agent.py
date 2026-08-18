import random

import torch.nn as nn
import torch
import numpy as np
import torch.optim as optim 

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


#define the neural network that your DQN agent uses to estimate the Q-value of each action
class DuelingTransformerQNetwork(nn.Module):
    def __init__(self, length, num_actions=5, d_model=128, nhead=4, num_layers=2, dim_feedforward=256):
        '''
            length: length of the protein sequence
            num_actions: number of possible actions (5 in this case) up, down, left, right, forward (move in same dir)
            d_model: dimension of the model (embedding size) - each residue is represented as a vector of size d_model
            nhead: number of heads in the multihead attention models
            num_layers: number of transformer layers
            dim_feedforward: dimension of the feedforward network in the transformer

            Its job is to take the current state x and output a Q-value for each possible action.
        
        '''
        super().__init__()
        self.length, self.num_actions = length, num_actions
        self.d_model, self.dim_feedforward = d_model, dim_feedforward

        #Splitting the 128-dimensional embedding
        #64 (Coordinates) + 32 (type) + 32 (position) = 128
        coord_dim = d_model // 2
        type_dim = d_model // 4
        index_dim = d_model - coord_dim - type_dim

        self.coord_proj = nn.Linear(3, coord_dim) #This lets the neural network learn a useful representation of the spatial coordinates.
        self.type_emb = nn.Embedding(3, type_dim)   # 0=P, 1=H, 2=unfilled
        self.index_proj = nn.Linear(1, index_dim) #The fifth input feature is apparently a normalized residue/index position:

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) #Let the Transformer use this special token to accumulate information about the entire sequence. It is like compressed token that represents entire sequence. It is used to get the final output of the Transformer.
        self.pos_encoder = PositionalEncoding(d_model) 
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward) #This is where the Transformer architecture is defined. It consists of multiple layers of self-attention and feedforward networks, allowing the model to capture complex dependencies in the input sequence.
        self.encoder = nn.TransformerEncoder(layer, num_layers)

        self.value_fc = nn.Linear(d_model, 1)
        self.advantage_fc = nn.Linear(d_model, num_actions)

    def forward(self, x):
        B = x.size(0) #B is the batch size.
        seq = x.view(B, self.length, 5) #resize to proper shape
        coords, aatype, index_norm = seq[..., :3], seq[..., 3].long(), seq[..., 4:5] #extarct 5 values
        aatype = torch.where(aatype == -1, torch.full_like(aatype, 2), aatype) #handle unfilled residues by replacing -1 with 2 (the index for unfilled residues in the embedding layer)
        
        emb = torch.cat(
                    [self.coord_proj(coords), self.type_emb(aatype), self.index_proj(index_norm)], dim=2
                ) #Each residue gets three embeddings.
        emb = torch.cat([self.cls_token.expand(B, -1, -1), emb], dim=1)
        emb = self.pos_encoder(emb).transpose(0, 1)
        cls_rep = self.encoder(emb)[0]  # (B, d_model)
        '''
            V-value V(s): How good is the current state overall?
            Advantage A(s,a): How much better is action a than the other actions in this state?
            Q-value Q(s,a): How good is taking action a from state s, combining both.
        '''
        value = self.value_fc(cls_rep)
        advantage = self.advantage_fc(cls_rep)
        return value + advantage - advantage.mean(dim=1, keepdim=True)



# PrioritizedReplayMemory → stores and selects past experiences.
# DQNAgent → uses those experiences to train the Transformer DQN.

'''
The replay memory stores experiences of the form:

(state, action, reward, next_state, done)

For example:

s  = current protein state
a  = action 2
r  = +1 reward
ns = new protein state
d  = False

'''
class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_increment=1e-6):
        self.capacity = capacity
        self.alpha = alpha #Controls how strongly prioritization is used. alpha = 0 → completely random sampling larger alpha → more preference for high-priority experiences
        self.beta = beta_start #beta controls how strongly we correct the bias caused by prioritized sampling.
        self.beta_increment = beta_increment
        self.eps = 1e-6 #A tiny number used so priority never becomes exactly zero.
        self.memory = [] #Stores the actual transitions.
        self.priorities = np.zeros(capacity, dtype=np.float32) #Stores a priority value for every experience.
        self.pos = 0 #Keeps track of where the next experience should be inserted.

    def push(self, s, a, r, ns, d):
        max_prio = self.priorities[: len(self.memory)].max() if self.memory else 1.0 
        if len(self.memory) < self.capacity: #If memory isn't full, append it.
            self.memory.append((s, a, r, ns, d))
        else: #Memory full so replace the oldest experience with the new one. This is a circular buffer.
            self.memory[self.pos] = (s, a, r, ns, d)
        self.priorities[self.pos] = max_prio #The new experience receives the maximum priority.
        self.pos = (self.pos + 1) % self.capacity

    '''
        sample() converts priorities into sampling probabilities, 
        then randomly selects a batch of experiences with higher-priority experiences being more likely to be chosen
    '''
    def sample(self, batch_size):
        prios = self.priorities[: len(self.memory)]
        probs = prios ** self.alpha
        probs /= probs.sum()
    
        indices = np.random.choice(len(self.memory), batch_size, p=probs) #randomly select a batch of experiences based on their probabilities
        weights = (len(self.memory) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
    
        s, a, r, ns, d = zip(*(self.memory[i] for i in indices))
        return (
                torch.FloatTensor(np.array(s)), torch.LongTensor(a),
                torch.FloatTensor(r), torch.FloatTensor(np.array(ns)),
                torch.BoolTensor(d), torch.FloatTensor(weights), indices,
            )
    def update_priorities(self, indices, td_errors):
            self.priorities[indices] = np.abs(td_errors) + self.eps
    
    def __len__(self):
            return len(self.memory)


class DQNAgent:
    def __init__(self, length, num_actions=5, gamma=0.99, lr=1e-3, batch_size=64,
                 memory_size=50000, target_update_freq=1000, device="cpu", **net_kwargs):
        self.num_actions, self.gamma, self.batch_size = num_actions, gamma, batch_size
        self.device, self.target_update_freq = device, target_update_freq
        '''
            both networks have the same architecture, but they are used differently.
            Policy network = you are the student who is constantly learning and changing their answers.
            Target network = an answer key that stays fixed for a while.
        
            The policy network is trained continuously:
                loss.backward()
                optimizer.step()

            The target network is NOT directly trained.
                Instead, every 1000 steps, you copy the policy network's weights into it:
                self.target_net.load_state_dict(
                        self.policy_net.state_dict()
                    )
        The two networks are structurally the same, but the policy network learns, while the target network provides a stable reference for learning.
        Both are learnign but at differnt rates. The policy network is updated every step, while the target network is updated less frequently. This helps stabilize learning by providing a more consistent target for the policy network to learn from.
        
        '''
        self.policy_net = DuelingTransformerQNetwork(length, num_actions, **net_kwargs).to(device) #because this network outputs the Q-values for each action, it is called the policy network. It is used to select actions during training and evaluation.
        self.target_net = DuelingTransformerQNetwork(length, num_actions, **net_kwargs).to(device) #The target network is a copy of the policy network that is updated less frequently. It is used to compute the target Q-values during training, which helps stabilize learning by providing a more consistent target for the policy network to learn from.
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = PrioritizedReplayMemory(memory_size)
        self.steps_done = 0

    def select_action(self, env, state, epsilon):
        '''
            With probability epsilon, explore by choosing a random valid action; 
            otherwise, exploit by choosing the valid action with the highest Q-value predicted by the policy network.
        '''
        valid = env.get_valid_actions() #range of 5 valid actionsa
        if random.random() < epsilon:  #If epsilon = 0.1, approximately 10% of the time the agent chooses a random valid action.
            valid_idx = np.flatnonzero(valid)
            return np.random.choice(valid_idx) if len(valid_idx) else random.randrange(self.num_actions)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q = self.policy_net(state_t).cpu().numpy().squeeze()
        q[~valid] = -1e9 #Invalid actions are given a huge negative Q-value
        return int(q.argmax())


    def store_transition(self, s, a, r, ns, d):
            self.memory.push(s, a, r, ns, d)

    def update(self):
            '''
                update() is the main learning function. It takes experiences from replay memory, 
                calculates the correct/target Q-values, compares them with the policy network's predictions, 
                and updates the policy network.

                The policy network learns here:
            
            '''
            if len(self.memory) < self.batch_size: #not enough experiences to form a batch, so return None and do not update the policy network
                return None
    
            states, actions, rewards, next_states, dones, weights, idx = self.memory.sample(self.batch_size) #Get a batch of experiences from Prioritized Replay Memory.
            states, next_states = states.to(self.device), next_states.to(self.device)
            actions, rewards = actions.to(self.device), rewards.to(self.device)
            dones, weights = dones.to(self.device), weights.to(self.device)
    
            q_sa = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1) #gather predictions for the actions that were actually taken in the sampled experiences. 
            #This gives us the Q-values predicted by the policy network for the actions that were taken in those states.
    
            with torch.no_grad():
                best_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, best_actions).squeeze(1) #evaluate the actions
                target = rewards + (1 - dones.float()) * self.gamma * next_q
    
            td_error = target - q_sa #calulate the difference between the target Q-values and the policy network's predictions. This is the temporal difference (TD) error, which tells us how much the policy network's predictions differ from what they should be.
            loss = (td_error.pow(2) * weights).mean()
    
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.memory.update_priorities(idx, td_error.detach().cpu().numpy()) 
    
            self.steps_done += 1
            if self.steps_done % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
    
            return loss.item()

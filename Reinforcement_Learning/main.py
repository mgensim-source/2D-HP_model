from datetime import datetime
import logging
import os
import numpy as np
from definitions import ACTIONS, _DIR_TABLE,_NEIGHBOR_OFFSETS
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import csv
import time
from agent import DQNAgent
#1. turn off scalinfg in obs
#2. change residue representation from 64 to 128


class HPProteinFoldingEnv(gym.Env):
    metadata = {"render.modes": ["human"]} #human-readable rendering mode
    '''
    The important methods are:
        __init__() → set up the environment
        reset() → start/restart an episode
        step(action) → perform an action and return the new state + reward
        render() → optionally visualize the environment
    '''
    def __init__(self, sequence):
        super().__init__()
        self.sequence = sequence
        self.length = len(sequence)
        self.radius = self.length // 2
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(
            low=-float(self.radius), high=float(self.radius),
            shape=(self.length * 5,), dtype=np.float32,
        )
        self.reset()

    def reset(self):
        '''
            Define the Starting position of the residues, Eg: here (0,0,0) is first residue and (1,0,0) is second residue.
            And define what the function should return -> 
        '''

        self.positions = [(0, 0, 0), (1, 0, 0)] 
        self.pos_set = set(self.positions)
        self.i = 2                # number of residues placed so far
        self.done = False
        self.broke_turn = False    # first non-forward turn must be "right"
        self.broke_z = False       # first vertical move must be "up"
        return self._obs()

    def get_valid_actions(self):
            """Boolean mask honoring symmetry-breaking + a one-step trap lookahead."""
            mask = np.zeros(5, dtype=bool)
            for a in range(5):
                if not self.broke_turn and a not in (0, 4):
                    continue
                if not self.broke_z and a == 2:
                    continue
    
                next_pos = self._next_position(self.positions, a)
                if not self._in_bounds(next_pos) or next_pos in self.pos_set:
                    continue
    
                if self.i + 1 >= self.length:
                    mask[a] = True
                    continue
    
                temp_positions = self.positions + [next_pos]
                temp_set = self.pos_set | {next_pos}
                if self._has_escape(temp_positions, temp_set):
                    mask[a] = True
            return mask
    
    def _hh_bonds(self):
        """Count non-sequential H-H contacts in O(#H) via neighbor lookups."""
        h_index = {p: i for i, p in enumerate(self.positions) if self.sequence[i] == "H"}
        count = 0
        for p, i in h_index.items():
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                j = h_index.get((p[0] + dx, p[1] + dy, p[2] + dz))
                if j is not None and abs(i - j) > 1:
                    count += 1
        return count // 2  # each bond counted from both ends
    
    
    def _obs(self): 
        '''
        _obs() takes the current protein structure and converts it into a list of numbers that the RL agent can understand 
        ie.Return the current state of the environment
        '''
        obs = np.full((self.length, 5), -1.0, dtype=np.float32) #Create a table with self.length rows and 5 columns, and fill every position with -1.0
        #[x, y, z, H/P, sequence position/coords]
        obs[:, :3] = 0.0 #The :3 means columns 0, 1, 2 all rows as 0.
        obs[:self.i, :3] = np.array(self.positions, dtype=np.float32) #put the current positions of the residues into the first 3 columns of the obs array
        denom = max(self.length - 1, 1)
        obs[:self.i, 4] = np.arange(self.i) / denom #scale the sequence position to be between 0 and 1, and put it in the last column of the obs array - This calculates a value used to scale the residue number.
        return obs.flatten()

    def _forward_vec(self,positions):
            if len(positions) < 2:
                return (1, 0, 0)
            (x1, y1, z1), (x2, y2, z2) = positions[-2], positions[-1]
            return (x2 - x1, y2 - y1, z2 - z1)

    def _next_position(self, positions, action):
            """Single implementation used for real state, lookahead, and trap checks."""
            f = self._forward_vec(positions)
            move = f if action == 0 else _DIR_TABLE[f][ACTIONS[action]]
            x, y, z = positions[-1]
            return (x + move[0], y + move[1], z + move[2])

    def _in_bounds(self, p):
            r = self.radius
            return -r <= p[0] <= r and -r <= p[1] <= r and -r <= p[2] <= r

    def _has_escape(self, positions, pos_set):
            """One-step lookahead: is there at least one legal next move?"""
            for a in range(5):
                nxt = self._next_position(positions, a)
                if self._in_bounds(nxt) and nxt not in pos_set:
                    return True
            return False
    
    def _is_trapped(self):
        return self.i < self.length and not self._has_escape(self.positions, self.pos_set)

    def step(self, action):
        #returns: (observation, reward, done, info)
        if self.done: #check if the episode is already done
            return self._obs(), 0.0, True, {}

        if self.i >= self.length: #self.i tracks now many sequence residues have been placed. If all residues have been placed, the episode is done.
            self.done = True
            return self._obs(), self._hh_bonds(), True, {}

        next_pos = self._next_position(self.positions, action)
        if not self._in_bounds(next_pos) or next_pos in self.pos_set: #Is the new position outside the grid? or Is the position already occupied?
            trapped = self._is_trapped()
            self.done = trapped
            return self._obs(), 0.0, trapped, {}
        self.positions.append(next_pos)
        self.pos_set.add(next_pos)        
        self.i += 1
        '''
            Note: the below is added to avoid symmetrical conformation formations and avoid lot of redundant conformations.
            If the first turn is not broken, then the agent can only go forward. Once the first turn is broken, then the agent can go in any direction. 
            Similarly, if the first z-axis turn is not broken, then the agent can only go in the x-y plane. 
        '''
        if not self.broke_turn and action == 4:
            self.broke_turn = True
        if not self.broke_z and action == 1:
            self.broke_z = True

        if self.i == self.length:
            self.done = True
            return self._obs(), self._hh_bonds(), True, {}

        trapped = self._is_trapped()
        self.done = trapped
        return self._obs(), 0.0, trapped, {}

    def render(self, show_dialog=True, filename=None):
            xs, ys, zs = zip(*self.positions)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(xs, ys, zs, "k-", lw=2)
            colors = ["r" if c == "H" else "b" for c in self.sequence[: len(self.positions)]]
            ax.scatter(xs, ys, zs, c=colors, s=100)
            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
            if filename:
                plt.savefig(filename)
            plt.show() if show_dialog else plt.close()
    
    def close(self):
        plt.close("all")
        
def run_episode(env, agent, epsilon, max_steps, train=True):
    #This runs the agent through one complete episode.
    #Let the agent interact with the environment and collect experiences/rewards for one episode.
    state = env.reset()
    total_reward = 0.0
    for _ in range(max_steps):
        action = agent.select_action(env, state, epsilon)
        next_state, reward, done, _ = env.step(action)
        if train:
            agent.store_transition(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            break
    return total_reward

def evaluate_agent(env, agent, num_episodes=10, max_steps=200):
    agent.policy_net.eval()
    rewards = [run_episode(env, agent, epsilon=0.0, max_steps=max_steps, train=False)
               for _ in range(num_episodes)]
    agent.policy_net.train()
    return float(np.mean(rewards))

def train_dqn(env, agent, run_dir, num_episodes=500, max_steps=200,
              epsilon_start=1.0, epsilon_end=0.01, eval_interval=1000):
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    train_rows, eval_rows = [], []
    best_reward = float("-inf")
    best_eval = float("-inf")
    start = time.perf_counter()

    for ep in range(1, num_episodes + 1):
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-5 * ep / num_episodes) #control the exploration rate, which decreases over time. Initially explores a lot and then uses it increasingly uses what it has learned.
        reward = run_episode(env, agent, epsilon, max_steps, train=True)
        loss = agent.update() or 0.0
        train_rows.append((ep, reward))

        if reward > best_reward:
            best_reward = reward
            env.render(show_dialog=False, filename=os.path.join(run_dir, f"best_ep{ep}.png"))

        if ep % 100 == 0:
            elapsed = time.perf_counter() - start
            logging.info(f"Ep {ep}/{num_episodes}  reward={reward:.1f}  loss={loss:.5f}  "
                         f"eps={epsilon:.3f}  eps/sec={ep/elapsed:.2f}")

        if ep % eval_interval == 0:
            avg_eval = evaluate_agent(env, agent, num_episodes=10, max_steps=max_steps)
            eval_rows.append((ep, avg_eval))
            logging.info(f"[Eval] Ep {ep}: avg_reward={avg_eval:.2f}")

            torch.save(agent.policy_net.state_dict(), os.path.join(ckpt_dir, f"ckpt_ep{ep}.pth"))
            if avg_eval > best_eval:
                best_eval = avg_eval
                torch.save(agent.policy_net.state_dict(), os.path.join(run_dir, "best_model.pth"))

    with open(os.path.join(run_dir, "training_rewards.csv"), "w", newline="") as f:
        csv.writer(f).writerows([("episode", "reward"), *train_rows])
    with open(os.path.join(run_dir, "evaluation_rewards.csv"), "w", newline="") as f:
        csv.writer(f).writerows([("episode", "avg_reward"), *eval_rows])

    return train_rows, eval_rows


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env = HPProteinFoldingEnv("HPHPPHHPHPPHPHHPPHPH")
    run_dir = f"./run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s",
                        handlers=[logging.FileHandler(os.path.join(run_dir, "training.log")),
                                  logging.StreamHandler()])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(
        length=env.length, num_actions=5, gamma=0.98, lr=5e-4, batch_size=512,
        memory_size=8000, target_update_freq=1000, device=device,
        d_model=64, nhead=4, num_layers=1, dim_feedforward=256,
    )

    train_rewards, eval_rewards = train_dqn(
            env, agent, run_dir, num_episodes=80000, max_steps=env.length * 2, eval_interval=1000,
        )
    torch.save(agent.policy_net.state_dict(), os.path.join(run_dir, "final_model.pth"))
    final_avg = evaluate_agent(env, agent, num_episodes=10, max_steps=env.length * 5)
    logging.info(f"Final evaluation avg reward: {final_avg:.2f}")
    
    plt.figure()
    eps, vals = zip(*train_rewards)
    plt.plot(eps, vals, label="Training Reward")
    if eval_rewards:
        eeps, evals = zip(*eval_rewards)
        plt.plot(eeps, evals, "ro-", label="Evaluation Reward")
    plt.xlabel("Episode"); plt.ylabel("Reward"); plt.legend()
    plt.savefig(os.path.join(run_dir, "training_rewards.png"))
    plt.close()
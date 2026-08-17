# Reinforcement Learning: Basic Cycle, Q-Learning, and Protein Folding

## Basic Cycle in Reinforcement Learning

**A (Agent), E (Environment)**

The agent interacts with the environment using a set of actions.

* An Environment gives the agent an observation **O1**
* To which the agent replies via an action **A1**
* The environment takes that action and gives the agent the reward **R1** and next observation **O2**

Reinforcement learning is fundamentally modeled using **Markov Decision Processes (MDPs)**.

**The Markov Property:** For a process to be an MDP, the future depends only on the current state and action, not on the past history.

An observation can act as the state (or a proxy for the state) to frame the agent's interaction with the environment within this mathematical structure.

---

## Delayed Rewards and Credit Assignment

Most of the environments are structured such that you only get a reward at the end (like - chess you win or lose), or sequence to sequence in LLM where you only know if the generated sentence [set of sequence] is good once you have all the sequence out.

**Q-Learning** is one of the ways for credit assignment [assignment of credits to all the sequential steps that has lead to final state].

---

## Q-Learning

### Policy

Policy - simply tells you what you should do - like if you are in state **S** what action you should perform.

**π(s) = a**

which means input a state and you'll get the output as action.

It can be deterministic or probabilistic output of which actions to prefer if in state **s**.

### Q Function

Q Function (it is modelled as a recursion function in Bellman) is:

**Q(S1, A1)**

which means if I am in state **S1** i.e. S1 is my current state and I perform action **A1** what will be the total reward I'll get at the end if I perform action A1 in my current state S1.

**Qπ(s, a1) = 5**

means if you are in state S and you perform action a1 and after that do what policy π tells you to do after state s you'll get the reward of 5 (total reward after the end of the episode).

## Bellman Equation

The Bellman equation expresses the Q-function recursively:

$$
Q_\pi(s,a) = r(s,a) + \gamma \sum_{s'} P(s'|s,a)\sum_{a'}\pi(a'|s')Q_\pi(s',a')
$$

For Q-Learning, we use the optimal Bellman equation:

$$
Q(s,a) = r(s,a) + \gamma \max_{a'} Q(s',a')
$$

Where:

- **Q(s,a)** = expected total reward for taking action `a` in state `s`
- **r(s,a)** = immediate reward received after taking action `a`
- **γ (gamma)** = discount factor, controlling how much we value future rewards
- **s'** = next state after taking action `a`
- **max Q(s',a')** = the best future Q-value possible from the next state

The key idea is:

**Current Q-value = Immediate Reward + Discounted Best Future Q-value**

This is what makes Q-Learning a recursive way of learning the value of actions.

The policy can then be derived from the learned Q-function:

$$
\pi(s) = \arg\max_a Q(s,a)
$$

Meaning: **in state `s`, choose the action `a` with the highest Q-value.**

Q-Learning is all about learning this Q function which states if I am currently in state S what action should I take to maximise total reward.

---

## Modelling the Q Function

You can model Q function as a table:

| State | Action | Q value |
| ----- | ------ | ------- |
| S     | A      | Q(S,A)  |

So given a state **S** and an action **A** what will be Q values it take a note of all this in table format and you search the table before taking action.

Or else:

You build a neural network to model this mapping from state to actions and Q values.

---

## Why Q-Learning Only Works for Simplified Protein Folding Models

**Q-function is hard to learn, but because the conformational search space grows exponentially with the length of the amino acid chain (known as Levinthal's Paradox).**

### Why Learning the Q-Function is Difficult in Protein Folding

> **Astronomical State Space:** Proteins consist of hundreds of amino acids with continuous rotational degrees of freedom (backbone torsion angles), making discrete state-action Q-tables impossible and extremely difficult to approximate.

> **Delayed Rewards:** A stable fold depends on long-range atomic interactions (like hydrophobic clustering) that only manifest at the very end of the folding process, resulting in sparse intermediate rewards for a Q-learning agent.

> **Curse of Dimensionality:** Standard Q-functions fail to generalize smoothly across minor atomic shifts, trapping agents in local energy minima rather than finding the true native global state.

import numpy as np
from collections import namedtuple

from scripts.options import Option
from scripts.policies import EpsilonGreedyPolicy


class IntraOptionQLearningAgent:
    def __init__(self, state_space, action_space, options, seed):
        """
        The Intra-Option Q-Learning Agent.

        Args:
            state_space (gym.Space): Environment state space.
            action_space (gym.Space): Environment action space.
            seed (int): Random seed for reproducibility.
        """

        self.GAMMA = 0.99        # discount factor

        '''Hyperparameters'''
        self.LR = None           # learning rate

        ''' Agent Environment Interaction '''
        self.state_space = state_space
        self.action_space = action_space
        self.state_size = self.state_space.n
        self.action_size = self.action_space.n
        self.seed = seed

        self.options = options
        self.Qtable = np.zeros((self.state_size, len(self.options)))
        self.option_policy = EpsilonGreedyPolicy(
            self.options, 1.0, 0.005, 0.999, seed=0
        )

        self.reset(seed)

    def reset(self, seed=0):
        self.Qtable = np.zeros((self.state_size, len(self.options)))
        self.option_policy = EpsilonGreedyPolicy(
            self.options, 1.0, 0.005, 0.999, seed=0
        )
        self.current_option = self.option_policy.actions[0]

    def update_hyperparameters(self, **kwargs):
        """This function updates hyperparameters overriding the
        default values.
        """
        for key, value in kwargs.items():
            print(key, value)
            setattr(self, key, value)

        self.reset()

    def update_agent_parameters(self):
        self.option_policy.update()

    def step(self, state, action, reward, next_state, done):

        Q = self.Qtable
        curr_idx = self.current_option.index
        curr_Q_value = Q[state, curr_idx]
        beta = self.current_option.termination[next_state]
        max_next_Q_value = np.max(Q[next_state])
        estimate_Q = (1.0-beta)*Q[next_state, curr_idx] + beta*max_next_Q_value

        # TODO Max over possible states only
        Q[state, curr_idx] = (
            curr_Q_value + self.LR *
            (reward + self.GAMMA*estimate_Q - curr_Q_value)
        )

        if self.current_option.is_terminated(next_state):

            self.current_option = self.option_policy.act(
                next_state, self.Qtable
            )

    def act(self, state):
        return self.current_option.policy.act(state)

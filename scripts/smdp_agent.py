import numpy as np
from collections import namedtuple

from scripts.options import Option
from scripts.policies import EpsilonGreedyPolicy


class SMDPQLearningAgent:
    def __init__(self, state_space, action_space, options, seed):
        """
        The SMDP Q-Learning Agent.

        Args:
            state_space (gym.Space): Environment state space.
            action_space (gym.Space): Environment action space.
            seed (int): Random seed for reproducibility.
        """

        self.GAMMA = 0.90        # discount factor

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

        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"]
        )

        self.reset(seed)

    def reset(self, seed=0):
        self.history = []
        self.Qtable = np.zeros((self.state_size, len(self.options)))
        self.option_policy = EpsilonGreedyPolicy(
            self.options, 1.0, 0.005, 0.999, seed=0
        )
        self.current_option = self.option_policy.options[0]

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
        self.history.clear()

    def step(self, state, action, reward, next_state, done):

        e = self.experience(state, action, reward, next_state, done)
        self.history.append(e)

        if self.current_option.is_terminated(next_state):

            time_steps = len(self.history)
            initial_state = self.history[0].state
            final_state = self.history[-1].next_state

            discounted_reward = 0
            for step in reversed(self.history):
                discounted_reward = step.reward + self.GAMMA * discounted_reward

            Q = self.Qtable
            cur_idx = self.current_option.index
            curr_Q_value = Q[initial_state, cur_idx]

            possible_options = []
            for option in self.options:
                if option.can_initiate(final_state):
                    possible_options.append(option.index)

            next_Q_value = np.max(Q[final_state, possible_options])

            Q[initial_state, cur_idx] = (
                curr_Q_value + self.LR *
                (discounted_reward + self.GAMMA **
                 time_steps*next_Q_value - curr_Q_value)
            )

            self.current_option = self.option_policy.act(
                next_state, self.Qtable
            )
            self.history.clear()

    def act(self, state):
        return self.current_option.policy.act(state)

    def set_option(self, state):
        self.current_option = self.option_policy.act(
            state, self.Qtable
        )

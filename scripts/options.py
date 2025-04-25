import numpy as np


class Option:

    def __init__(self, index, initiation_set, policy, termination, seed=None):
        """
        Initialize an Option for SMDP Q-learning.

        Args:
            index (int): An index to identify and index the Qtables
            initiation_set (Any): An object that supports __getitem__ access. 
                Determines whether the option can be initiated in a given state. 

            policy (Callable): Policy to be used while executing option

            termination (Any): An object that supports __getitem__ access. Returns 
                the probability of the option terminating at a given state. 

            seed (int): Random seed for reproducibility.
        """

        self.index = index

        self.initiation_set = initiation_set
        self.policy = policy
        self.termination = termination

        self.rng = np.random.default_rng(seed)

    def can_initiate(self, state):
        return self.initiation_set[state]

    def is_terminated(self, state):
        p = self.termination[state]
        return self.rng.uniform(0.0, 1.0) <= p

    def update_policy_parameters(self):
        self.policy.update()
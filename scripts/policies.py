import numpy as np

from scripts.taxi_utils import TaxiUtils as tu


class EpsilonGreedyPolicy:

    def __init__(self, options, eps_start, eps_end, eps_decay,
                 decay_type="exponential", seed: int = None):
        """
        Args:
            options (int or list): List of possible options.
            Qtable (ndarray): Reference to Q-table.
            eps_start (float): Starting epsilon value.
            eps_end (float): Minimum epsilon value.
            eps_decay (float): Decay factor.
            decay_type (str): 'exponential' or 'linear'.
            seed (int): Random seed for reproducibility.
        """
        if isinstance(options, int):
            self.options = list(range(options))
        else:
            self.options = list(options)

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.decay_type = decay_type

        self.eps = eps_start
        self.rng = np.random.default_rng(seed)

    def act(self, state, Qtable):
        """Select an action using the epsilon-greedy strategy."""
        option_values = Qtable[state]
        # TODO Check if option can be initiated before returning

        possible_options = []
        for option in self.options:
            if option.can_initiate(state):
                possible_options.append(option.index)

        if self.rng.uniform(0.0, 1.0) <= self.eps:
            option_choice = self.rng.choice(possible_options)
            return self.options[option_choice]
        else:
            idx = np.argmax(option_values[possible_options])
            option_choice = possible_options[idx]
            return self.options[option_choice]

    def update(self):
        """Update epsilon based on decay type."""
        if self.decay_type == 'linear':
            self.eps = max(self.eps_end, self.eps - self.eps_decay)
        elif self.decay_type == 'exponential':
            self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def reset(self, eps_start: float, eps_end: float, eps_decay: float):
        """Reset the decay schedule."""
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.eps = eps_start


class MoveTaxiPolicy:

    def __init__(self, source):

        self.adj_list = tu.build_taxi_graph()
        self.dist, self.prev = tu.dijkstra(source, self.adj_list)

    def act(self, state):
        taxi_row, taxi_col, passenger_location, destination = (
            tu.decode_env_state(state)
        )

        curr_loc = (taxi_row, taxi_col)
        next_loc = self.prev[curr_loc]

        if next_loc is not None:
            move = (next_loc[0]-curr_loc[0], next_loc[1]-curr_loc[1])
            return tu.DIR_TO_ACTION[move]
        else:

            if passenger_location == tu.IN_TAXI:

                if tu.LOC_TO_COLOR[curr_loc] == destination:
                    return tu.DROPOFF
                else:
                    return tu.DROPOFF
            else:

                if tu.LOC_TO_COLOR[curr_loc] == passenger_location:
                    return tu.PICKUP
                else:
                    return tu.PICKUP


class PickUpPassenger:

    def __init__(self):

        self.sub_policies = {
            tu.RED: MoveTaxiPolicy(tu.COLOR_TO_LOC[tu.RED]),
            tu.GREEN: MoveTaxiPolicy(tu.COLOR_TO_LOC[tu.GREEN]),
            tu.YELLOW: MoveTaxiPolicy(tu.COLOR_TO_LOC[tu.YELLOW]),
            tu.BLUE: MoveTaxiPolicy(tu.COLOR_TO_LOC[tu.BLUE]),
        }

    def act(self, state):

        taxi_row, taxi_col, passenger_location, destination = (
            tu.decode_env_state(state)
        )

        if passenger_location != tu.IN_TAXI:
            return self.sub_policies[passenger_location].act(state)
        else:
            return tu.PICKUP


class DropOffPassenger:

    def __init__(self):

        self.sub_policies = {
            tu.RED: MoveTaxiPolicy(tu.COLOR_TO_LOC[tu.RED]),
            tu.GREEN: MoveTaxiPolicy(tu.COLOR_TO_LOC[tu.GREEN]),
            tu.YELLOW: MoveTaxiPolicy(tu.COLOR_TO_LOC[tu.YELLOW]),
            tu.BLUE: MoveTaxiPolicy(tu.COLOR_TO_LOC[tu.BLUE]),
        }

    def act(self, state):

        taxi_row, taxi_col, passenger_location, destination = (
            tu.decode_env_state(state)
        )

        if passenger_location != tu.IN_TAXI:
            return tu.DROPOFF
        else:
            return self.sub_policies[destination].act(state)


class OptimalPolicy:

    def __init__(self):

        self.sub_policies = {
            tu.RED: MoveTaxiPolicy(tu.COLOR_TO_LOC[tu.RED]),
            tu.GREEN: MoveTaxiPolicy(tu.COLOR_TO_LOC[tu.GREEN]),
            tu.YELLOW: MoveTaxiPolicy(tu.COLOR_TO_LOC[tu.YELLOW]),
            tu.BLUE: MoveTaxiPolicy(tu.COLOR_TO_LOC[tu.BLUE]),
        }

    def act(self, state):

        taxi_row, taxi_col, passenger_location, destination = (
            tu.decode_env_state(state)
        )

        if passenger_location != tu.IN_TAXI:
            return self.sub_policies[passenger_location].act(state)
        else:
            return self.sub_policies[destination].act(state)


if __name__ == '__main__':

    mv = MoveTaxiPolicy((0, 0))
    print(mv.prev)

    mv = MoveTaxiPolicy((0, 4))
    print(mv.prev)

    mv = MoveTaxiPolicy((4, 0))
    print(mv.prev)

    mv = MoveTaxiPolicy((4, 3))
    print(mv.prev)

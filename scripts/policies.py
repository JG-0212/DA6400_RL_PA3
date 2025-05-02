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


def get_termination_move_taxi(option_destination_location):
    # ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination
    termination = [0 for i in range(500)]

    base = (
        100*option_destination_location[0] + 20*option_destination_location[1]
    )
    option_destination = tu.LOC_TO_COLOR[option_destination_location]

    # "Possible" locations for termination
    for destination in range(4):
        for passenger_location in range(5):
            termination[base + passenger_location*4 + destination] = 1

    # Excluding a few states from termination set
    # Enabling passenger Pickup
    for destination in range(4):
        termination[base + option_destination*4 + destination] = 0

    # Enabling passenger Dropoff
    termination[base + tu.IN_TAXI*4 + option_destination] = 0

    return termination


def get_initiation_move_taxi(option_destination_location):
    initiation_set = [True for i in range(500)]

    # Disable initiation in cases where taxi is already at the option's destination
    taxi_row, taxi_col = option_destination_location
    for passenger_location in range(5):
        for destination in range(4):
            initiation_set[taxi_row*100 + taxi_col*20 +
                           passenger_location*4 + destination] = False

    return initiation_set


def get_termination_pick_passenger():
    # ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination
    termination = [0 for i in range(500)]

    for taxi_row in range(5):
        for taxi_col in range(5):
            for destination in range(4):
                termination[taxi_row*100 + taxi_col *
                            20 + tu.IN_TAXI*4 + destination] = 1

    return termination


def get_termination_drop_passenger():
    # ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination
    termination = [0 for i in range(500)]

    for destination in range(4):
        for passenger_location in range(4):
            taxi_row, taxi_col = tu.COLOR_TO_LOC[destination]
            termination[taxi_row*100 + taxi_col*20 +
                        passenger_location*4 + destination] = 1

    return termination


if __name__ == '__main__':

    mv = MoveTaxiPolicy((0, 0))
    print(mv.prev)

    mv = MoveTaxiPolicy((0, 4))
    print(mv.prev)

    mv = MoveTaxiPolicy((4, 0))
    print(mv.prev)

    mv = MoveTaxiPolicy((4, 3))
    print(mv.prev)

import numpy as np
import heapq


def decode_env_state(state):
    destination = state % 4
    state = state//4
    passenger_location = state % 5
    state = state//5
    taxi_col = state % 5
    state = state//5
    taxi_row = state

    return taxi_row, taxi_col, passenger_location, destination


class EpsilonGreedyPolicy:

    def __init__(self, actions, eps_start, eps_end, eps_decay,
                 decay_type="exponential", seed: int = None):
        """
        Args:
            actions (int or list): Total number of actions or list of possible actions / options.
            Qtable (ndarray): Reference to Q-table.
            eps_start (float): Starting epsilon value.
            eps_end (float): Minimum epsilon value.
            eps_decay (float): Decay factor.
            decay_type (str): 'exponential' or 'linear'.
            seed (int): Random seed for reproducibility.
        """
        if isinstance(actions, int):
            self.actions = list(range(actions))
        else:
            self.actions = list(actions)

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.decay_type = decay_type

        self.eps = eps_start
        self.rng = np.random.default_rng(seed)

    def act(self, state, Qtable):
        """Select an action using the epsilon-greedy strategy."""
        action_values = Qtable[state]
        # TODO Check if option can be initiated before returning
        if self.rng.uniform(0.0, 1.0) <= self.eps:
            return self.rng.choice(self.actions)
        else:
            return self.actions[np.argmax(action_values)]

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

        self.adj_list = {}
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        self.direction_map = {
            directions[0]: 0,
            directions[1]: 2,
            directions[2]: 1,
            directions[3]: 3
        }

        self.location_map = {
            (0, 0): 0,
            (4, 0): 2,
            (0, 4): 1,
            (4, 3): 3
        }

        self.PASSENGER_IN_TAXI = 4

        self.DROP_PASSENGER = 5
        self.PICK_PASSENGER = 4

        for i in range(5):
            for j in range(5):
                self.adj_list[(i, j)] = []
                for x, y in directions:
                    if (0 <= i+x < 5) and (0 <= j+y < 5):
                        self.adj_list[(i, j)].append((i+x, j+y))

        obstacles = [
            [(4, 0), (4, 1)],
            [(3, 0), (3, 1)],
            [(4, 2), (4, 3)],
            [(3, 2), (3, 3)],
            [(0, 1), (0, 2)],
            [(1, 1), (1, 2)]
        ]

        for node1, node2 in obstacles:
            self.adj_list[node1].remove(node2)
            self.adj_list[node2].remove(node1)

        dist = {}
        prev = {}
        visited = {}

        dist[source] = 0
        prev[source] = None

        pq = []
        heapq.heappush(pq, (dist[source], source))

        while len(pq):
            _, u = heapq.heappop(pq)
            visited[u] = True

            for neighbour in self.adj_list[u]:
                if visited.get(neighbour, False):
                    continue

                alt = dist[u] + 1
                d = dist.get(neighbour, -1)
                if (d == -1) or (alt < d):
                    dist[neighbour] = alt
                    prev[neighbour] = u
                    heapq.heappush(pq, (alt, neighbour))

        self.dist = dist
        self.prev = prev

    def act(self, state):
        taxi_row, taxi_col, passenger_location, destination = (
            decode_env_state(state)
        )

        curr_loc = (taxi_row, taxi_col)
        next_loc = self.prev[curr_loc]

        if next_loc is not None:
            move = (next_loc[0]-curr_loc[0], next_loc[1]-curr_loc[1])
            return self.direction_map[move]
        else:
            if (passenger_location == self.PASSENGER_IN_TAXI) and (self.location_map[curr_loc] != passenger_location):
                return self.DROP_PASSENGER
            else:
                return self.PICK_PASSENGER


if __name__ == '__main__':

    mv = MoveTaxiPolicy((0, 0))
    print(mv.prev)

    mv = MoveTaxiPolicy((0, 4))
    print(mv.prev)

    mv = MoveTaxiPolicy((4, 0))
    print(mv.prev)

    mv = MoveTaxiPolicy((4, 3))
    print(mv.prev)

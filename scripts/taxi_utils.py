import heapq


class TaxiUtils:
    """
    Constants and utility functions for the Taxi-v3 environment.
    To improve readability and simplify working with the environment.
    """

    # Grid
    GRID_ROWS = 5
    GRID_COLS = 5

    # Action space
    SOUTH = 0
    NORTH = 1
    EAST = 2
    WEST = 3
    PICKUP = 4
    DROPOFF = 5

    # Passenger locations / destinations
    RED = 0
    GREEN = 1
    YELLOW = 2
    BLUE = 3
    IN_TAXI = 4  # Passenger is inside the taxi

    # Mapping from movement directions to actions
    DIR_TO_ACTION = {
        (1, 0): SOUTH,     # move down
        (-1, 0): NORTH,    # move up
        (0, 1): EAST,      # move right
        (0, -1): WEST      # move left
    }

    # Mapping from grid locations to passenger locations / destinations
    LOC_TO_COLOR = {
        (0, 0): RED,
        (0, 4): GREEN,
        (4, 0): YELLOW,
        (4, 3): BLUE
    }

    # Reverse mapping
    DIRECTIONS = list(DIR_TO_ACTION.keys())
    ACTION_TO_DIR = {v: k for k, v in DIR_TO_ACTION.items()}
    COLOR_TO_LOC = {v: k for k, v in LOC_TO_COLOR.items()}

    @staticmethod
    def decode_env_state(state):
        """Decode the environment state,
        (taxi_row, taxi_col, passenger_location, destination).
        """
        destination = state % 4
        state //= 4
        passenger_location = state % 5
        state //= 5
        taxi_col = state % 5
        taxi_row = state // 5

        return taxi_row, taxi_col, passenger_location, destination

    @staticmethod
    def encode_env_state(taxi_row, taxi_col, passenger_location, destination):
        """Encode (row, col, passenger, destination) tuple to environment state.
        """
        return (((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination)

    @staticmethod
    def build_taxi_graph():
        """Constructs the adjacency list representing the Taxi-v3 grid.
        Returns:
            adj_list: dict mapping each (row, col) to a list of accessible neighbor (row, col) cells.
        """
        adj_list = {}

        # Initialize 5x5 grid with all valid neighbors
        for i in range(5):
            for j in range(5):
                adj_list[(i, j)] = []
                for dx, dy in TaxiUtils.DIRECTIONS:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < 5 and 0 <= nj < 5:
                        adj_list[(i, j)].append((ni, nj))

        # Hardcoded obstacles from the Taxi-v3 map
        obstacles = [
            [(4, 0), (4, 1)], [(3, 0), (3, 1)],
            [(4, 2), (4, 3)], [(3, 2), (3, 3)],
            [(0, 1), (0, 2)], [(1, 1), (1, 2)]
        ]

        # Remove connections between nodes with obstacles in between
        for node1, node2 in obstacles:
            if node2 in adj_list[node1]:
                adj_list[node1].remove(node2)
            if node1 in adj_list[node2]:
                adj_list[node2].remove(node1)

        return adj_list

    @staticmethod
    def dijkstra(source, adj_list):
        """Dijkstra's algorithm from a source node to compute shortest paths.

        Args:
            source (tuple): (row, col) grid location from which paths are computed.
            adj_list (dict): adjacency list for the Taxi-v3 grid (see build_taxi_graph()).

        Returns:
            dist (dict): mapping from each node to its shortest distance from the source.
            prev (dict): mapping from each node to the previous node on its shortest path from the source.

        Notes:
            As the graph is undirected, starting from any node, following prev[node] recursively will lead
            back to the source along the shortest path.
        """

        dist = {source: 0}
        prev = {source: None}
        visited = {}

        pq = [(0, source)]  # Priority queue of (distance, node)

        while pq:
            _, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited[u] = True

            for neighbor in adj_list[u]:
                if neighbor in visited:
                    continue

                alt = dist[u] + 1
                if neighbor not in dist or alt < dist[neighbor]:
                    dist[neighbor] = alt
                    prev[neighbor] = u
                    heapq.heappush(pq, (alt, neighbor))

        return dist, prev

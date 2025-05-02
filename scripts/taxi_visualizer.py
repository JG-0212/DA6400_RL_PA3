import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from matplotlib import colormaps as cm

from scripts.taxi_utils import TaxiUtils as tu


class TaxiVisualizer:
    """
    Visualization functions for the Taxi-v3 environment.
    To visualize states, Q-Values, option/action choices.
    """

    @staticmethod
    def visualize_heatmap(heatmap):
        """Visualizes a heatmap (eg. Q-values) over the Taxi-v3 grid.

        Args:
            heatmap (np.ndarray): A 2D array representing the Q-values 
                to display over the environment.
        """

        ax = plt.gca()

        im = ax.imshow(
            heatmap,
            cmap="plasma",
            extent=[0, tu.GRID_COLS, tu.GRID_ROWS, 0],
            interpolation="nearest",
            alpha=0.5,
        )

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    @staticmethod
    def visualize_locations():
        """Visualizes the 4 Destinations in Taxi-v3.
        """

        color_map = {tu.RED: "red", tu.GREEN: "green",
                     tu.YELLOW: "yellow", tu.BLUE: "blue"}

        ax = plt.gca()
        for clr in color_map:
            pos = tu.COLOR_TO_LOC[clr]
            rectangle = plt.Rectangle(
                (pos[1], pos[0]),
                1, 1, facecolor=color_map[clr],
                edgecolor="none", alpha=0.8
            )
            ax.add_patch(rectangle)

    @staticmethod
    def visualize_adjacency():
        """Visualizes the Taxi-v3 grid (with obstacles)
        """
        adj_list = tu.build_taxi_graph()
        ax = plt.gca()

        for row in range(tu.GRID_ROWS):
            for col in range(tu.GRID_COLS):
                current = (row, col)
                neighbors = adj_list.get(current, [])

                for i, j in tu.DIRECTIONS:
                    if (0 <= row+i < 5) and (0 <= col+j < 5):

                        lw = 0.5 if ((row+i, col+j) in neighbors) else 3.0

                        rlim, clim = ([row + (i == 1), row + (i >= 0)],
                                      [col + (j == 1), col + (j >= 0)])

                        ax.plot(clim, rlim, color="black", linewidth=lw)

        ax.plot([0, tu.GRID_COLS],
                [tu.GRID_ROWS, tu.GRID_ROWS], color="black", linewidth=3)

        ax.plot([0, tu.GRID_COLS],
                [0, 0], color="black", linewidth=3)

        ax.plot([0, 0],
                [0, tu.GRID_ROWS], color="black", linewidth=3)

        ax.plot([tu.GRID_COLS, tu.GRID_COLS],
                [0, tu.GRID_ROWS], color="black", linewidth=3)

    @staticmethod
    def visualize_taxi_passenger_destination(state):
        """Visualizes the taxi, passenger, and destination for a given Taxi-v3 state.

        Args:
            state (int): Taxi-v3 encoded state, in [0, 500].
        """

        def add_img_util(ax, path, xy, zoom=0.1):
            """Adds an image to the plot at a given location.
            """
            img = mpimg.imread(path)
            imagebox = OffsetImage(img, zoom=zoom)
            ab = AnnotationBbox(imagebox, xy, frameon=False)
            ax.add_artist(ab)

        taxi_row, taxi_col, passenger_location, destination = (
            tu.decode_env_state(state)
        )

        taxi_location = (taxi_row, taxi_col)

        destination = tu.COLOR_TO_LOC[destination]

        ax = plt.gca()

        if passenger_location == tu.IN_TAXI:
            add_img_util(
                ax, "assets/passenger_in_taxi.png",
                (taxi_col + 0.5, taxi_row + 0.5),
                zoom=0.35)
        elif tu.COLOR_TO_LOC.get(passenger_location, None) == taxi_location:
            passenger_location = tu.COLOR_TO_LOC[passenger_location]
            add_img_util(
                ax, "assets/taxi.png",
                (taxi_col + 0.25, taxi_row + 0.5),
                zoom=0.25)
            add_img_util(
                ax, "assets/passenger.png",
                (passenger_location[1] + 0.75, passenger_location[0] + 0.5),
                zoom=0.25)
        else:
            passenger_location = tu.COLOR_TO_LOC[passenger_location]
            add_img_util(
                ax, "assets/taxi.png",
                (taxi_col + 0.5, taxi_row + 0.5),
                zoom=0.30)
            add_img_util(
                ax, "assets/passenger.png",
                (passenger_location[1] + 0.5, passenger_location[0] + 0.5),
                zoom=0.30)

        add_img_util(
            ax, "assets/destination.png",
            (destination[1] + 0.5, destination[0] + 0.5),
            zoom=0.30)

    @staticmethod
    def visualize_actions(action_heatmap):
        """Visualizes action directions on the Taxi-v3 grid (as a quiver plot).

        Args:
            action_heatmap (np.ndarray): A 5x5 array where each cell contains an action 
                (encoded in Taxi-v3 format).
        """
        ax = plt.gca()
        for row in range(tu.GRID_ROWS):
            for col in range(tu.GRID_COLS):

                direction = tu.ACTION_TO_DIR.get(
                    action_heatmap[row][col], (0, 0))
                if direction == (0, 0):
                    continue

                ax.annotate("", xy=(col + 0.5 + 0.2*direction[1],
                                    row + 0.5 + 0.2*direction[0]),
                            xytext=(col + 0.5 - 0.2*direction[1],
                                    row + 0.5 - 0.2*direction[0]),
                            arrowprops=dict(linewidth=2.5, facecolor="blue",
                                            edgecolor="blue", arrowstyle="->"))

    @staticmethod
    def visualize_options(option_heatmap, option_labels=None):
        """Visualizes selected/assigned options over the Taxi-v3 grid.

        Args:
            option_heatmap (np.ndarray): A 5x5 array where each cell contains the
                selected option. 
            option_labels (dict): Descriptive names mapped to each option.
        """

        tab10 = cm.get_cmap("tab10")
        ax = plt.gca()
        vis = {}
        for row in range(tu.GRID_ROWS):
            for col in range(tu.GRID_COLS):
                option = option_heatmap[row][col]
                label = f"Option {option}" if option_labels is None else option_labels[option]
                show_label = not vis.get(option, False)

                circle = plt.Circle(
                    (col + 0.5, row + 0.5),
                    0.2,
                    facecolor=tab10(option),
                    edgecolor="none",
                    alpha=0.8,
                    label=label if show_label else None
                )
                ax.add_patch(circle)
                vis[option] = True

    @staticmethod
    def visualize_options_bubble_plot(option_values, option_labels=None, norm_axis=None):
        """Visualize Q-values for different options as a bubble plot.

        Args:
            option_values (np.ndarray): 3D array of Q-Values (rows, cols, options).
            option_labels (dict): Descriptive names mapped to each option.
            norm_axis (int): Axis over which Q-values are normalized for plotting (-1, None).
                - 'None' (default) normalizes across all states.
                - '-1' normalizes Q-values per state.
        """

        ax = plt.gca()
        tab10 = cm.get_cmap("tab10")

        def circular_arrange(k, total_width):
            if k == 1:
                scale, centres = 0.5, np.array([[0, 0]])
                return scale, centres

            R = (total_width/2.0)/(1.0 + np.sin(np.pi/k))
            scale = R*np.sin(np.pi/k)

            tht = np.linspace(0, 2*np.pi, k, endpoint=False)+np.pi/k
            centres = np.array([R*np.cos(tht), R*np.sin(tht)]).T
            return scale, centres

        n, m, k = option_values.shape

        values = option_values[:]
        values = (
            values / np.expand_dims(
                np.max(np.abs(values)+1e-15, axis=norm_axis), axis=-1)
        )

        scale, centres = circular_arrange(k, 1)
        # values *= scale

        vis = {}
        sf = 10
        for row in range(n):
            for col in range(m):
                for v in range(k):
                    r = values[row, col, v]
                    cx, cy = centres[v, :]

                    label = f"Option {v}" if option_labels is None else option_labels[v]
                    show_label = not vis.get(v, False)

                    circle = plt.Circle(
                        (cx + col + 0.5, cy + row + 0.5),
                        scale*np.log10((sf-1.0)*abs(r)+1.0)/np.log10(sf),
                        facecolor=tab10(v),
                        edgecolor="none" if r > 0 else "black",
                        linestyle="solid" if r > 0 else "dashed",
                        alpha=0.8,
                        label=label if show_label else None
                    )
                    ax.add_patch(circle)
                    vis[v] = True

def vis(state, Qtable, option_labels=None):

    taxi_row, taxi_col, passenger_location, destination = tu.decode_env_state(state)
    
    fig, ax = plt.subplots(figsize=(5, 5))

    TaxiVisualizer.visualize_adjacency()
    TaxiVisualizer.visualize_locations()

    option_values = Qtable[:,:,passenger_location, destination,:]
    options = np.argmax(option_values, axis=-1)

    TaxiVisualizer.visualize_options(options, option_labels)
    
    # tv.visualize_options_bubble_plot(option_values, option_labels, norm_axis=None)
    
    # tv.visualize_heatmap(qvals)
    # tv.visualize_taxi_passenger_destination(state)


    fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=10)

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()
    

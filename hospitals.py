import random
import matplotlib.pyplot as plt
import numpy as np

class Space():
    def __init__(self, height, width, num_hospitals):
        """Create a new state space with given dimensions."""
        self.height = height
        self.width = width
        self.num_hospitals = num_hospitals
        self.houses = set()
        self.hospitals = set()
        self.history = []  # To store hospital positions during optimization

    def add_house(self, row, col):
        """Add a house at a particular location in state space."""
        self.houses.add((row, col))

    def available_spaces(self):
        """Returns all cells not currently used by a house or hospital."""
        candidates = set(
            (row, col)
            for row in range(self.height)
            for col in range(self.width)
        )
        for house in self.houses:
            candidates.discard(house)
        for hospital in self.hospitals:
            candidates.discard(hospital)
        return candidates

    def hill_climb(self, maximum=None, log=False):
        """Performs hill-climbing to find a solution."""
        count = 0

        # Start by initializing hospitals randomly
        self.hospitals = set()
        for _ in range(self.num_hospitals):
            self.hospitals.add(random.choice(list(self.available_spaces())))
        
        self.history.append(list(self.hospitals))  # Track initial position

        if log:
            print("Initial state: cost", self.get_cost(self.hospitals))

        # Continue until we reach maximum number of iterations
        while maximum is None or count < maximum:
            count += 1
            best_neighbors = []
            best_neighbor_cost = None

            # Consider all hospitals to move
            for hospital in self.hospitals:
                for replacement in self.get_neighbors(*hospital):
                    neighbor = self.hospitals.copy()
                    neighbor.remove(hospital)
                    neighbor.add(replacement)

                    # Check if neighbor is best so far
                    cost = self.get_cost(neighbor)
                    if best_neighbor_cost is None or cost < best_neighbor_cost:
                        best_neighbor_cost = cost
                        best_neighbors = [neighbor]
                    elif best_neighbor_cost == cost:
                        best_neighbors.append(neighbor)

            # None of the neighbors are better than the current state
            if best_neighbor_cost >= self.get_cost(self.hospitals):
                return self.hospitals

            # Move to a highest-valued neighbor
            else:
                self.hospitals = random.choice(best_neighbors)
                self.history.append(list(self.hospitals))  # Track new position

                if log:
                    print(f"Step {count}: Found better neighbor with cost {best_neighbor_cost}")

    def get_cost(self, hospitals):
        """Calculates sum of distances from houses to nearest hospital."""
        cost = 0
        for house in self.houses:
            cost += min(
                abs(house[0] - hospital[0]) + abs(house[1] - hospital[1])
                for hospital in hospitals
            )
        return cost

    def get_neighbors(self, row, col):
        """Returns neighbors not already containing a house or hospital."""
        candidates = [
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1)
        ]
        neighbors = []
        for r, c in candidates:
            if (r, c) in self.houses or (r, c) in self.hospitals:
                continue
            if 0 <= r < self.height and 0 <= c < self.width:
                neighbors.append((r, c))
        return neighbors

    def generate_heatmap(self):
        """Generates a heatmap of Manhattan distances to all houses."""
        heatmap = np.zeros((self.height, self.width))
        for row in range(self.height):
            for col in range(self.width):
                heatmap[row, col] = sum(
                    abs(row - house[0]) + abs(col - house[1])
                    for house in self.houses
                )
        return heatmap

    def plot_heatmap_with_path(self):
        """Plots heatmap with the optimization path."""
        heatmap = self.generate_heatmap()

        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(heatmap, cmap='hot', interpolation='nearest')
        plt.colorbar(label="Total Manhattan Distance")
        plt.title("Heat Map of Manhattan distances of hospital to all houses")
        plt.xlabel("X-coordinate")
        plt.ylabel("Y-coordinate")

        # Plot the optimization path
        path = self.history
        for i in range(len(path) - 1):
            for hospital_idx in range(len(path[i])):
                start = path[i][hospital_idx]
                end = path[i + 1][hospital_idx]
                plt.arrow(
                    start[1], start[0],  # Arrow starts here
                    end[1] - start[1],  # Horizontal movement
                    end[0] - start[0],  # Vertical movement
                    color="green", head_width=0.3, head_length=0.4, length_includes_head=True
                )

        # Mark final hospital locations
        for hospital in path[-1]:
            plt.plot(hospital[1], hospital[0], "g^", markersize=10, label="Final Hospital")

        # Mark initial hospital locations
        for hospital in path[0]:
            plt.plot(hospital[1], hospital[0], "gx", markersize=10, label="Initial Hospital")

        plt.legend()
        plt.show()


# Initialize the space
s = Space(height=15, width=20, num_hospitals=1)

# Add random houses
random.seed(1)  # For consistent house placement
for _ in range(20):
    s.add_house(random.randint(0, s.height - 1), random.randint(0, s.width - 1))

# Perform hill climbing
s.hill_climb(log=True)

# Plot the heatmap with the optimization path
s.plot_heatmap_with_path()

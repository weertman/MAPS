import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Tools:
    class CreateSyntheticDataset:
        def __init__(self, num_points=9, num_timesteps=1000):
            self.num_points = num_points
            self.num_timesteps = num_timesteps
            self.star = self.create_star(1, 0.5)

        def create_star(self, radius1, radius2):
            angles = np.linspace(0, 2 * np.pi, self.num_points*2, endpoint=False)
            radius = np.array([radius1, radius2] * self.num_points)
            x = radius * np.cos(angles)
            y = radius * np.sin(angles)
            return np.column_stack((x, y))[::2]

        def simulate_movement(self, motion_type='random_walk'):
            star_points = np.copy(self.star)
            trajectories = np.zeros((self.num_points, self.num_timesteps, 2))

            for t in range(self.num_timesteps):
                if motion_type == 'linear_translation':
                    movement = np.array([0.1, 0.1])
                elif motion_type == 'random_walk':
                    movement = np.random.normal(0, 0.1, 2)
                elif motion_type == 'sinusoidal':
                    movement = np.array([np.sin(0.1 * t), np.cos(0.1 * t)])
                elif motion_type == 'spiral':
                    r = 0.001 * t
                    theta = 0.1 * t
                    movement = np.array([r * np.cos(theta), r * np.sin(theta)])
                elif motion_type == 'rotation':
                    theta = 0.01 * t
                    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                                [np.sin(theta), np.cos(theta)]])
                    star_points = star_points - star_points.mean(axis=0)
                    star_points = np.dot(star_points, rotation_matrix)
                    star_points = star_points + star_points.mean(axis=0)
                    movement = np.array([0, 0])  # No translation in pure rotation
                elif motion_type == 'random_weighted_walk':
                    movement = np.random.normal(0, 0.1, 2) + np.array([0.1, 0])
                elif motion_type == 'parabolic_trajectory':
                    movement = np.array([0.1, -0.0005 * (t ** 2) + 0.1 * t])
                elif motion_type == 'sinusoidally_accelerating':
                    movement = np.array([np.sin(0.1 * t) * t, np.cos(0.1 * t) * t])
                elif motion_type == 'climbing':
                    movement = np.array([0, 0.0002 * (t ** 2)])
                else:
                    raise ValueError(f"Unknown motion type: {motion_type}")

                if motion_type != 'rotation':
                    star_points += movement

                trajectories[:, t, :] = star_points

            return trajectories

        def create_dataframe(self, trajectories):
            data = {f'a{i}': trajectories[i].tolist() for i in range(self.num_points)}
            return pd.DataFrame(data)

        def plot_star_movement(self, trajectories, ax, title):
            for i in range(self.num_points):
                x_coords = trajectories[i, :, 0]
                y_coords = trajectories[i, :, 1]
                ax.plot(x_coords, y_coords, '-o', markersize=2, alpha=0.2)

            for t in range(self.num_timesteps):
                x_coords = trajectories[:, t, 0]
                y_coords = trajectories[:, t, 1]
                ax.plot(x_coords, y_coords, 'k-', alpha=0.05)

            ax.set_title(title)
            ax.axis('equal')

        def generate_and_visualize(self, motion_types):
            fig, axes = plt.subplots(3, 3, figsize=(15, 15))
            axes = axes.flatten()
            dataframes = {}

            for i, motion_type in enumerate(motion_types):
                trajectories = self.simulate_movement(motion_type)
                dataframes[motion_type] = self.create_dataframe(trajectories)
                self.plot_star_movement(trajectories, axes[i], motion_type)

            plt.tight_layout()
            plt.show()
            return dataframes

# Example usage:
tools = Tools.CreateSyntheticDataset(num_points=100, num_timesteps=1000)
motion_types = [
    'linear_translation', 'random_walk', 'sinusoidal', 'spiral', 'rotation',
    'random_weighted_walk', 'parabolic_trajectory', 'sinusoidally_accelerating', 'climbing'
]
dataframes = tools.generate_and_visualize(motion_types)

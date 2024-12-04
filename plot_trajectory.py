import matplotlib.pyplot as plt
from path_handler import PathHandler
# Example data: Replace this with your recorded path

path_handler=PathHandler('training_paths.json')
path=path_handler.get_next_path()
points = path['points']
#for p in points:

obstacles = [
    (-35, 0),  # Example obstacles (x, y)
    (-55, 0),
    (-75, 0),
    (-95, 0)
]

target = (-50, 40)  # Target position (x, y)
start = (-20, 60)  # Start position (x, y)

# Extract X and Y coordinates for plotting
x_coords, y_coords = zip(*drone_path)
obstacle_x, obstacle_y = zip(*obstacles)

# Create a 2D plot
plt.figure(figsize=(8, 8))
plt.plot(x_coords, y_coords, '-o', label="Drone Path", color="blue")  # Drone's path
plt.scatter(obstacle_x, obstacle_y, c="red", label="Obstacles")  # Obstacles
plt.scatter(*start, c="green", label="Start", s=100, marker="x")  # Start point
plt.scatter(*target, c="purple", label="Target", s=100, marker="*")  # Target point

# Add titles and labels
plt.title("Drone Path (Top-Down View)")
plt.xlabel("X-Coordinate")
plt.ylabel("Y-Coordinate")
plt.legend()
plt.grid(True)
plt.axis("equal")  # Ensure equal scaling

# Show the plot
plt.show()

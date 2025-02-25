import numpy as np

def generate_3d_line_coordinates(start, end, num_points):
    # Generate linearly spaced points for x, y, and z
    x_coords = np.linspace(start[0], end[0], num_points)
    y_coords = np.linspace(start[1], end[1], num_points)
    z_coords = np.linspace(start[2], end[2], num_points)
    
    # Combine into a single string
    coordinates = []
    for x, y, z in zip(x_coords, y_coords, z_coords):
        coordinates.append(f"{x:.5f} {y:.5f} {z:.5f}")
    
    return " ".join(coordinates)

# Usage
start_point = (0, 0, 0.7)  # Starting point of the line
end_point = (0, 0, 0.2) # Ending point of the line
num_points = 3          # Number of points to generate, careful: cooresponds to the number of joints not links

coordinate_string = generate_3d_line_coordinates(start_point, end_point, num_points)
print(coordinate_string)

import json
import numpy as np
class PathHandler:
    def __init__(self, json_file):
        """
        Initialize the PathHandler with the JSON file containing paths.
        
        Args:
            json_file (str): Path to the JSON file.
        """
        self.json_file = json_file
        self.current_path_index = -1  # Start before the first path
        self.current_point_index = -1  # Start before the first point
        self.paths = self._load_paths()
    
    def _load_paths(self):
        """
        Load paths from the JSON file.
        
        Returns:
            list: List of paths from the JSON file.
        """
        with open(self.json_file, 'r') as file:
            data = json.load(file)
        
        if "paths" not in data or not data["paths"]:
            raise ValueError("Invalid JSON file: No paths found.")
        
        return data["paths"]

    def get_next_path(self):
        """
        Get the next path in the sequence.
        
        Returns:
            dict: The next path containing path ID and points.
        """
        self.current_path_index = (self.current_path_index + 1) % len(self.paths)
        self.current_point_index = -1  # Reset point index for the new path
        return self.paths[self.current_path_index]

    def get_next_point(self):
        """
        Get the next point inside the current path.
        
        Returns:
            dict: A dictionary containing start and target positions for the next point.
        """
        if self.current_path_index == -1:
            raise ValueError("No path selected. Call get_next_path() first.")
        
        current_path = self.paths[self.current_path_index]
        points = current_path.get("points", [])
        
        self.current_point_index += 1
        if self.current_point_index < len(points):
            return points[self.current_point_index]
        return None
        
        #if not points:
        #    raise ValueError(f"Current path (ID {current_path['path_id']}) has no points.")
        
        # Ensure indices match the length of points
        #self.current_point_index = (self.current_point_index + 1) % len(points)
        
        #return points[self.current_point_index]


# File: plot_training_and_positions.py

import json
import pandas as pd
import matplotlib.pyplot as plt

# Function to read and process the JSON file
def process_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    x_list = []
    y_list=[]
    # Iterate through paths and points
    for path in data['paths']:
        x_path=[]
        y_path=[]
        for point in path['points']:
            start = point['start_position']
            target = point['target_position']
            
            # Calculate mean x and use target's y
            mean_x = (start[0] + target[0]) / 2
            mean_y = target[1]
            
            x_path.append(mean_x)
            y_path.append(mean_y)
        x_list.append(x_path)
        y_list.append(y_path)
    return x_list,y_list

# Function to read the CSV file and extract x, y positions
def process_csv(csv_file):
    df = pd.read_csv(csv_file)
    
    # Assuming the columns for x, y positions are named 'x' and 'y'
    x = df['X'].tolist()
    y = df['Y'].tolist()
    
    return x, y

# Function to plot the results
def plot_positions(json_x,json_y, csv_x, csv_y,csv_x2, csv_y2):
    plt.figure(figsize=(10, 8))
    count=1
    # Plot calculated positions from JSON as red X  
    for x_path,y_path in zip(json_x,json_y) :
        plt.scatter(x_path,y_path, color='red', marker='x')
        count+=1
    
    # Plot positions from CSV as a line
    plt.plot(csv_x,csv_y, color='blue', label='PPO')
    plt.plot(csv_x2,csv_y2, color='green', label='DQN')
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f"Best Episode Models' Trajectories")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main execution
def main():
    # Input file paths
    json_file = 'path_obstacles.json'  # Replace with the JSON file path
    csv_file = 'PPO_best_episode_positions.csv'  # Replace with the CSV file path
    csv_file_2 = 'DQN_best_episode_positions.csv'  # Replace with the CSV file path
    
    # Process files
    json_x,json_y = process_json(json_file)
    



    csv_x, csv_y = process_csv(csv_file)
    csv_x2, csv_y2 = process_csv(csv_file_2)
    # Plot the positions
    plot_positions(json_x,json_y, csv_x, csv_y,csv_x2,csv_y2)

if __name__ == '__main__':
    main()

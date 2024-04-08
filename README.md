# A-star-Implementation-on-TurtleBot3
Contains files and folders for implementing A* algorithm on turtlebot3 

# Team member names
| Name         | UID     | DirID |
|--------------|-----------|------------|
| Mayank Deshpande | 120387333   |  msdshp4  |
| Tanmay Pancholi  |  120116711  |  tamy2909 |
| Suraj Kalwaghe   | 120417634   | Suraj108  |
## Introduction
This repository implements the A* algorithm on a rigid robot navigating through obstacles in a 2D environment. The algorithm finds the optimal path from a start node to a goal node while avoiding obstacles. 

## Getting Started
To run the script, follow these steps:
1. Clone the repository:
```
git clone https://github.com/Suraj-Kalwaghe/A-star-Implementation-on-TurtleBot3.git
```
2. Run the script:
```
python3 a_star_mayank_suraj_tanmay.py
```
3. Follow the on-screen instructions to input the desired parameters such as clearance, RPM values, start and end coordinates, and start theta.

## Dependencies
- Python 3.x
- Pygame
- NumPy
- OpenCV (cv2)

## Explanation of Functions

### cost_fn(Xi, Yi, Thetai, UL, UR, Nodes_list, Path_list, clearance, robot_radius)
- Calculates the cost of moving from a given point with specific wheel rotations.
- Xi, Yi, Thetai: Coordinates and angle of the input point.
- UL, UR: Wheel rotations.
- Nodes_list, Path_list: Lists to store node and path information.
- clearance: Clearance around obstacles.
- robot_radius: Radius of the robot.

### is_valid(x, y, robot_radius, clearance)
- Checks if a given point is valid (not within an obstacle).
- x, y: Coordinates of the point.
- robot_radius: Radius of the robot.
- clearance: Clearance around obstacles.

### a_star(start_position, goal_position, rpm1, rpm2, clearance, robot_radius)
- Implements the A* algorithm to find the optimal path.
- start_position, goal_position: Start and goal nodes.
- rpm1, rpm2: Rotations per minute of the robot wheels.
- clearance: Clearance around obstacles.
- robot_radius: Radius of the robot.

### backtrack(goal_node)
- Backtracks from the goal node to generate the shortest path.
- goal_node: Goal node obtained from A* algorithm.

### plot_path(start_node, goal_node, x_path, y_path, Nodes_list, Path_list, clearance, frame_rate)
- Plots the path and obstacles using Pygame.
- start_node, goal_node: Start and goal nodes.
- x_path, y_path: Coordinates of the shortest path.
- Nodes_list, Path_list: Lists to store node and path information.
- clearance: Clearance around obstacles.
- frame_rate: Frame rate of the animation.

### frames_to_video(frames_dir, output_video)
- Converts generated frames into a video.
- frames_dir: Directory containing frames.
- A_Star_Visualization:[Part01 Video](https://drive.google.com/file/d/1_j--3CNOnS8agndLOYiFqevRaNKXQGpn/view?usp=drive_link).
- Gazebo Simulation: [Part02 Video](https://drive.google.com/file/d/1ZFe5THaUJNQ3GxU8qaQwpMXy0VyHn2xf/view?usp=sharing).

### scale_coordinates(input_x, input_y, canvas_width, canvas_height)
- Scales coordinates to fit within a specified canvas size.
- input_x, input_y: Input coordinates.
- canvas_width, canvas_height: Size of the canvas.

### Controller
- Implemented a controller for turtlebot3 waffle which navigates from point A (fixed point) to point B (user inputed point).
- Considers Clearence, Robot Radius, Obstacle space, RPM values, etc while navigation.
- Publisher code can be found out at
  ```
  cd src/turtlebot3_project3/scripts
  ```
  use
  ```
  python3 Publisher.py
  ```
  to run the publisher code and imput the desired values.

  ### Test Case for A star implementation
|User Inputs | Values     |
|--------------|----------|
| Clearance        |  6   |  
| Left Wheel RPM   |  5   |  
| Right Wheel RPM  | 10   | 
| Start X          | 500  |
| Start Y          | 1000 |
| Theta            | 0    |
| Goal X           | 5750 |
| Goal Y           | 1000 |

  ### Test Case for Publisher
  |User Inputs | Values     |
|--------------|----------|
| Goal X           | 5750 |
| Goal Y           | 1000 |
  
  

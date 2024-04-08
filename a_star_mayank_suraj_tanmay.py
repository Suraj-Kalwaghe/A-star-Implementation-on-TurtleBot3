#!/usr/bin/env python3
# Github repository:  https://github.com/MayankD409/A_star_algorithm_on_rigid_robot.git

import pygame
import numpy as np
import time
import math
import os
import cv2

# Define a class to represent nodes in the search space
class Node:
    def __init__(self, x, y, parent,theta,rpm_left,rpm_right, c2c, c2g, cst):
        self.x = x
        self.y = y
        self.parent = parent
        self.theta = theta
        self.rpm_left = rpm_left
        self.rpm_right = rpm_right
        self.c2c = c2c
        self.c2g = c2g
        self.cst = cst
        
    def __lt__(self,other):
        return self.cst < other.cst

# Define possible actions and associated cost increments
def cost_fn(Xi, Yi, Thetai, UL, UR, present_nodes, parent_nodes, clearance, robot_radius):
    '''
    Xi, Yi,Thetai: Input point's coordinates
    Xs, Ys: Start point coordinates for plot function
    Xn, Yn, Thetan: End point coordintes
    '''
    t = 0
    r = 3.3
    L = 28.7
    dt = 0.1
    cost = 0
    Xn = Xi
    Yn = Yi
    Thetan = 3.14 * Thetai / 180

    while t < 1:
        t = t + dt
        Xs = Xn
        Ys = Yn
        Xn += r*0.5 * (UL + UR) * math.cos(Thetan) * dt
        Yn += r*0.5 * (UL + UR) * math.sin(Thetan) * dt
        Thetan += (r / L) * (UR - UL) * dt
        
        if is_valid(Xn, Yn, robot_radius, clearance):
            c2g = math.dist((Xs, Ys), (Xn, Yn))
            cost = cost + c2g
            present_nodes.append((Xn, Yn))
            parent_nodes.append((Xs, Ys))
        else:
            return None
    
    Thetan = 180 * (Thetan) / 3.14
    return [Xn, Yn, Thetan, cost, present_nodes, parent_nodes]


# Function to check if a point is inside a rectangle
def is_point_inside_rectangle(x, y, vertices):
    x_min = min(vertices[0][0], vertices[1][0], vertices[2][0], vertices[3][0])
    x_max = max(vertices[0][0], vertices[1][0], vertices[2][0], vertices[3][0])
    y_min = min(vertices[0][1], vertices[1][1], vertices[2][1], vertices[3][1])
    y_max = max(vertices[0][1], vertices[1][1], vertices[2][1], vertices[3][1])
    return x_min <= x <= x_max and y_min <= y <= y_max

# Function to check if a point is inside a Circle
def is_point_inside_circle(x, y, center_x, center_y, diameter):
    radius = diameter / 2.0  # Calculate the radius from the diameter
    distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    return distance <= radius

# Function to create configuration space with obstacles
def is_valid(x, y, robot_radius, clearance):

    # Creating buffer space for obstacles    
    rectangle1_buffer_vts = [(150 - (robot_radius + clearance), 200), (175 + (robot_radius + clearance), 200), (175 + (robot_radius + clearance), 100 - (robot_radius + clearance)), (150 - (robot_radius + clearance), 100 - (robot_radius + clearance))]
    rectangle2_buffer_vts = [(250 - (robot_radius + clearance), 100 + (robot_radius + clearance)), (275 + (robot_radius + clearance), 100 - (robot_radius + clearance)), (275 + (robot_radius + clearance), 0), (250 - (robot_radius + clearance), 0)]

    rect1_buffer = is_point_inside_rectangle(x,y, rectangle1_buffer_vts)
    rect2_buffer = is_point_inside_rectangle(x, y, rectangle2_buffer_vts)
    circ_buffer = is_point_inside_circle(x, y, 420, 120, 120 + 2*(robot_radius + clearance))
    
    # Setting buffer space constraints to obtain obstacle space
    if rect1_buffer or rect2_buffer or circ_buffer:
        return False
    
    # Adding check if obstacle is in walls
    if x <= (robot_radius + clearance) or y >= 200 - (robot_radius + clearance) or x >= 600 - (robot_radius + clearance) or y <= (robot_radius + clearance):
        return False

    return True

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

# Function to check if the goal node is reached
def is_goal(present, goal):
    dt = math.dist((present.x, present.y), (goal.x, goal.y))             
    if dt <= 15:
        return True
    else:
        return False
    
# A* algorithm implementation
def a_star(start_position, goal_position, rpm1, rpm2, clearance, robot_radius):
    if is_goal(start_position, goal_position):
        return None, 1

    goal = goal_position
    start = start_position
    
    moves = [[rpm1, 0], 
             [0, rpm1], 
             [rpm1, rpm1], 
             [0, rpm2], 
             [rpm2, 0], 
             [rpm2, rpm2], 
             [rpm1, rpm2],
             [rpm2, rpm1]]
       
    unexplored = {}  # Dictionary of all unexplored nodes
    explored_coords = set()  # Set of explored coordinates
    
    start_coords = (start.x, start.y)  # Generating a unique key for identifying the node
    unexplored[start_coords] = start
    
    present_nodes = []  # Stores all nodes that have been traversed, for visualization purposes.
    parent_nodes = []
    
    while unexplored:
        # Select the node with the lowest combined cost and heuristic estimate
        present_coords = min(unexplored, key=lambda k: unexplored[k].cst)
        present_node = unexplored.pop(present_coords)
        
        if is_goal(present_node, goal):
            goal.parent = present_node.parent
            goal.cst = present_node.cst
            print("Goal Node found")
            return 1, present_nodes, parent_nodes

        explored_coords.add(present_coords)
        
        for move in moves:
            X1 = cost_fn(present_node.x, present_node.y, present_node.theta, move[0], move[1],
                            present_nodes, parent_nodes, clearance, robot_radius)
            
            if X1 is not None:

                c2g = math.dist((X1[0], X1[1]), (goal.x, goal.y))
                new_node = Node(X1[0], X1[1], present_node, X1[2], move[0], move[1], present_node.c2c + X1[3], c2g, present_node.c2c + X1[3] + c2g)
                new_coords = (new_node.x, new_node.y)
    
                if not is_valid(new_node.x, new_node.y, robot_radius, clearance) or new_coords in explored_coords:
                    continue
                
                if new_coords not in unexplored:
                    unexplored[new_coords] = new_node
                elif new_node.cst < unexplored[new_coords].cst:
                    unexplored[new_coords] = new_node
        
        # Explore nodes within a radius of 10 units from the current node
        for coord in unexplored.copy():
            if math.dist(coord, present_coords) <= 10:  # Check if within radius of 10 units
                explored_coords.add(coord)
                if is_goal(unexplored[coord], goal):  # Check if this node is the goal node
                    goal.parent = present_node.parent
                    goal.cst = present_node.cst + unexplored[coord].c2c
                    print("Goal Node found")
                    return 1, present_nodes, parent_nodes
                del unexplored[coord]


    return 0, present_nodes, parent_nodes


# Function to backtrack and generate shortest path
def backtrack(goal_node):  
    x_path = []
    y_path = []
    x_path.append(goal_node.x)
    y_path.append(goal_node.y)

    parent_node = goal_node.parent
    while parent_node != -1:
        x_path.append(parent_node.x)
        y_path.append(parent_node.y)
        parent_node = parent_node.parent
        
    x_path.reverse()
    y_path.reverse()
    
    return x_path, y_path


# Function to draw a vector
def draw_vector(screen, color, start, end):
    BLUE = (0, 0, 255)
    pygame.draw.line(screen, color, start, end, width=1)
    pygame.draw.circle(screen, BLUE, end, 2)

# Function to plot the path
def plot_path(start_node, goal_node, x_path, y_path, present_nodes, parent_nodes, clearance, frame_rate):
    BLUE = (0, 0, 255)
    RED = (255, 0, 0)
    WHITE = (255, 255, 255)
    LIGHT_GREY = (190, 190, 190)
    DARK_GREY = (100, 100, 100)

    padding = clearance
    center_x, center_y = 420, 120

    # Initialize Pygame and plot the map
    pygame.init()
    screen = pygame.display.set_mode((600, 200))
    clock = pygame.time.Clock()
    if not os.path.exists("frames"):
        os.makedirs("frames")

    frame_count = 0
    running = True
    while running and frame_count < len(present_nodes):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill(LIGHT_GREY)
        padding_rect = pygame.Rect(padding, padding, 600 - 2 * padding, 200 - 2 * padding) # Background Canvas
        pygame.draw.rect(screen, WHITE, padding_rect) # Original Canvas
        pygame.draw.circle(screen, LIGHT_GREY, (center_x, 200-center_y), 60 + padding)
        pygame.draw.circle(screen, DARK_GREY, (center_x, 200-center_y), 60)
        pygame.draw.rect(screen, LIGHT_GREY, pygame.Rect(150 - clearance, 0, 25 + 2 * clearance,
                                                         100 + clearance))  # Rectangle1 Clearance
        pygame.draw.rect(screen, LIGHT_GREY, pygame.Rect(250 - clearance, 100 - clearance, 25 + 2 * clearance,
                                                         100 + clearance))  # Rectangle2 Clearance
        pygame.draw.rect(screen, DARK_GREY, pygame.Rect(150, 0, 25, 100))  # Rectangle1 Obstacle
        pygame.draw.rect(screen, DARK_GREY, pygame.Rect(250, 100, 25, 100))  # Rectangle2 Obstacle
        pygame.draw.rect(screen, RED, (start_node.x, 200 - start_node.y, 3, 3))  # Invert y-axis for start node
        pygame.draw.rect(screen, RED, (goal_node.x, 200 - goal_node.y, 3, 3))  # Invert y-axis for goal node


        for i in range(len(present_nodes) - 1):
            present_node = present_nodes[i]
            parent_node = parent_nodes[i]
            start = (present_node[0], 200 - present_node[1])  # Invert y-axis for present node
            end = (parent_node[0], 200 - parent_node[1])  # Invert y-axis for parent node
            pygame.draw.line(screen, BLUE, start, end, width=1)
            pygame.draw.circle(screen, (0, 255, 0), start, -1)
            frame_count += 1
            if frame_count % 250 == 0:  # Save frame every 100th frame
                pygame.image.save(screen, os.path.join("frames", f"frame_{frame_count}.png"))
            pygame.display.update()

        
        for i in range(len(x_path) - 1):
            pygame.draw.line(screen, RED, (x_path[i], 200 - y_path[i]), (x_path[i + 1], 200 - y_path[i + 1]), width=4)
            pygame.image.save(screen, os.path.join("frames", f"frame_{frame_count}.png"))
            frame_count += 1
            pygame.display.update()

        clock.tick(frame_rate)  # Ensure frame rate

    pygame.quit()


def frames_to_video(frames_dir, output_video):
    frames = [img for img in os.listdir(frames_dir) if img.endswith(".png")]
    frames.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))  # Sort frames by frame number
    frame = cv2.imread(os.path.join(frames_dir, frames[0]))
    height, width, _ = frame.shape
    # print("Creating Videowriter")
    video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 35, (width, height))
    print("Writing Video")
    for frame in frames:
        video.write(cv2.imread(os.path.join(frames_dir, frame)))

    cv2.destroyAllWindows()
    video.release()

def scale_coordinates(input_x, input_y, canvas_width, canvas_height):
    # Scale coordinates to fit within canvas_width, canvas_height
    scaled_x = (input_x / canvas_width) * 600
    scaled_y = (input_y / canvas_height) * 200
    return scaled_x, scaled_y

if __name__ == '__main__':
    large_canvas_width = 6000
    large_canvas_height = 2000
    robot_radius = 22
    
    # Taking start and end node coordinates as input from the user
    CLEARANCE = int(input("Enter the desired CLEARANCE (1-10): "))/10
    RPM1 = int(input("Enter the desired left_wheel RPM: "))
    RPM2 = int(input("Enter the desired right_wheel RPM: "))

    start_input_x = int(input("Enter the Start X: "))
    start_input_y = int(input("Enter the Start Y: "))
    start_theta = int(input("Enter the Theta_Start: "))
    end_input_x = int(input("Enter the End X: "))
    end_input_y = int(input("Enter the End Y: "))
    
    start_x, start_y = scale_coordinates(start_input_x, start_input_y, large_canvas_width, large_canvas_height)
    end_x, end_y = scale_coordinates(end_input_x, end_input_y, large_canvas_width, large_canvas_height)
    
    if start_theta % 30 != 0:
        print("Please enter valid theta values. Theta should be a multiple of 30 degrees.")
        exit()

    print("Finding the optimal path......!!!!")
    c2g = math.dist((start_x,start_y), (end_x, end_y))
    cst =  c2g
    # Define start and goal nodes
    start_node = Node(start_x, start_y,-1,start_theta,0,0,0,c2g,cst)
    goal_node = Node(end_x, end_y, -1,0,0,0,c2g,0,cst)
    save_dir = "frames"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    timer_begin = time.time()
    flag,present_nodes,parent_nodes = a_star(start_node, goal_node,RPM1,RPM2, CLEARANCE, robot_radius)
    timer_end = time.time()
    print("Time taken to explore:", timer_end - timer_begin, "seconds")

    if flag:
        x_path, y_path = backtrack(goal_node)
        optimal_cost = goal_node.cst  # Cost of the optimal path
        print("Optimal path cost:", optimal_cost)
        plot_path(start_node, goal_node, x_path, y_path, present_nodes, parent_nodes, CLEARANCE, frame_rate=30)
        output_video = "output_video.mp4"
        print("Generating Video")
        frames_to_video(save_dir, output_video)
        print("Video created successfully!")
    else:
        print("Goal not found!")

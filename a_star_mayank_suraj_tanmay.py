#!/usr/bin/env python3
# Github repository:  https://github.com/MayankD409/A_star_algorithm_on_rigid_robot.git

import pygame
import numpy as np
import time
import heapq
import math
import os
import cv2

# Define a class to represent nodes in the search space
class Node:
    def __init__(self, x, y, parent,theta,UL,UR, c2c, c2g, total_cost):
        self.x = x
        self.y = y
        self.parent = parent
        self.theta = theta
        self.UL = UL
        self.UR = UR
        self.c2c = c2c
        self.c2g = c2g
        self.total_cost = total_cost
        
    def __lt__(self,other):
        return self.total_cost < other.total_cost

# Define possible actions and associated cost increments
def cost_fn(Xi, Yi, Thetai, UL, UR, Nodes_list, Path_list, clearance, robot_radius):
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
            Nodes_list.append((Xn, Yn))
            Path_list.append((Xs, Ys))
        else:
            return None
    
    Thetan = 180 * (Thetan) / 3.14
    return [Xn, Yn, Thetan, cost, Nodes_list, Path_list]


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
# def is_valid(x, y, robot_radius, clearance):

#     # Creating buffer space for obstacles    
#     rectangle1_buffer_vts = [(150 - (robot_radius + clearance), 200), (175 + (robot_radius + clearance), 200), (175 + (robot_radius + clearance), 100 - (robot_radius + clearance)), (150 - (robot_radius + clearance), 100 - (robot_radius + clearance))]
#     rectangle2_buffer_vts = [(250 - (robot_radius + clearance), 100 + (robot_radius + clearance)), (275 + (robot_radius + clearance), 100 - (robot_radius + clearance)), (275 + (robot_radius + clearance), 0), (250 - (robot_radius + clearance), 0)]

#     rect1_buffer = is_point_inside_rectangle(x,y, rectangle1_buffer_vts)
#     rect2_buffer = is_point_inside_rectangle(x, y, rectangle2_buffer_vts)
#     circ_buffer = is_point_inside_circle(x, y, 420, 120, 120 + 2*(robot_radius + clearance))
    
#     # Setting buffer space constraints to obtain obstacle space
#     if rect1_buffer or rect2_buffer or circ_buffer:
#         return False
    
#     # Adding check if obstacle is in walls
#     if x <= (robot_radius + clearance) or y >= 200 - (robot_radius + clearance) or x >= 600 - (robot_radius + clearance) or y <= (robot_radius + clearance):
#         return False

#     return True

def is_valid(x, y, radius, clearance):
    total_space = radius + clearance

    obstacle1 = ((np.square(x - 420)) + (np.square(y - 120)) <= np.square(60 + total_space))
    obstacle2 = (x >= 150 - total_space) and (x <= 175 + total_space) and (y >= 100 - total_space)
    obstacle3 = (x >= 250 - total_space) and (x <= 275 + total_space) and (y <= 100 + total_space)
 
    border1 = (x <= 0 + total_space)     
    border2 = (x >= 600 - total_space)
    border3 = (y <= 0 + total_space)
    border4 = (y >= 200 - total_space)

    if obstacle1 or obstacle2 or obstacle3 or border1 or border2 or border3 or border4:
        return False
    else:
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
# def a_star(start_position, goal_position, rpm1, rpm2, clearance, robot_radius):
#     if is_goal(start_position, goal_position):
#         return None, 1

#     goal = goal_position
#     start = start_position
    
#     moves = [[rpm1, 0], 
#              [0, rpm1], 
#              [rpm1, rpm1], 
#              [0, rpm2], 
#              [rpm2, 0], 
#              [rpm2, rpm2], 
#              [rpm1, rpm2],
#              [rpm2, rpm1]]
       
#     unexplored = {}  # Dictionary of all unexplored nodes
#     explored_coords = set()  # Set of explored coordinates
    
#     start_coords = (start.x, start.y)  # Generating a unique key for identifying the node
#     unexplored[start_coords] = start
    
#     Nodes_list = []  # Stores all nodes that have been traversed, for visualization purposes.
#     Path_list = []
    
#     while unexplored:
#         # Select the node with the lowest combined cost and heuristic estimate
#         present_coords = min(unexplored, key=lambda k: unexplored[k].total_cost)
#         present_node = unexplored.pop(present_coords)
        
#         if is_goal(present_node, goal):
#             goal.parent = present_node.parent
#             goal.total_cost = present_node.total_cost
#             print("Goal Node found")
#             return 1, Nodes_list, Path_list

#         explored_coords.add(present_coords)
        
#         for move in moves:
#             X1 = cost_fn(present_node.x, present_node.y, present_node.theta, move[0], move[1],
#                             Nodes_list, Path_list, clearance, robot_radius)
            
#             if X1 is not None:
#                 angle = X1[2]
#                 x = (round(X1[0] * 10) / 10)
#                 y = (round(X1[1] * 10) / 10)
#                 th = (round(angle / 15) * 15)

#                 c2g = math.dist((x, y), (goal.x, goal.y))
#                 new_node = Node(x, y, present_node, th, move[0], move[1], present_node.c2c + X1[3], c2g, present_node.c2c + X1[3] + c2g)
#                 new_coords = (new_node.x, new_node.y)
    
#                 if not is_valid(new_node.x, new_node.y, robot_radius, clearance) or new_coords in explored_coords:
#                     continue
                
#                 if new_coords not in unexplored:
#                     unexplored[new_coords] = new_node
#                 elif new_node.total_cost < unexplored[new_coords].total_cost:
#                     unexplored[new_coords] = new_node
        
#         # Explore nodes within a radius of 10 units from the current node
#         # for coord in unexplored.copy():
#         #     if math.dist(coord, present_coords) <= (robot_radius):
#         #         if is_goal(unexplored[coord], goal):
#         #             node = unexplored[coord]
#         #             goal.parent = node.parent
#         #             goal.total_cost = node.total_cost
#         #             print("Goal Node found")
#         #             return 1, Nodes_list, Path_list
#         #         explored_coords.add(coord)
#         #         del unexplored[coord]

#     return 0, Nodes_list, Path_list

def key(node):
    key = 1000 * node.x + 111 * node.y
    return key

def a_star(start_node, goal_node, rpm1, rpm2, clearance, radius):

    # Check if the goal node is reached 
    if is_goal(start_node, goal_node):
        return 1, None, None
    
    start_node = start_node
    start_node_id = key(start_node)
    goal_node = goal_node

    Nodes_list = []  # List to store all the explored nodes
    Path_list = []  # List to store the final path from start to goal node

    closed_node = {}  # Dictionary to store all the closed nodes
    open_node = {}  # Dictionary to store all the open nodes
    
    open_node[start_node_id] = start_node   # Add the start node to the open nodes dictionary

    priority_list = []  # Priority queue to store nodes based on their total cost
    
    # All the possible moves of the robot
    moves = [[rpm1, 0], 
             [0, rpm1], 
             [rpm1, rpm1], 
             [0, rpm2], 
             [rpm2, 0], 
             [rpm2, rpm2], 
             [rpm1, rpm2],
             [rpm2, rpm1]]

    # Push the start node into the priority queue with its total cost
    heapq.heappush(priority_list, [start_node.total_cost, start_node])

    while (len(priority_list) != 0):

        # Pop the node with the minimum cost from the priority queue
        current_nodes = (heapq.heappop(priority_list))[1]
        current_id = key(current_nodes)

        # Check if the popped node is the goal node
        if is_goal(current_nodes, goal_node):
            goal_node.parent = current_nodes.parent
            goal_node.total_cost = current_nodes.total_cost
            print("Goal Node found")
            return 1, Nodes_list, Path_list
        
        # Add the popped node to the closed nodes dictionary
        if current_id in closed_node:  
            continue
        else:
            closed_node[current_id] = current_nodes
        
        del open_node[current_id]
        
        # Loop through all the possible moves
        for move in moves:
            action = cost_fn(current_nodes.x, current_nodes.y, current_nodes.theta, move[0], move[1],
                            Nodes_list, Path_list, clearance, robot_radius)
           
            # Check if the move is valid
            if (action != None):
                angle = action[2]
                
                # Round off the coordinates and the angle to nearest integer
                theta_lim = 30
                x = (round(action[0] * 10) / 10)
                y = (round(action[1] * 10) / 10)
                theta = (round(angle / theta_lim) * theta_lim)
                
                # Calculate the new orientation and the cost to move to the new node
                c2g = math.dist((x,y), (goal_node.x, goal_node.y))
                new_node = Node(x, y, current_nodes, theta, move[0], move[1], current_nodes.c2c+action[3], c2g, current_nodes.c2c+action[3]+c2g)

                new_node_id = key(new_node)
                
                # Check if the new node is valid and has not already been visited
                if not is_valid(new_node.x, new_node.y, radius, clearance):
                    continue
                elif new_node_id in closed_node:
                    continue

                # Update the node information if it already exists in the open list
                if new_node_id in open_node:
                    if new_node.total_cost < open_node[new_node_id].total_cost:
                        open_node[new_node_id].total_cost = new_node.total_cost
                        open_node[new_node_id].parent = new_node

                # Add the new node to the open list if it doesn't already exist        
                else:
                    open_node[new_node_id] = new_node
                    heapq.heappush(priority_list, [ open_node[new_node_id].total_cost, open_node[new_node_id]])
            
    return 0, Nodes_list, Path_list

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
def plot_path(start_node, goal_node, x_path, y_path, Nodes_list, Path_list, clearance, frame_rate):
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
    while running and frame_count < len(Nodes_list):
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


        for i in range(len(Nodes_list) - 1):
            present_node = Nodes_list[i]
            parent_node = Path_list[i]
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

    # Define start and goal nodes
    c2g = math.dist((start_x,start_y), (end_x, end_y))
    total_cost =  c2g
    start_node = Node(start_x, start_y,-1,start_theta,0,0,0,c2g,total_cost)
    goal_node = Node(end_x, end_y, -1,0,0,0,c2g,0,total_cost)
    save_dir = "frames"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    timer_begin = time.time()
    flag,Nodes_list,Path_list = a_star(start_node, goal_node,RPM1,RPM2, CLEARANCE, 22)
    timer_end = time.time()
    print("Time taken to explore:", timer_end - timer_begin, "seconds")

    if flag:
        x_path, y_path = backtrack(goal_node)
        optimal_cost = goal_node.total_cost  # Cost of the optimal path
        print("Optimal path cost:", optimal_cost)
        plot_path(start_node, goal_node, x_path, y_path, Nodes_list, Path_list, CLEARANCE, frame_rate=30)
        output_video = "output_video.mp4"
        print("Generating Video")
        frames_to_video(save_dir, output_video)
        print("Video created successfully!")
    else:
        print("Goal not found!")

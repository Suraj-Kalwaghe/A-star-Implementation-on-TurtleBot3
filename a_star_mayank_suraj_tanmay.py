
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
    def __init__(self, x, y, parent,current_theta, change_theta,UL,UR, c2c, c2g, total_cost ):
        self.x = x
        self.y = y
        self.parent = parent
        self.current_theta = current_theta
        self.change_theta = change_theta
        self.UL = UL
        self.UR = UR
        self.c2c = c2c
        self.c2g = c2g
        self.total_cost = total_cost 
        
    def __lt__(self,other):
        return self.total_cost < other.total_cost

# Define possible actions and associated cost increments
def plot_curve(Xi, Yi, Thetai, UL, UR,c, plot, Nodes_list, Path_list, obs_space):
    t = 0
    r = 0.038
    L = 0.354
    dt = 0.1
    cost = 0
    Xn = Xi
    Yn = Yi
    Thetan = 3.14 * Thetai / 180

    # Xi, Yi,Thetai: Input point's coordinates
    # Xs, Ys: Start point coordinates for plot function
    # Xn, Yn, Thetan: End point coordintes

    while t < 1:
        t = t + dt
        Xs = Xn
        Ys = Yn
        Xn += r*0.5 * (UL + UR) * math.cos(Thetan) * dt
        Yn += r*0.5 * (UL + UR) * math.sin(Thetan) * dt
        Thetan += (r / L) * (UR - UL) * dt
        if  is_valid(Xn, Yn, obs_space):
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

# Function to check if a point is inside a hexagon
def is_point_inside_hexagon(x, y, center_x, center_y, side_length):
    cx, cy = center_x, center_y
    vertices = []
    angle_deg = 60
    angle_rad = math.radians(angle_deg)
    for i in range(6):
        px = cx + side_length * math.cos(angle_rad * i + math.radians(30))
        py = cy + side_length * math.sin(angle_rad * i + math.radians(30))
        vertices.append((px, py))
    odd_nodes = False
    j = 5
    for i in range(6):
        if (vertices[i][1] < y and vertices[j][1] >= y) or (vertices[j][1] < y and vertices[i][1] >= y):
            if (vertices[i][0] + (y - vertices[i][1]) / (vertices[j][1] - vertices[i][1]) * (vertices[j][0] - vertices[i][0])) < x:
                odd_nodes = not odd_nodes
        j = i
    return odd_nodes

# Function to check if a point is inside a C-shaped block
def is_point_inside_block(point, vertices):
    odd_nodes = False
    j = len(vertices) - 1
    for i in range(len(vertices)):
        if (vertices[i][1] < point[1] and vertices[j][1] >= point[1]) or (vertices[j][1] < point[1] and vertices[i][1] >= point[1]):
            if (vertices[i][0] + (point[1] - vertices[i][1]) / (vertices[j][1] - vertices[i][1]) * (vertices[j][0] - vertices[i][0])) < point[0]:
                odd_nodes = not odd_nodes
        j = i
    return odd_nodes

# Function to create configuration space with obstacles
def Configuration_space(width, height, robot_radius, clearance):
    obs_space = np.full((height, width), 0)
    
    for y in range(height):
        for x in range(width):
            # Creating buffer space for obstacles    
            rectangle1_buffer_vts = [(50 - (robot_radius + clearance), 250), (87.5 + (robot_radius + clearance), 250), (87.5 + (robot_radius + clearance), 50 - (robot_radius + clearance)), (50 - (robot_radius + clearance), 50 - (robot_radius + clearance))]
            rectangle2_buffer_vts = [(137.5 - (robot_radius + clearance), 200 + (robot_radius + clearance)), (175 + (robot_radius + clearance), 200 - (robot_radius + clearance)), (175 + (robot_radius + clearance), 0), (137.5 - (robot_radius + clearance), 0)]
            cblock_buffer_vts = [(450 - (robot_radius + clearance), 225 + (robot_radius + clearance)), (450 - (robot_radius + clearance), 187.5 - (robot_radius + clearance)), (510 - (robot_radius + clearance), 187.5 - (robot_radius + clearance)), (510 - (robot_radius + clearance), 62.5 + (robot_radius + clearance)), (450 - (robot_radius + clearance), 62.5 + (robot_radius + clearance)), 
                                 (450 - (robot_radius + clearance), 25 - (robot_radius + clearance)), (550 + (robot_radius + clearance), 25 - (robot_radius + clearance)), (550 + (robot_radius + clearance), 225 + (robot_radius + clearance))]

            rect1_buffer = is_point_inside_rectangle(x,y, rectangle1_buffer_vts)
            rect2_buffer = is_point_inside_rectangle(x, y, rectangle2_buffer_vts)
            hexa_buffer = is_point_inside_hexagon(x, y, 325, 125, 75 + (robot_radius + clearance))
            cblock_buffer = is_point_inside_block((x, y), cblock_buffer_vts)
            
            # Setting buffer space constraints to obtain obstacle space
            if cblock_buffer or rect1_buffer or rect2_buffer or hexa_buffer:
                obs_space[y, x] = 1
             
            # Plotting actual obstacle space using half-plane equations
            rectangle1_vts = [(50, 250), (87.5, 250), (87.5, 50), (50, 50)]
            rectangle2_vts = [(137.5, 200), (175, 200), (175, 0), (137.5, 0)]
            cblock_vertices = [(450, 225), (450, 187.5), (510, 187.5), (510, 62.5), (450, 62.5), (450, 25), (550, 25), (550, 225)]

            rect1 = is_point_inside_rectangle(x, y, rectangle1_vts)
            rect2 = is_point_inside_rectangle(x, y, rectangle2_vts)
            hexa = is_point_inside_hexagon(x, y, 325, 125, 75)
            cblock = is_point_inside_block((x, y), cblock_vertices)

            # Setting the constraints to obtain the obstacle space without buffer
            if cblock or rect1 or rect2 or hexa:
                obs_space[y, x] = 2
                
    # Adding clearance for walls
    for i in range(height):
        for j in range(3 + (robot_radius + clearance)):
            obs_space[i][j] = 1
            obs_space[i][width - j - 1] = 1

    for i in range(width):
        for j in range(3 + (robot_radius + clearance)):  
            obs_space[j][i] = 1
            obs_space[height - j - 1][i] = 1 

    return obs_space

# Function to check if a move is valid
def is_valid(x, y, obs_space):
    height, width = obs_space.shape
    
    # Check if coordinates are within the boundaries of the obstacle space and not occupied by an obstacle
    if x < 0 or x >= width or y < 0 or y >= height or obs_space[round(y)][round(x)] == 1 or obs_space[round(y)][round(x)] == 2:
        return False
    
    return obs_space[round(y), round(x)] == 0

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

# Function to check if the goal node is reached
def is_goal(present, goal):
    dt = euclidean_distance((present.x, present.y), (goal.x, goal.y))             
    if dt <= 1.5:
        return True
    else:
        return False
    
# A* algorithm implementation
def key(node):
    key = 1022*node.x + 111*node.y 
    return key

### Function to implement A star ########
def Astar_algorithm(start_node, goal_node,rpm1,rpm2,r,clearance, obs_space):

    if is_goal(start_node, goal_node):
        return 1,None,None
    
    start_node = start_node
    start_node_id = key(start_node)
    goal_node = goal_node

    Nodes_list = []
    Path_list = []
    
    explored_nodes = {}
    unexplored_nodes = {}
    
    unexplored_nodes[start_node_id] = start_node
    
    
    priority_list = []
    
    moves = [[rpm1, 0], 
             [0, rpm1], 
             [rpm1, rpm1], 
             [0, rpm2], 
             [rpm2, 0], 
             [rpm2, rpm2], 
             [rpm1, rpm2],
             [rpm2, rpm1]]
    
    heapq.heappush(priority_list, [start_node.total_cost, start_node])

    while (len(priority_list) != 0):

        present_node = (heapq.heappop(priority_list))[1]
        current_id = key(present_node)

        if is_goal(present_node, goal_node):
            goal_node.parent = present_node.parent
            goal_node.total_cost = present_node.total_cost
            print("Goal Node found")
            return 1,Nodes_list,Path_list
        
        if current_id in explored_nodes:  
            continue
        else:
            explored_nodes[current_id] = present_node
        
        del unexplored_nodes[current_id]
        

        for move in moves:
            X1 = plot_curve(present_node.x, present_node.y, present_node.current_theta, move[0], move[1],
                            clearance, 0, Nodes_list, Path_list, obs_space)
           
            
            if (X1 != None):
                angle = X1[2]
                
                theta_threshold = 15
                x = (round(X1[0] * 10) / 10)
                y = (round(X1[1] * 10) / 10)
                th = (round(angle / theta_threshold) * theta_threshold)
                ct = present_node.change_theta - th
                
                c2g = math.dist((x,y), (goal_node.x, goal_node.y))
                new_node = Node(x,y,present_node,th,ct,move[0],move[1],present_node.c2c+X1[3],c2g,present_node.c2c+X1[3]+c2g)

                new_node_id = key(new_node)

                if not is_valid(new_node.x, new_node.y,obs_space):
                    continue
                elif new_node_id in explored_nodes:
                    continue
                if new_node_id in unexplored_nodes:
                    if new_node.total_cost < unexplored_nodes[new_node_id].total_cost:
                        unexplored_nodes[new_node_id].total_cost = new_node.total_cost
                        unexplored_nodes[new_node_id].parent = new_node
                else:
                    unexplored_nodes[new_node_id] = new_node
                    heapq.heappush(priority_list, [ unexplored_nodes[new_node_id].total_cost, unexplored_nodes[new_node_id]])
            
    return 0,Nodes_list,Path_list

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

# Function to draw a hexagon
def draw_hexagon(screen, color, center_x, center_y, side_length):
    vertices = []
    angle_deg = 60
    angle_rad = math.radians(angle_deg)
    for i in range(6):
        x = center_x + side_length * math.cos(angle_rad * i + math.radians(30))  # Adding 30 degrees to start with vertex up
        y = center_y + side_length * math.sin(angle_rad * i + math.radians(30))
        vertices.append((x, y))
    pygame.draw.polygon(screen, color, vertices)

# Function to draw a hexagon with padding
def draw_padded_hexagon(screen, color, center_x, center_y, side_length, padding):
    enlarged_side_length = side_length + padding
    draw_hexagon(screen, color, center_x, center_y, enlarged_side_length)

# Function to draw C obstacle
def draw_C(screen, color):
    vertices = [(450, 225), (450, 187.5), (510, 187.5), (510, 62.5), (450, 62.5), (450, 25), (550, 25), (550, 225)]
    pygame.draw.polygon(screen, color, vertices)

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
    center_x, center_y = 325, 125
    side_length = 75

    # Initialize Pygame and plot the map
    pygame.init()
    screen = pygame.display.set_mode((600, 250))
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
        padding_rect = pygame.Rect(padding, padding, 600 - 2 * padding, 250 - 2 * padding)
        pygame.draw.rect(screen, WHITE, padding_rect)
        draw_padded_hexagon(screen, LIGHT_GREY, center_x, center_y, side_length, padding)
        draw_hexagon(screen, DARK_GREY, center_x, center_y, side_length)
        cblock_vertices = [(450 - (clearance), 225 + (clearance)), (450 - (clearance), 187.5 - (clearance)),
                           (510 - (clearance), 187.5 - (clearance)), (510 - (clearance), 62.5 + (clearance)),
                           (450 - (clearance), 62.5 + (clearance)),
                           (450 - (clearance), 25 - (clearance)), (550 + (clearance), 25 - (clearance)),
                           (550 + (clearance), 225 + (clearance))]
        pygame.draw.polygon(screen, LIGHT_GREY, cblock_vertices)
        draw_C(screen, DARK_GREY)
        pygame.draw.rect(screen, LIGHT_GREY, pygame.Rect(50 - clearance, 0, 37.5 + 2 * clearance,
                                                         200 + clearance))  # Rectangle1 Clearance
        pygame.draw.rect(screen, LIGHT_GREY, pygame.Rect(137.5 - clearance, 50 - clearance, 37.5 + 2 * clearance,
                                                         200 + clearance))  # Rectangle2 Clearance
        pygame.draw.rect(screen, DARK_GREY, pygame.Rect(50, 0, 37.5, 200))  # Rectangle1 Obstacle
        pygame.draw.rect(screen, DARK_GREY, pygame.Rect(137.5, 50, 37.5, 200))  # Rectangle2 Obstacle
        pygame.draw.rect(screen, RED, (start_node.x, 250 - start_node.y, 4, 4))  # Invert y-axis for start node
        pygame.draw.rect(screen, RED, (goal_node.x, 250 - goal_node.y, 4, 4))  # Invert y-axis for goal node
        for i in range(len(Nodes_list) - 1):
            present_node = Nodes_list[i]
            parent_node = Path_list[i]
            start = (present_node[0], 250 - present_node[1])  # Invert y-axis for present node
            end = (parent_node[0], 250 - parent_node[1])  # Invert y-axis for parent node
            pygame.draw.line(screen, BLUE, start, end, width=1)
            pygame.draw.circle(screen, (0, 255, 0), start, 2)
            frame_count += 1
            if frame_count % 250 == 0:  # Save frame every 100th frame
                pygame.image.save(screen, os.path.join("frames", f"frame_{frame_count}.png"))
            pygame.display.update()
        for i in range(len(x_path) - 1):
            pygame.draw.line(screen, RED, (x_path[i], 250 - y_path[i]), (x_path[i + 1], 250 - y_path[i + 1]), width=4)
            pygame.image.save(screen, os.path.join("frames", f"frame_{frame_count}.png"))
            frame_count += 1
            pygame.display.update()

        clock.tick(frame_rate)  # Ensure frame rate

    pygame.quit()


def frames_to_video(frames_dir, output_video):
    frames = [img for img in os.listdir(frames_dir) if img.endswith(".png")]
    frames.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))  # Sort frames by frame number
    frame = cv2.imread(os.path.join(frames_dir, frames[0]))
    height, width, layers = frame.shape
    # print("Creating Videowriter")
    video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 35, (width, height))
    print("Writing Video")
    for frame in frames:
        video.write(cv2.imread(os.path.join(frames_dir, frame)))

    cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':
    width = 600
    height = 250
    RPM1, RPM2 = 50, 50
    c2g = 0
    
    # Taking start and end node coordinates as input from the user
    CLEARANCE = int(input("Enter the desired CLEARANCE: "))
    robot_radius = 1
    robot_step_size = 2
    print("Minimum x and y values will be addition of clearance (of wall) and robot radius:",CLEARANCE+robot_radius)
    start_input_x = input("Enter the Start X: ")
    start_input_y = input("Enter the Start Y: ")
    start_theta = int(input("Enter the Theta_Start: "))

    start_x = int(start_input_x)
    start_y = int(start_input_y)

    end_input_x = input("Enter the End X: ")
    end_input_y = input("Enter the End Y: ")
    
    end_x = int(end_input_x)
    end_y = int(end_input_y)
    
    if start_theta % 30 != 0:
        print("Please enter valid theta values. Theta should be a multiple of 30 degrees.")
        exit()

    if robot_step_size < 1 or robot_step_size > 10:
        print("Please enter a valid step size between 1 to 10 inclusive.")
        exit()

    print("Setting up Configuration space. Wait a few seconds....")
    obs_space = Configuration_space(width, height, robot_radius, CLEARANCE)
    # Define start and goal nodes
    c2g = math.dist((start_x,start_y), (end_x, end_y))
    total_cost =  c2g
    start_node = Node(start_x, start_y,-1,start_theta,0,0,0,0,c2g,total_cost)
    goal_node = Node(end_x, end_y, -1,0,0,0,0,c2g,0,total_cost)
    save_dir = "frames"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    timer_begin = time.time()
    flag,Nodes_list,Path_list = Astar_algorithm(start_node, goal_node,RPM1,RPM2,robot_radius,CLEARANCE, obs_space)
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

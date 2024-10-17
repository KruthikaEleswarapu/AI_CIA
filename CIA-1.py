import tkinter as tk
from tkinter import messagebox
from collections import deque
import heapq

# Define global variables
adjacency_matrix_entries = []
heuristic_entries = []

class Node:
    def __init__(self, vertex, cost):
        self.vertex = vertex
        self.cost = cost

class PriorityQueue:
    def __init__(self):
        self.heap = []
    
    def enqueue(self, vertex, cost):
        node = Node(vertex, cost)
        self.heap.append(node)
        self._heapify_up(len(self.heap) - 1)

    def dequeue(self):
        if not self.heap:
            return None
        self._swap(0, len(self.heap) - 1)
        min_node = self.heap.pop()
        self._heapify_down(0)
        return min_node

    def _heapify_up(self, index):
        while index > 0:
            parent_index = (index - 1) // 2
            if self.heap[index].cost < self.heap[parent_index].cost:
                self._swap(index, parent_index)
                index = parent_index
            else:
                break

    def _heapify_down(self, index):
        while index < len(self.heap):
            smallest = index
            left = 2 * index + 1
            right = 2 * index + 2

            if left < len(self.heap) and self.heap[left].cost < self.heap[smallest].cost:
                smallest = left
            if right < len(self.heap) and self.heap[right].cost < self.heap[smallest].cost:
                smallest = right
            if smallest != index:
                self._swap(index, smallest)
                index = smallest
            else:
                break

    def _swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def is_empty(self):
        return len(self.heap) == 0

# Hill Climbing Algorithm (simplified implementation)
def hill_climbing(graph, heuristic, start, goal, n):
    current = start
    total_cost = 0
    visited = [False] * n
    path = [current]  # Initialize path

    result = f"Starting Hill Climbing from node {start}\n"
    visited[current] = True

    while current != goal:
        result += f"Currently at node {current}\n"
        next_node = -1
        min_heuristic = float('inf')

        for i in range(n):
            if graph[current][i] != 0 and not visited[i]:
                if heuristic[i] < min_heuristic:
                    min_heuristic = heuristic[i]
                    next_node = i

        if next_node == -1:
            result += f"No better neighbors found. Stuck at node {current}.\n"
            return result

        result += f"Moving to node {next_node} with heuristic {min_heuristic}\n"
        total_cost += graph[current][next_node]
        visited[next_node] = True
        path.append(next_node)  # Add to path
        current = next_node

    result += f"Goal node {goal} reached with total cost: {total_cost}\n"
    result += f"Path: {' -> '.join(map(str, path))}\n"
    return result

def branch_and_bound_heuristic_cost(graph, heuristic, start, goal, n):
    visited = [False] * n
    queue = [(0 + heuristic[start], start, 0, [start])]  # (f(n), node, g(n), path)
    best_cost = float('inf')
    best_path = []

    while queue:
        queue.sort()  # Sort by f(n)
        f_n, current, g_cost, path = queue.pop(0)  # Get the best node

        if current == goal:
            if g_cost < best_cost:
                best_cost = g_cost
                best_path = path
            continue

        visited[current] = True
        for i in range(n):
            if graph[current][i] > 0 and not visited[i]:
                new_cost = g_cost + graph[current][i]
                queue.append((new_cost + heuristic[i], i, new_cost, path + [i]))

    if best_path:
        return f"Goal node {goal} reached with total cost: {best_cost}\nPath: {' -> '.join(map(str, best_path))}\n"
    return "Goal not reachable\n"

def oracle_search(graph, heuristic, start, goal, n):
    g_cost = [float('inf')] * n
    visited = [False] * n
    parent = [-1] * n
    g_cost[start] = 0
    queue = [(g_cost[start] + heuristic[start], start)]  # (f(n), node)

    while queue:
        queue.sort()  # Sort by f(n)
        _, current = queue.pop(0)

        if visited[current]:
            continue

        visited[current] = True

        if current == goal:
            path = []
            while current != -1:
                path.append(current)
                current = parent[current]
            path.reverse()
            return f"Goal node {goal} reached with total cost: {g_cost[goal]}\nPath: {' -> '.join(map(str, path))}\n"

        for i in range(n):
            if graph[current][i] != 0 and not visited[i]:
                new_cost = g_cost[current] + graph[current][i]
                if new_cost < g_cost[i]:
                    g_cost[i] = new_cost
                    parent[i] = current
                    queue.append((g_cost[i] + heuristic[i], i))

    return "Goal not reachable\n"

def best_first_search(graph, heuristic, start, goal, n):
    pq = PriorityQueue()
    visited = [False] * n
    pq.enqueue(start, 0)
    visited[start] = True
    result = f"Starting Best First Search from node {start}\n"

    while not pq.is_empty():
        current = pq.dequeue()
        current_vertex = current.vertex
        result += f"Visiting node {current_vertex}\n"

        if current_vertex == goal:
            result += "Goal reached\n"
            return result

        for i in range(n):
            if graph[current_vertex][i] > 0 and not visited[i]:
                pq.enqueue(i, current.cost + graph[current_vertex][i])
                visited[i] = True

    result += "Goal not reachable\n"
    return result

def a_star_search(graph, heuristic, start, goal, n):
    g_cost = [float('inf')] * n
    visited = [False] * n
    parent = [-1] * n
    g_cost[start] = 0
    queue = [(g_cost[start] + heuristic[start], start)]  # (f(n), node)

    while queue:
        queue.sort()  # Sort by f(n)
        _, current = queue.pop(0)

        if visited[current]:
            continue

        visited[current] = True

        if current == goal:
            path = []
            while current != -1:
                path.append(current)
                current = parent[current]
            path.reverse()
            return f"Goal node {goal} reached with total cost: {g_cost[goal]}\nPath: {' -> '.join(map(str, path))}\n"

        for i in range(n):
            if graph[current][i] > 0 and not visited[i]:
                new_cost = g_cost[current] + graph[current][i]
                if new_cost < g_cost[i]:
                    g_cost[i] = new_cost
                    parent[i] = current
                    queue.append((g_cost[i] + heuristic[i], i))

    return "Goal not reachable\n"

def branch_and_bound(graph, start, goal, n):
    visited = [False] * n
    queue = [(0, start, [start])]  # (g(n), node, path)
    best_cost = float('inf')
    best_path = []

    while queue:
        queue.sort()  # Sort by g(n)
        g_n, current, path = queue.pop(0)

        if current == goal:
            if g_n < best_cost:
                best_cost = g_n
                best_path = path
            continue

        visited[current] = True

        for i in range(n):
            if graph[current][i] > 0 and not visited[i]:
                queue.append((g_n + graph[current][i], i, path + [i]))

    if best_path:
        return f"Goal node {goal} reached with total cost: {best_cost}\nPath: {' -> '.join(map(str, best_path))}\n"
    return "Goal not reachable\n"

def british_museum_search(graph, start, goal, n):
    visited = [False] * n
    stack = [(start, [start])]  # Stack holds tuples of (current node, path to this node)

    result = f"Starting British Museum Search from node {start}\n"
    all_paths = []  # List to store all paths that reach the goal

    while stack:
        current, path = stack.pop()

        if current == goal:
            result += f"Goal node {goal} reached!\n"
            all_paths.append(path)  # Store the path that led to the goal
            continue  # Continue to find other paths

        visited[current] = True
        for i in range(n):
            if graph[current][i] > 0 and not visited[i]:
                stack.append((i, path + [i]))  # Append the new path

    if all_paths:
        result += "All paths to the goal:\n"
        for p in all_paths:
            result += f"Path: {' -> '.join(map(str, p))}\n"
    else:
        result += "Goal not reachable\n"

    return result

def dfs(graph, start, goal, n):
    visited = [False] * n
    stack = [(start, [start])]  # Stack holds tuples of (current node, path)

    result = f"Starting DFS from node {start}\n"

    while stack:
        current, path = stack.pop()

        if current == goal:
            return f"Goal node {goal} reached\nPath: {' -> '.join(map(str, path))}\n"

        visited[current] = True
        for i in range(n):
            if graph[current][i] > 0 and not visited[i]:
                stack.append((i, path + [i]))

    return "Goal not reachable\n"

def bfs(graph, start, goal, n):
    visited = [False] * n
    queue = [(start, [start])]  # Queue holds tuples of (current node, path)

    result = f"Starting BFS from node {start}\n"

    while queue:
        current, path = queue.pop(0)

        if current == goal:
            return f"Goal node {goal} reached\nPath: {' -> '.join(map(str, path))}\n"

        visited[current] = True
        for i in range(n):
            if graph[current][i] > 0 and not visited[i]:
                queue.append((i, path + [i]))

    return "Goal not reachable\n"


def beam_search(graph, heuristic, start, goal, n, beam_width=2):
    visited = [False] * n
    queue = [(heuristic[start], start, [start], 0)]  # (h(n), node, path, g(n))

    while queue:
        # Sort queue by heuristic values
        queue.sort(key=lambda x: x[0])  # Sort by h(n)
        queue = queue[:beam_width]  # Limit the number of paths to beam width

        next_queue = []
        for h_n, current, path, g_n in queue:
            if current == goal:
                return f"Goal node {goal} reached with total cost: {g_n}\nPath: {' -> '.join(map(str, path))}\n"

            visited[current] = True
            for i in range(n):
                if graph[current][i] > 0 and not visited[i]:
                    next_cost = g_n + graph[current][i]  # Update the cost
                    next_queue.append((heuristic[i], i, path + [i], next_cost))

        queue = next_queue

    return "Goal not reachable\n"

def heuristic(node, goal):
    return abs(goal - node)

d = float('inf')  # Represents no connection or infinite distance

# Branch and Bound + Dead Horse Algorithm
def branch_and_bound_dead_horse(graph, start, goal, n, gui_text):
    # Priority queue for Branch and Bound (min-heap)
    pq = []
    
    # Initial node: (cost, current_node, path)
    heapq.heappush(pq, (0, start, [start]))  # Starting node with cost 0
    
    # To track the best cost to reach each node (oracle cost)
    best_cost = {i: d for i in range(n)}
    best_cost[start] = 0
    
    while pq:
        # Get the node with the lowest cost
        current_cost, current_node, path = heapq.heappop(pq)
        
        # Print expanded node in GUI
        gui_text.insert(tk.END, f"Expanding node {current_node} with cost {current_cost}\n")
        
        # If the current node is the goal, print success and return the path
        if current_node == goal:
            gui_text.insert(tk.END, f"Goal reached with path: {path} and cost: {current_cost}\n")
            return path
        
        # Dead Horse check: if current node's cost exceeds oracle, eliminate it
        if current_cost > best_cost[current_node]:
            gui_text.insert(tk.END, f"Eliminating node {current_node} - cost exceeds oracle ({current_cost} > {best_cost[current_node]})\n")
            continue
        
        # Explore neighbors of the current node
        for neighbor in range(n):
            # Adjust the condition to check for no connection (0 represents no path except for self-loops)
            if neighbor != current_node and graph[current_node][neighbor] != 0:  
                new_cost = current_cost + graph[current_node][neighbor]
                
                # If the neighbor has already been visited with a better cost
                if new_cost >= best_cost[neighbor]:
                    gui_text.insert(tk.END, f"Eliminating node {neighbor} - already visited with better cost ({new_cost} >= {best_cost[neighbor]})\n")
                    continue
                
                # Update the best cost for the neighbor
                best_cost[neighbor] = new_cost
                
                # Add the neighbor to the priority queue
                heapq.heappush(pq, (new_cost, neighbor, path + [neighbor]))
    
    # If the goal is not reached, return None
    gui_text.insert(tk.END, "No path found to the goal\n")
    return None

def heuristic(node, goal):
    return abs(goal - node)

# AO* Algorithm logic with node expansion details
def ao_star(graph, node, visited, num_vertices, goal):
    visited[node] = True
    
    best_successor = -1
    min_cost = INF
    
    # Print the node being expanded
    print(f"Expanding node {node}...")
    
    # Check all successors of the current node
    for i in range(num_vertices):
        if graph[node][i] != INF and not visited[i]:
            cost = graph[node][i] + heuristic(i, goal)
            if cost < min_cost:
                min_cost = cost
                best_successor = i
    
    if best_successor != -1:
        # Print the best successor node and its cost
        print(f"Best successor of node {node} is node {best_successor} with cost {min_cost}")
        ao_star(graph, best_successor, visited, num_vertices, goal)
    else:
        print(f"Goal reached at node {node}")

def reconstruct_path(parent, goal):
    """ Reconstruct the path from start to goal using the parent array. """
    path = []
    while goal != -1:
        path.append(goal)
        goal = parent[goal]
    return " -> ".join(map(str, reversed(path)))

# Function to dynamically generate the input fields for adjacency matrix and heuristic
def generate_input_fields():
    try:
        n = int(nodes_entry.get())
    except ValueError:
        messagebox.showerror("Invalid input", "Number of nodes must be an integer")
        return

    # Clear previous entries if any
    for entry in adjacency_matrix_entries:
        entry.destroy()
    adjacency_matrix_entries.clear()

    for entry in heuristic_entries:
        entry.destroy()
    heuristic_entries.clear()

    # Adjacency matrix input fields
    adjacency_matrix_label.config(text=f"Enter adjacency matrix (for {n}x{n} matrix):")
    for i in range(n):
        entry = tk.Entry(window)
        adjacency_matrix_entries.append(entry)
        entry.pack()

    # Heuristic input field (1 row of n values)
    heuristic_label.config(text=f"Enter heuristic values (space-separated, {n} values):")
    heuristic_entry = tk.Entry(window)
    heuristic_entries.append(heuristic_entry)
    heuristic_entry.pack()

# Function to run the selected algorithm
def run_algorithm():
    try:
        n = int(nodes_entry.get())
        start = int(start_entry.get())
        goal = int(goal_entry.get())
    except ValueError:
        messagebox.showerror("Invalid input", "Please enter valid integers for nodes, start, and goal")
        return

    # Get adjacency matrix
    graph = []
    for i in range(n):
        try:
            row = list(map(int, adjacency_matrix_entries[i].get().split()))
            if len(row) != n:
                raise ValueError("Invalid matrix dimensions")
            graph.append(row)
        except ValueError:
            messagebox.showerror("Invalid input", f"Row {i+1} must contain {n} integers")
            return

    # Get heuristic values
    try:
        heuristic = list(map(int, heuristic_entries[0].get().split()))
        if len(heuristic) != n:
            raise ValueError("Invalid number of heuristic values")
    except ValueError:
        messagebox.showerror("Invalid input", f"Heuristic must contain {n} integers")
        return

    result = ""
    if selected_algorithm.get() == "Hill Climbing":
        result = hill_climbing(graph, heuristic, start, goal, n)
    elif selected_algorithm.get() == "Branch and Bound + Heuristic + Cost":
        result = branch_and_bound_heuristic_cost(graph, heuristic, start, goal, n)
    elif selected_algorithm.get() == "Oracle Search":
        result = oracle_search(graph, heuristic, start, goal, n)
    elif selected_algorithm.get() == "Best First Search":
        result = best_first_search(graph, heuristic, start, goal, n)
    elif selected_algorithm.get() == "A* Search":
        result = a_star_search(graph, heuristic, start, goal, n)
    elif selected_algorithm.get() == "Branch and Bound":
        result = branch_and_bound(graph, start, goal, n)
    elif selected_algorithm.get() == "British Museum Search":
        result = british_museum_search(graph, start, goal, n)
    elif selected_algorithm.get() == "AO*":
        visited = [False] * n
        result = ao_star(graph, start, visited, n, goal)
    elif selected_algorithm.get() == "DFS":
        result = dfs(graph, start, goal, n)
    elif selected_algorithm.get() == "BFS":
        result = bfs(graph, start, goal, n)
    elif selected_algorithm.get() == "Branch and Bound + Dead Horse Principle":
        text_box = tk.Text(window, height=20, width=50)
        text_box.pack()
        result = branch_and_bound_dead_horse(graph, start, goal, n, text_box)
    elif selected_algorithm.get() == "Beam Search":
        result = beam_search(graph, heuristic, start, goal, n)

    messagebox.showinfo("Result", result)

# GUI Setup
window = tk.Tk()
window.title("Search Algorithm GUI")

selected_algorithm = tk.StringVar(window)
selected_algorithm.set("Hill Climbing")

algorithm_label = tk.Label(window, text="Select Algorithm:")
algorithm_label.pack()

algorithm_menu = tk.OptionMenu(window, selected_algorithm, "Hill Climbing", "Branch and Bound + Heuristic + Cost", "Oracle Search", "Best First Search", "A* Search", "Branch and Bound", "British Museum Search", "AO*", "DFS", "BFS", "Branch and Bound + Dead Horse Principle", "Beam Search")
algorithm_menu.pack()

# Number of nodes input
nodes_label = tk.Label(window, text="Enter number of nodes:")
nodes_label.pack()
nodes_entry = tk.Entry(window)
nodes_entry.pack()

# Button to generate adjacency matrix and heuristic input fields
generate_button = tk.Button(window, text="Generate Input Fields", command=generate_input_fields)
generate_button.pack()

# Adjacency matrix label
adjacency_matrix_label = tk.Label(window, text="Enter adjacency matrix:")
adjacency_matrix_label.pack()

# Heuristic label
heuristic_label = tk.Label(window, text="Enter heuristic values:")
heuristic_label.pack()

# Start and goal input
start_label = tk.Label(window, text="Enter start node:")
start_label.pack()
start_entry = tk.Entry(window)
start_entry.pack()

goal_label = tk.Label(window, text="Enter goal node:")
goal_label.pack()
goal_entry = tk.Entry(window)
goal_entry.pack()

# Run button to execute the selected algorithm
run_button = tk.Button(window, text="Run Algorithm", command=run_algorithm)
run_button.pack()

window.mainloop()

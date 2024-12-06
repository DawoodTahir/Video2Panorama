from fibheap import makefheap


# Define the graph
graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('A', 1), ('C', 2), ('D', 5)],
    'C': [('A', 4), ('B', 2), ('D', 1)],
    'D': [('B', 5), ('C', 1)]
}

# Define multiple sources
sources = ['A', 'B']

# Initialize distances and Fibonacci heap
distances = {node: float('inf') for node in graph}
heap = makefheap()
heap_nodes = {}

for source in sources:
    distances[source] = 0
    heap_nodes[source] = heap.insert((0, source))  # Only pass two arguments as required by the library

# Dijkstra's algorithm loop
while not heap.is_empty():
    min_node = heap.extract_min()
    current_distance, current_vertex = min_node.key  # Unpack tuple

    for neighbor, weight in graph[current_vertex]:
        distance = current_distance + weight

        if distance < distances[neighbor]:
            distances[neighbor] = distance
            if neighbor in heap_nodes:
                heap.decrease_key(heap_nodes[neighbor], (distance, neighbor))
            else:
                heap_nodes[neighbor] = heap.insert((distance, neighbor))

print("Shortest distances from sources:", distances)
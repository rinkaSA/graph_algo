import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

graph = np.array([
    [0, 20, 12, 0, 0, 0, 0, 0, 0, 0],
    [20, 0, 12, 9, 20, 0, 0, 0, 0, 0],
    [12, 12, 0, 20, 0, 0, 20, 7, 0, 0],
    [0, 9, 20, 0, 0, 0, 2, 18, 0, 0],
    [0, 20, 0, 0, 0, 10, 0, 0, 10, 18],
    [0, 0, 0, 0, 10, 0, 1, 0, 20, 18],
    [0, 0, 20, 2, 0, 1, 0, 7, 16, 0],
    [0, 0, 7, 18, 0, 0, 7, 0, 17, 0],
    [0, 0, 0, 0, 10, 20, 16, 17, 0, 8],
    [0, 0, 0, 0, 18, 18, 0, 0, 8, 0]
]
)

G = nx.from_numpy_matrix(np.array(graph), create_using=nx.Graph)
color_map = []


def show_graph_with_labels():
    labels = nx.get_edge_attributes(G, 'weight')
    pos = nx.circular_layout(G)
    for n in G.nodes():
        color_map.append('purple')
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels)
    nx.draw_circular(G, node_color=color_map, with_labels=True, edge_labels=True)
    plt.show()


def show_changed_color_graph():
    labels = nx.get_edge_attributes(G, 'weight')
    pos = nx.circular_layout(G)
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels)
    nx.draw_circular(G, node_color=color_map, with_labels=True, edge_labels=True)
    plt.show()
    return G


def color_change(c, number):
    counter = 0
    while list(G.nodes)[counter] != number:
        counter += 1
    color_map[counter] = c


def bfs_algo_1(graph, source):
    show_graph_with_labels()
    visited = []
    queue = []
    queue.append(source)
    visited.append(source)
    while queue:
        vis = queue[0]
        print(vis)
        color_change('red', vis)
        show_changed_color_graph()
        queue.pop(0)
        for i in range(graph.shape[0]):
            if graph[vis][i] != 0 and i not in visited:
                queue.append(i)
                visited.append(i)
    return visited


visited = [False] * graph.shape[1]
vis_for_print = []


def dfs_algo(graph, ver, visited):
    if ver == 0:
        show_graph_with_labels()
    vis_for_print.append(ver)
    print(ver)
    color_change('red', ver)
    show_changed_color_graph()
    visited[ver] = True
    for i in range(graph.shape[0]):
        if graph[ver][i] != 0 and visited[i] is False:
            dfs_algo(graph, i, visited)
    return vis_for_print


def min_key_Prima(graph, key, MST):
    minimum = float('Inf')  # like maxint
    for v in range(graph.shape[0]):
        if key[v] < minimum and MST[v] is False:
            minimum = key[v]
            minimum_index = v

    return minimum_index


def print_MST(graph, parent):
    for i in range(1, graph.shape[0]):
        print(parent[i], "-", i, "\t", graph[i][parent[i]])
    new_print(graph, parent)


def Prima(graph):
    key = [1000] * graph.shape[0]
    parent = [None] * graph.shape[0]
    key[0] = 0
    MST = [False] * graph.shape[0]
    parent[0] = -1
    for count in range(graph.shape[0]):
        u = min_key_Prima(graph, key, MST)
        MST[u] = True
        for v in range(graph.shape[0]):
            if 0 < graph[u][v] < key[v] and MST[v] is False:
                key[v] = graph[u][v]
                parent[v] = u
    print_MST(graph, parent)


def new_print(graph, parent):
    new_graph = np.zeros((graph.shape[0], graph.shape[1]), dtype=int)
    parent[0] = 0
    for i in range(new_graph.shape[0]):
        new_graph[parent[i], i] = int(graph[i][parent[i]])
        new_graph[i, parent[i]] = int(graph[i][parent[i]])
    # print(new_graph)
    new_g = nx.from_numpy_matrix(np.array(new_graph), create_using=nx.Graph)
    labels = nx.get_edge_attributes(new_g, 'weight')
    pos = nx.circular_layout(new_g)
    for n in new_g.nodes():
        color_map.append('magenta')
    nx.draw_networkx_edge_labels(new_g, pos=pos, edge_labels=labels)
    nx.draw_circular(new_g, node_color=color_map, with_labels=True, edge_labels=True)
    plt.show()


def print_dij_solution(graph, dist, par_edges):
    color_edge_tuples = []
    for vertex in range(graph.shape[0]):
        print(vertex, "\t", dist[vertex])
    for i in range(1, graph.shape[0]):
        color_edge_tuples.append((par_edges[i], i))
        print(par_edges[i], "-", i, "\t", graph[i][par_edges[i]])
    color_edge_map = ['green' if e in color_edge_tuples else 'white' for e in G.edges]
    nx.draw_networkx_edges(G, pos=nx.circular_layout(G),
                           edgelist=G.edges, edge_color=color_edge_map)
    new_print(graph, par_edges)


def minDistance(graph, dist, vis_vertex):
    mini = 1000
    for v in range(graph.shape[0]):
        if dist[v] < mini and vis_vertex[v] is False:
            mini = dist[v]
            min_index = v
    return min_index


def Dijkstra(graph, root):
    dist = [10000] * graph.shape[0]
    dist[root] = 0
    par_edges = [None] * graph.shape[0]
    vis_vertex = [False] * graph.shape[0]
    par_edges[0] = -1

    for count in range(graph.shape[0]):
        u = minDistance(graph, dist, vis_vertex)
        vis_vertex[u] = True
        for v in range(graph.shape[0]):
            if 0 < graph[u][v] and dist[v] > dist[u] + graph[u][v] and vis_vertex[v] is False:
                dist[v] = dist[u] + graph[u][v]
                par_edges[v] = u

    print_dij_solution(graph, dist, par_edges)


def bfs_algo_2(graph, source, sink, parent):
    visited = []
    queue = []
    queue.append(source)
    visited.append(source)
    color_change('r', source)
    while queue:
        vis = queue[0]
        queue.pop(0)
        for i in range(graph.shape[0]):
            if graph[vis][i] != 0 and i not in visited:
                queue.append(i)
                visited.append(i)
                color_change('r', i)
                #show_changed_color_graph()
                parent[i] = vis

    print(visited)
    return True if sink in visited else False


def FordFulkerson(graph, source, sink):
    show_graph_with_labels()
    parent = [-1] * graph.shape[1]
    max_flow = 0
    while bfs_algo_2(graph, source, sink, parent):
        print(parent)
        path_flow = float("Inf")
        s = sink
        while s != source:
            color_change('b', s)
            path_flow = min(path_flow, graph[parent[s]][s])
            s = parent[s]
            color_change('b', s)
            show_changed_color_graph()
        color_change('b',source)
        show_changed_color_graph()
        for i in G.nodes:
            color_change('purple',i)
        show_changed_color_graph()
        max_flow += path_flow
        v = sink
        while v != source:
            u = parent[v]
            graph[u][v] -= path_flow
            graph[v][u] -= path_flow
            v = parent[v]
        print(max_flow)
    return max_flow
#bfs_algo_1(graph, 0)
#dfs_algo(graph, 0 , visited)
#show_graph_with_labels()
#Prima(graph)
#Dijkstra(graph,0)
#print(FordFulkerson(graph, 0, 3))

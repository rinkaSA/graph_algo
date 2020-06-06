import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

graph = np.array([
    [0, 2, 5, 0, 0, 0],
    [2, 0, 4, 8, 0, 0],
    [5, 4, 0, 10, 3, 0],
    [0, 8, 10, 0, 0, 5],
    [0, 0, 3, 0, 0, 0],
    [0, 0, 0, 5, 0, 0]
]
)

G = nx.from_numpy_matrix(np.array(graph), create_using=nx.Graph)
color_map = []

def show_graph_with_labels(graph):
    G = nx.from_numpy_matrix(np.array(graph), create_using=nx.Graph)
    labels = nx.get_edge_attributes(G, 'weight')
    pos = nx.circular_layout(G)
    for n in G.nodes():
        color_map.append('purple')
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels)
    nx.draw_circular(G,node_color=color_map, with_labels=True, edge_labels=True)
    plt.show()
    return G


def show_result_bfs(graph, source):
    path = bfs_algo_1(graph, source)
    Graph = show_graph_with_labels(graph)
    colors = [i / len(path) for i in path]
    for i in path:
        labels = nx.get_edge_attributes(Graph, "weight")
        nx.draw_networkx_edge_labels(Graph, pos=nx.circular_layout(Graph), edge_labels=labels)
        nx.draw_circular(Graph, node_color=colors, with_labels=True, edge_labels=True)
    plt.show()





def color_change(c, number):
    counter = 0
    while list(G.nodes)[counter] != number:
        counter += 1
    color_map[counter] = c


def bfs_algo_1(graph, source):
    show_graph_with_labels(graph)
    visited = []
    queue = []
    queue.append(source)
    visited.append(source)
    while queue:
        vis = queue[0]
        print(vis)
        queue.pop(0)
        for i in range(graph.shape[0]):
            if graph[vis][i] != 0 and i not in visited:
                queue.append(i)
                visited.append(i)
    return visited


visited = [False] * graph.shape[1]
vis_for_print = []


def dfs_algo(graph, ver, visited):
    vis_for_print.append(ver)
    print(ver)
    visited[ver] = True
    for i in range(graph.shape[0]):
        if graph[ver][i] != 0 and visited[i] is False:
            dfs_algo(graph, i, visited)
    return vis_for_print


def show_dfs_resuls(graph, source):
    show_graph_with_labels(graph)
    path = dfs_algo(graph, source, visited)
    print(path)
    Graph = show_graph_with_labels(graph)
    colors = [i / len(path) for i in path]
    for i in path:
        nx.draw_circular(Graph, node_color=colors, with_labels=True, edge_labels=True)
    plt.show()


def min_key_Prima(graph, key, MST):
    min = 1000  # like maxint
    for v in range(graph.shape[0]):
        if key[v] < min and MST[v] is False:
            min = key[v]
            min_index = v

    return min_index


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
    # np.nditer(prima_graph, op_flags=['readwrite'])
    print(new_graph)
    show_graph_with_labels(new_graph)


def print_dij_solution(graph, dist, par_edges):
    Graph = show_graph_with_labels(graph)
    # color_edge = [(par_edges[i], i) for i in range(1, graph.shape[0]) for i in range(1, graph.shape[0])]
    color_edge_tuples = []
    color_edge_map = []
    for vertex in range(graph.shape[0]):
        print(vertex, "\t", dist[vertex])
    for i in range(1, graph.shape[0]):
        color_edge_tuples.append((par_edges[i], i))
    color_edge_map = ['magenta' if e in color_edge_tuples else 'black' for e in G.edges]
    print(par_edges[i], "-", i, "\t", graph[i][par_edges[i]])
    nx.draw_networkx_edges(Graph, pos=nx.circular_layout(Graph),
                           edgelist=G.edges, edge_color=color_edge_map)
    new_print(graph, par_edges)


def minDistance(graph, dist, vis_vertex):
    min = 1000
    for v in range(graph.shape[0]):
        if dist[v] < min and vis_vertex[v] is False:
            min = dist[v]
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
    # color_change('r', source)
    while queue:
        vis = queue[0]
        queue.pop(0)
        for i in range(graph.shape[0]):
            if graph[vis][i] != 0 and i not in visited:
                queue.append(i)
                visited.append(i)
                # color_change('r', i)
                parent[i] = vis
    print(visited)
    return True if sink in visited else False


def FordFulkerson(graph, source, sink):
    show_graph_with_labels(graph)
    parent = [-1] * graph.shape[1]
    max_flow = 0
    while bfs_algo_2(graph, source, sink, parent):
        # print(parent)
        path_flow = float("Inf")
        s = sink
        while s != source:
            path_flow = min(path_flow, graph[parent[s]][s])
            s = parent[s]
        max_flow += path_flow
        v = sink
        while v != source:
            u = parent[v]
            color_change('r', v)
            color_change('r', u)
            graph[u][v] -= path_flow
            graph[v][u] -= path_flow
            v = parent[v]
        G = nx.from_numpy_matrix(np.array(graph), create_using=nx.Graph)
        labels = nx.get_edge_attributes(G, 'weight')
        pos = nx.circular_layout(G)
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels)
        nx.draw_circular(G,node_color=color_map, with_labels=True, edge_labels=True)
        plt.show()
    return max_flow


#show_result_bfs(graph, 0)
# Dijkstra(graph,0)
# show_dfs_resuls(graph,0)
# Prima(graph)
#print(FordFulkerson(graph, 0, 3))
show_graph_with_labels(graph)
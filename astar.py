
from heapq import heappush, heappop
from itertools import count
import networkx as nx
import matplotlib.pyplot as plt

color_map = []

def color_change(c, number):
    counter = 0
    while list(G.nodes)[counter] != number:
        counter += 1
    color_map[counter] = c

def dist(vertex1, vertex2):
    x1 = pos[vertex1][0]
    x2 = pos[vertex2][0]
    y1 = pos[vertex1][1]
    y2 = pos[vertex2][1]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def astarpath(G, source, target, heuristic, weight='weight'):

    if source not in G or target not in G:
        return "Either source or target is not in graph"
    push = heappush
    pop = heappop

    c = count()
    queue = [(0, next(c), source, 0, None)]

    enqueued = {}
    explored = {}

    while queue:
        _, __, curnode, dist, parent = pop(queue) # Pop the smallest item from queue

        if curnode == target:
            path = [curnode]
            node = parent
            while node is not None:
                path.append(node)
                node = explored[node]
            path.reverse()
            return path

        if curnode in explored:
            if explored[curnode] is None:
                continue

            qcost, h = enqueued[curnode]
            if qcost < dist:
                continue

        explored[curnode] = parent

        for neighbor, w in G[curnode].items():
            ncost = dist + w.get(weight, 1)
            if neighbor in enqueued:
                qcost, h = enqueued[neighbor]
                if qcost <= ncost:
                    continue
            else:
                h = heuristic(neighbor, target)
            enqueued[neighbor] = ncost, h
            push(queue, (ncost + h, next(c), neighbor, ncost, curnode))

    return "Node is not reachable"



def astarpathlength(G, source, target, heuristic, weight='weight'):

    if source not in G or target not in G:
        return "Either source or target is not in graph"

    path = astarpath(G, source, target, heuristic, weight)
    for v in path:
         color_change('r', v)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos=pos, labels=labels)
    nx.draw(G, pos=pos, node_color=color_map, with_labels=True, edge_labels=True)
    plt.show()
    return sum(G[u][v].get(weight, 1) for u, v in zip(path[:-1], path[1:]))


G = nx.Graph()
G.add_node(0, pos=(9, 5))
G.add_node(1, pos=(7, 6))
G.add_node(2, pos=(3, 6))
G.add_node(3, pos=(1, 4))
G.add_node(4, pos=(2, 0))
G.add_node(5, pos=(6, 0))
G.add_edge(0, 1, weight=2)
G.add_edge(0, 2, weight=5)
G.add_edge(1, 2, weight=4)
G.add_edge(1, 3, weight=8)
G.add_edge(2, 3, weight=1)
G.add_edge(2, 4, weight=3)
G.add_edge(3, 5, weight=5)

for n in G.nodes():
    color_map.append('violet')
pos = nx.get_node_attributes(G,'pos')
labels = nx.get_edge_attributes(G,'weight')
nx.draw_networkx_edge_labels(G, pos=pos, labels=labels)
nx.draw(G, pos=pos, node_color=color_map, with_labels=True, edge_labels=True)
plt.show()


print(astarpathlength(G, 0, 5, heuristic=dist))
print(astarpath(G, 0, 5,heuristic=dist))




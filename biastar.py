
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

def biastar(G, source, target, weight='weight'):
    if source == target:
        return (0, [source])
    push = heappush
    pop = heappop
    dists = [{}, {}]
    paths = [{source: [source]}, {target: [target]}]
    dist_node = [[], []]
    seen = [{source: 0}, {target: 0}]
    c = count()

    push(dist_node[0], (0, next(c), source, 0, None))
    push(dist_node[1], (0, next(c), target, 0, None))
    if G.is_directed():
        neighs = [G.successors, G.predecessors]
    else:
        neighs = [G.neighbors, G.neighbors]
    finalpath = []
    direction = 1
    while dist_node[0] and dist_node[1]:
        direction = 1 - direction
        (_, __, curnode, dist, parent) = pop(dist_node[direction])
        if v in dists[direction]:
            continue
        dists[direction][v] = dist  # seen[dir][v]
        if v in dists[1 - direction]:
            return "distance: " + str(finaldist) + "\n" + "path " + str(finalpath)

        for w in neighs[direction](v):
            if (direction == 0):
                h = dist(v, w)
                minimum_weight = dist+G[v][w].get(weight, 1)
                vw_length = dists[direction][v] + minimum_weight
            else:
                h = dist(w, v)
                minimum_weight = dist+G[w][v].get(weight, 1)
                vw_length = dists[direction][v] + minimum_weight

            if w in dists[direction]:
                if vw_length < dists[direction][w]:
                    return "Contradictory paths found"
            elif w not in seen[direction] or vw_length < seen[direction][w]:
                seen[direction][w] = vw_length
                push(dist_node[direction], (vw_length+h, next(c),v,vw_length, w))
                paths[direction][w] = paths[direction][v] + [w]
                if w in seen[0] and w in seen[1]:
                    totaldist = seen[0][w] + seen[1][w]
                    if finalpath == [] or finaldist > totaldist:
                        finaldist = totaldist
                        revpath = paths[1][w][:]
                        revpath.reverse()
                        finalpath = paths[0][w] + revpath[1:]
                        for v in finalpath:
                            color_change('r', v)
                        labels = nx.get_edge_attributes(G, "w")
                        nx.draw_networkx_edge_labels(G, pos=nx.circular_layout(G), labels=labels)
                        nx.draw(G, pos=nx.circular_layout(G), node_color=color_map, with_labels=True, edge_labels=True)
                        plt.show()
    return "NO PATH"


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


print(biastar(G,0,5))


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

def bidir_dijkstra(G, source, target, weight='weight'):

    if source == target:
        return (0, [source])
    push = heappush
    pop = heappop
    dists = [{}, {}]
    paths = [{source: [source]}, {target: [target]}]
    dist_node = [[], []]
    seen = [{source: 0}, {target: 0}]
    c = count()

    push(dist_node[0], (0, next(c), source))
    push(dist_node[1], (0, next(c), target))
    if G.is_directed():
        neighs = [G.successors, G.predecessors]
    else:
        neighs = [G.neighbors, G.neighbors]
    finalpath = []
    direction = 1
    while dist_node[0] and dist_node[1]:
        direction = 1 - direction
        (dist, _, v) = pop(dist_node[direction])
        if v in dists[direction]:
            continue
        dists[direction][v] = dist  # visited[dir][v]
        if v in dists[1 - direction]:
            return "distance: " + str(finaldist) + "\n" + "path " + str(finalpath)

        for w in neighs[direction](v):
            if (direction == 0):
                minimum_weight = G[v][w].get(weight, 1)
                vw_length = dists[direction][v] + minimum_weight
            else:
                minimum_weight = G[w][v].get(weight, 1)
                vw_length = dists[direction][v] + minimum_weight

            if w in dists[direction]:
                if vw_length < dists[direction][w]:
                    return "Contradictory paths found"
            elif w not in seen[direction] or vw_length < seen[direction][w]:
                seen[direction][w] = vw_length
                push(dist_node[direction], (vw_length, next(c), w))
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


G = nx.DiGraph()
G.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
G.add_weighted_edges_from([(0, 1, 10), (0, 2, 50), (1, 3, 20), (1, 4, 70),
                           (4, 8, 30), (4, 7, 10), (3, 7, 80), (3, 5, 40),
                           (8, 9, 40), (9, 7, 30), (2, 6, 180), (6, 5, 10)])
for n in G.nodes():
    color_map.append('violet')
labels = nx.get_edge_attributes(G, "w")
nx.draw_networkx_edge_labels(G, pos=nx.circular_layout(G), labels=labels)
nx.draw_circular(G, node_color=color_map, with_labels=True)
plt.show()

print(bidir_dijkstra(G, 0, 9))

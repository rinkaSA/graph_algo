
from heapq import heappush, heappop
from itertools import count
import networkx as nx
import matplotlib.pyplot as plt

color_map = []
G = nx.DiGraph()
G.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
G.add_weighted_edges_from([(0, 1, 10), (0, 2, 50), (1, 3, 20), (1, 4, 70),
                               (4, 8, 30), (4, 7, 10), (3, 7, 80), (3, 5, 40),
                               (8, 9, 40), (9, 7, 30), (2, 6, 180), (6, 5, 10)])
def show_graph():
    labels = {}
    for u, v, data in G.edges(data=True):
        labels[(u, v)] = data['weight']
    for n in G.nodes():
        color_map.append('violet')
    labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos=nx.circular_layout(G), edge_labels=labels)
    nx.draw_circular(G, node_color=color_map, with_labels=True, edge_labels=labels)
    plt.show()

def color_change(c, number):
    counter = 0
    while list(G.nodes)[counter] != number:
        counter += 1
    color_map[counter] = c



def show_changed_color_graph():
    labels = nx.get_edge_attributes(G, 'weight')
    pos = nx.circular_layout(G)
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels)
    nx.draw_circular(G, node_color=color_map, with_labels=True, edge_labels=labels)
    plt.show()
    return G

def bi_dij(G, source, aim, weight='weight'):
    show_graph()
    if source == aim:
        return (0, [source])
    flag = False
    push = heappush
    pop = heappop
    dists = [{}, {}]
    paths = [{source: [source]}, {aim: [aim]}]
    dist_node = [[], []]
    seen = [{source: 0}, {aim: 0}]
    c = count()
    color_change('purple',source)
    color_change('purple', aim)
    show_changed_color_graph()
    push(dist_node[0], (0, next(c), source))
    push(dist_node[1], (0, next(c), aim))
    if G.is_directed():
        neighs = [G.successors, G.predecessors]
    else:
        neighs = [G.neighbors, G.neighbors]
    finalpath = []
    direction = 1
    while dist_node[0] and dist_node[1]:
        direction = 1 - direction
        (dist, _, v) = pop(dist_node[direction])
        if flag is False:
            color_change('purple',v)
            show_changed_color_graph()
        if v in dists[direction]:
            continue
        dists[direction][v] = dist  # visited[dir][v]
        if v in dists[1 - direction]:
            print(dists)
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
            elif w not in seen[direction] or vw_length < seen[direction][w]:#важно тут выбираем нужную инцид вершину
                seen[direction][w] = vw_length
                push(dist_node[direction], (vw_length, next(c), w))
                paths[direction][w] = paths[direction][v] + [w]
                if color_map[w]!='purple':
                    color_change('r',w)
                show_changed_color_graph()
                if w in seen[0] and w in seen[1]:
                    color_change('purple', w)
                    show_changed_color_graph()
                    totaldist = seen[0][w] + seen[1][w]
                    if finalpath == [] or finaldist > totaldist:
                        finaldist = totaldist
                        revpath = paths[1][w][:]
                        revpath.reverse()
                        finalpath = paths[0][w] + revpath[1:]
                        flag = True

    return "NO PATH"



print(bi_dij(G, 0, 9))

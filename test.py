import networkx as nx


g = nx.Graph()
g.add_node(1, kind=1, f=3, l=4)
g.add_node(2, kind=1, f=3, l=4)
g.add_node(3, kind=1, f=3, l=4)
g.add_edge(1, 2, weight=4)
g.add_edge(1, 3)
print(g.nodes(data=True), list(g.nodes(data=True))[1])
print(g.edges(data=True))
nx.set_edge_attributes(g, {(1, 2): {'weight': 45}, (1, 3): {'weight': 5}, (2, 3): {'weight': 5}})
print(g.edges(data=True))
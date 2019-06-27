import networkx as nx

G = nx.DiGraph()
G.add_weighted_edges_from(
	[(1, 2, 0.5), (2, 1, 0.75),
	(2, 3, -0.2), (3, 2, 0.1),
	(1, 3, -0.4), (3, 1, -0.1)])
print(G[1][2])

def f(node):
	inc = node.
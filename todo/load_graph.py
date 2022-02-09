import numpy as np
import networkx as nx
from itertools import repeat
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

##### Layer 2:
# topological sort [67, 62, 17, 56, 44, 84, 93]
# sergio           [67, 17, 56, 44]
# diff             [62, 84, 93]
##### Layer 1:
# topo [74, 1, 14]
# sergio [1, 74, 14, 84, 93, 62]
# diff [84, 93, 62]

def load_graph():
  graph = nx.DiGraph()
  with open('SERGIO/data_sets/De-noised_100G_9T_300cPerT_4_DS1/Interaction_cID_4.txt', 'r') as f:
    for line in f.readlines():
      fields = np.array(line.split(','), dtype=float)
      target, num_regulators = int(fields[0]), int(fields[1])
      regulators = map(int, fields[2:2+num_regulators])
      contributions = fields[2+num_regulators : 2+2*num_regulators]
      graph.add_weighted_edges_from(zip(regulators, repeat(target), contributions))
  return graph


def to_levels(graph):
  levels = nx.topological_generations(graph)
  return levels


graph = load_graph()
levels = list(to_levels(graph))
print(levels)
### 2 ways to show graph
p=nx.drawing.nx_pydot.to_pydot(graph)
p.write_png('example.png')

nx.draw(graph, cmap = plt.get_cmap('jet'), pos=graphviz_layout(graph, prog='dot'), with_labels=True)
plt.savefig('graph.png')
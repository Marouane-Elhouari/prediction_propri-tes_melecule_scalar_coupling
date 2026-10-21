import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

BOND_CUTOFF = 1.65
H_CUTOFF = 1.2
J = 'scalar_coupling_constant'
SYM = {0: 'H', 1: 'C', 2: 'O', 3: 'N', 4: 'F'}
COL = {0: 'white', 1: 'lightgray', 2: 'red', 3: 'dodgerblue', 4: 'limegreen'}
SIZE = {0: 350, 1: 800, 2: 750, 3: 780, 4: 600}


def molecule_graph(mol):
    k = np_dict[mol]
    a, b = ssx[k], ssx[k + 1]
    xyz = np_xyz[a:b].astype('float64')
    at = np_atoms[a:b]
    d = np.linalg.norm(xyz[None, :, :] - xyz[:, None, :], axis=2)
    is_H = at == 0
    cutoff = np.where(is_H[:, None] | is_H[None, :], H_CUTOFF, BOND_CUTOFF)
    B = (d < cutoff) & (d > 0)
    G = nx.Graph()
    for n in range(len(at)):
        G.add_node(n, atom=int(at[n]))
    for u, v in zip(*np.where(np.triu(B))):
        G.add_edge(int(u), int(v), length=float(d[u, v]))
    return G, xyz


def layout_2d(G, xyz, how='kk'):
    if how == 'pca':
        c = xyz - xyz.mean(0)
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        p = c @ vt[:2].T
        return {n: p[n] for n in G.nodes}
    return nx.kamada_kawai_layout(G)


def draw_molecule_graph(row_id, ax=None, how='kk', edge_labels=False):
    row = train_df.loc[row_id]
    mol = row['molecule_name']
    i, j = int(row['atom_index_0']), int(row['atom_index_1'])
    G, xyz = molecule_graph(mol)
    pos = layout_2d(G, xyz, how)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    atoms = nx.get_node_attributes(G, 'atom')
    nodes = list(G.nodes)
    nx.draw_networkx_edges(G, pos, ax=ax, width=2, edge_color='gray')
    if G.has_edge(i, j):
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(i, j)], width=6, edge_color='#e24b4a')
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nodes,
                           node_color=[COL[atoms[n]] for n in nodes],
                           node_size=[SIZE[atoms[n]] for n in nodes],
                           edgecolors=['#7a5cd6' if n in (i, j) else 'black' for n in nodes],
                           linewidths=[4 if n in (i, j) else 1 for n in nodes])
    nx.draw_networkx_labels(G, pos, ax=ax, labels={n: f'{SYM[atoms[n]]}{n}' for n in nodes}, font_size=8)
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, ax=ax, font_size=7,
                                     edge_labels={e: f"{G.edges[e]['length']:.2f}" for e in G.edges})
    title = f"{row['type']} | {mol}\nJ = {row[J]:.1f} Hz | {G.number_of_nodes()} atoms, {G.number_of_edges()} bonds"
    if 'a1_n_bonds' in train_df.columns:
        title += f" | atom {j} has {int(row['a1_n_bonds'])} bonds"
    ax.set_title(title, fontsize=10)
    ax.axis('off')


sub = train_df[train_df['type'] == '1JHC']
if 'a1_n_bonds' in sub.columns:
    ids = [sub[sub['a1_n_bonds'] == nb].sample(1, random_state=50).index[0] for nb in (2, 3, 4)]
else:
    ids = list(sub.sample(3, random_state=50).index)

fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
for ax, rid in zip(axes, ids):
    draw_molecule_graph(rid, ax=ax)
plt.tight_layout()
plt.show()
import warnings
warnings.filterwarnings('ignore', message='.*eval_set.*')
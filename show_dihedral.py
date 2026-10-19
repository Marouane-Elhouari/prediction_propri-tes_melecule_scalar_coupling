import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc

BOND_CUTOFF = 1.65
H_CUTOFF = 1.2
J = 'scalar_coupling_constant'
SYM = {0: 'H', 1: 'C', 2: 'O', 3: 'N', 4: 'F'}
COL = {0: 'white', 1: 'lightgray', 2: 'red', 3: 'dodgerblue', 4: 'limegreen'}
SIZE = {0: 70, 1: 220, 2: 200, 3: 210, 4: 160}
C_I = '#7a5cd6'
C_J = '#1d9e75'
C_PATH = '#e24b4a'


def mol_data(mol):
    k = np_dict[mol]
    a, b = ssx[k], ssx[k + 1]
    xyz = np_xyz[a:b].astype('float64')
    at = np_atoms[a:b]
    d = np.linalg.norm(xyz[None, :, :] - xyz[:, None, :], axis=2)
    is_H = at == 0
    cutoff = np.where(is_H[:, None] | is_H[None, :], H_CUTOFF, BOND_CUTOFF)
    B = (d < cutoff) & (d > 0)
    return xyz, at, B


def path_3J(B, i, j):
    T = B[i][:, None] & B & B[j][None, :]
    T[j, :] = False
    T[:, i] = False
    idx = np.argwhere(T)
    return (int(idx[0][0]), int(idx[0][1])) if len(idx) else None


def dihedral_deg(p0, p1, p2, p3):
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    return np.degrees(np.arctan2(np.dot(np.cross(b1n, v), w), np.dot(v, w)))


def pick_row(type_, near, n_try):
    cand = train_df[train_df['type'] == type_].sample(n_try, random_state=50)
    best = None
    for rid, r in cand.iterrows():
        i, j = int(r['atom_index_0']), int(r['atom_index_1'])
        xyz, at, B = mol_data(r['molecule_name'])
        p = path_3J(B, i, j)
        if p is None:
            continue
        phi = abs(dihedral_deg(xyz[i], xyz[p[0]], xyz[p[1]], xyz[j]))
        score = 0 if near is None else abs(phi - near)
        if best is None or score < best[0]:
            best = (score, rid)
        if near is None:
            break
    return best[1]


def show_dihedral(row_id=None, type_='3JHH', near=None, n_try=300):
    if row_id is None:
        row_id = pick_row(type_, near, n_try)
    row = train_df.loc[row_id]
    mol = row['molecule_name']
    i, j = int(row['atom_index_0']), int(row['atom_index_1'])
    xyz, at, B = mol_data(mol)
    p = path_3J(B, i, j)
    if p is None:
        print('no 3J path found for this row')
        return
    m, n = p
    phi = dihedral_deg(xyz[i], xyz[m], xyz[n], xyz[j])

    u = xyz[n] - xyz[m]
    u /= np.linalg.norm(u)
    h = xyz[i] - xyz[m]
    e_up = h - np.dot(h, u) * u
    e_up /= np.linalg.norm(e_up)
    e_right = np.cross(u, e_up)

    def proj(k):
        r = xyz[k] - xyz[m]
        return np.array([np.dot(r, e_right), np.dot(r, e_up)])

    fig = plt.figure(figsize=(14, 6.5))
    fig.suptitle(f"{row['type']} | {mol} | path {SYM[at[i]]}{i}-{SYM[at[m]]}{m}-{SYM[at[n]]}{n}-{SYM[at[j]]}{j}"
                 f" | dihedral = {abs(phi):.1f}° | J = {row[J]:.2f} Hz", fontsize=13)

    ax3 = fig.add_subplot(1, 2, 1, projection='3d')
    for a_ in range(len(xyz)):
        for b_ in range(a_ + 1, len(xyz)):
            if B[a_, b_]:
                ax3.plot(*zip(xyz[a_], xyz[b_]), color='gray', lw=1.5)
    for a_, b_ in [(i, m), (m, n), (n, j)]:
        ax3.plot(*zip(xyz[a_], xyz[b_]), color=C_PATH, lw=5)
    for k in range(len(xyz)):
        face = COL[at[k]]
        edge = 'black'
        if k == i:
            face, edge = C_I, 'black'
        if k == j:
            face, edge = C_J, 'black'
        ax3.scatter(*xyz[k], s=SIZE[at[k]] * (1.8 if k in (i, j, m, n) else 1), c=face, edgecolors=edge, depthshade=False)
    for k in (i, m, n, j):
        ax3.text(*(xyz[k] + 0.08), f'{SYM[at[k]]}{k}', fontsize=11, fontweight='bold')
    mid = xyz.mean(0)
    R = np.ptp(xyz, axis=0).max() / 2 + 0.6
    ax3.set_xlim(mid[0] - R, mid[0] + R)
    ax3.set_ylim(mid[1] - R, mid[1] + R)
    ax3.set_zlim(mid[2] - R, mid[2] + R)
    ax3.set_box_aspect((1, 1, 1))
    ax3.view_init(elev=15, azim=40)
    ax3.set_axis_off()
    ax3.set_title('3D molecule (path in red)')

    ax = fig.add_subplot(1, 2, 2)
    ax.add_patch(Circle((0, 0), 0.9, fill=False, lw=2.5, color='black'))
    L = 2.1
    for k in np.where(B[m])[0]:
        if k == n:
            continue
        x, y = proj(k)
        a = np.arctan2(y, x)
        ax.plot([0, L * np.cos(a)], [0, L * np.sin(a)], color='black', lw=2.5, zorder=1)
        face = C_I if k == i else COL[at[k]]
        ax.add_patch(Circle((L * np.cos(a), L * np.sin(a)), 0.3, fc=face, ec='black', zorder=3))
        ax.text(L * np.cos(a), L * np.sin(a), f'{SYM[at[k]]}{k}', ha='center', va='center', fontsize=9, zorder=4)
    for k in np.where(B[n])[0]:
        if k == m:
            continue
        x, y = proj(k)
        a = np.arctan2(y, x)
        ax.plot([0.9 * np.cos(a), L * np.cos(a)], [0.9 * np.sin(a), L * np.sin(a)], color='black', lw=2.5, zorder=1)
        face = C_J if k == j else COL[at[k]]
        ax.add_patch(Circle((L * np.cos(a), L * np.sin(a)), 0.3, fc=face, ec='black', zorder=3))
        ax.text(L * np.cos(a), L * np.sin(a), f'{SYM[at[k]]}{k}', ha='center', va='center', fontsize=9, zorder=4)

    a_i = 90.0
    xj, yj = proj(j)
    a_j = np.degrees(np.arctan2(yj, xj))
    diff = (a_j - a_i + 180) % 360 - 180
    t1, t2 = (a_i, a_i + diff) if diff >= 0 else (a_i + diff, a_i)
    ax.add_patch(Arc((0, 0), 2.7, 2.7, theta1=t1, theta2=t2, color=C_PATH, lw=3, zorder=2))
    ax.plot([0, 0.9 * np.cos(np.radians(a_j))], [0, 0.9 * np.sin(np.radians(a_j))], '--', color=C_J, lw=2, zorder=2)
    mid_a = np.radians(a_i + diff / 2)
    ax.text(1.75 * np.cos(mid_a), 1.75 * np.sin(mid_a), f'φ = {abs(diff):.1f}°', color=C_PATH, fontsize=14,
            fontweight='bold', ha='center', va='center', zorder=5,
            bbox=dict(fc='white', ec='none', alpha=0.8, pad=1.5))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Newman projection (look along {SYM[at[m]]}{m} -> {SYM[at[n]]}{n}): '
                 f'center = {SYM[at[m]]}{m} (front), circle = {SYM[at[n]]}{n} (back)', fontsize=10)
    if abs(abs(diff) - abs(phi)) > 1.0:
        print('warning: screen angle and computed dihedral differ', abs(diff), abs(phi))
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()


for target in [0, 60, 180]:
    show_dihedral(type_='3JHH', near=target)
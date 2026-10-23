import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit

MAX_ROWS = 150_000
J = 'scalar_coupling_constant'
FEATS = ['n_bonds', 'n_H', 'n_C', 'n_N', 'n_O', 'n_F',
         'min_heavy_bl', 'mean_bl', 'mean_angle']
BOND_CUTOFF = 1.65
H_CUTOFF = 1.2


GEO_COLS = ['ang1', 'ang2', 'dih', 'cos_dih', 'cos2_dih', 'd1', 'd2', 'd3', 'pm_atom', 'pn_atom']
PATH_COLS = GEO_COLS + ['pm_' + f for f in FEATS] + ['pn_' + f for f in FEATS]


def angle(p, q, r):
    a = p - q
    b = r - q
    c = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def dihedral(p0, p1, p2, p3):
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / np.linalg.norm(b1, axis=1, keepdims=True)
    v = b0 - (b0 * b1n).sum(1, keepdims=True) * b1n
    w = b2 - (b2 * b1n).sum(1, keepdims=True) * b1n
    x = (v * w).sum(1)
    y = (np.cross(b1n, v) * w).sum(1)
    return np.degrees(np.arctan2(y, x))


def path_features(df):
    n_rows = len(df)
    cols = {c: np.full(n_rows, np.nan, dtype='float32') for c in PATH_COLS}
    groups = df.groupby('molecule_name').indices
    I_all = df['atom_index_0'].values
    J_all = df['atom_index_1'].values
    t_all = df['type'].str[0].values

    for mol, pos in groups.items():
        k = np_dict[mol]
        a, b = ssx[k], ssx[k + 1]
        n = b - a
        xyz = np_xyz[a:b].astype('float64')
        at = np_atoms[a:b]
        d = np.linalg.norm(xyz[None, :, :] - xyz[:, None, :], axis=2)
        is_H = at == 0
        cutoff = np.where(is_H[:, None] | is_H[None, :], H_CUTOFF, BOND_CUTOFF)
        B = (d < cutoff) & (d > 0)
        I = I_all[pos]
        Jx = J_all[pos]
        t = t_all[pos]

        s2 = np.where(t == '2')[0]
        if len(s2):
            C = B[I[s2]] & B[Jx[s2]]
            ok = C.any(1)
            if ok.any():
                s = s2[ok]
                i = I[s]
                j = Jx[s]
                m = C[ok].argmax(1)
                g = pos[s]
                cols['ang1'][g] = angle(xyz[i], xyz[m], xyz[j])
                cols['d1'][g] = d[i, m]
                cols['d2'][g] = d[m, j]
                cols['pm_atom'][g] = at[m]
                for f in FEATS:
                    cols['pm_' + f][g] = F[f][a + m]

        s3 = np.where(t == '3')[0]
        if len(s3):
            r = len(s3)
            T = B[I[s3]][:, :, None] & B[None, :, :] & B[Jx[s3]][:, None, :]
            ar = np.arange(r)
            T[ar, Jx[s3], :] = False
            T[ar, :, I[s3]] = False
            flat = T.reshape(r, -1)
            ok = flat.any(1)
            if ok.any():
                first = flat[ok].argmax(1)
                m = first // n
                q = first % n
                s = s3[ok]
                i = I[s]
                j = Jx[s]
                g = pos[s]
                dih = dihedral(xyz[i], xyz[m], xyz[q], xyz[j])
                cols['dih'][g] = dih
                cols['cos_dih'][g] = np.cos(np.radians(dih))
                cols['cos2_dih'][g] = np.cos(2 * np.radians(dih))
                cols['ang1'][g] = angle(xyz[i], xyz[m], xyz[q])
                cols['ang2'][g] = angle(xyz[m], xyz[q], xyz[j])
                cols['d1'][g] = d[i, m]
                cols['d2'][g] = d[m, q]
                cols['d3'][g] = d[q, j]
                cols['pm_atom'][g] = at[m]
                cols['pn_atom'][g] = at[q]
                for f in FEATS:
                    cols['pm_' + f][g] = F[f][a + m]
                    cols['pn_' + f][g] = F[f][a + q]

    return pd.DataFrame(cols, index=df.index)


if 'cos_dih' not in train_df.columns:
    train_df = pd.concat([train_df, path_features(train_df)], axis=1)
    test_df = pd.concat([test_df, path_features(test_df)], axis=1)

TYPES_E = sorted(t for t in train_df['type'].unique() if t[0] in '23')
print('coverage of path features (share of rows with a path found):')
for T in TYPES_E:
    print(f"  {T}: {train_df.loc[train_df['type'] == T, 'd1'].notna().mean():.3f}")

a_cols = ['a0_' + f for f in FEATS] + ['a1_' + f for f in FEATS]
experiments = {'dist + a0 + a1': ['dist'] + a_cols,
               '+ path features': ['dist'] + a_cols + PATH_COLS}

try:
    import lightgbm as lgb

    def make_model():
        return lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.1, num_leaves=64,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
                                 random_state=50, n_jobs=-1, verbose=-1)

    def fit(model, Xtr, ytr, Xva, yva):
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric='l1',
                  callbacks=[lgb.early_stopping(100, verbose=False)])
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor

    def make_model():
        return HistGradientBoostingRegressor(max_iter=500, learning_rate=0.1, random_state=50)

    def fit(model, Xtr, ytr, Xva, yva):
        model.fit(Xtr, ytr)


def run(T):
    sub = train_df[train_df['type'] == T]
    if MAX_ROWS and len(sub) > MAX_ROWS:
        sub = sub.sample(MAX_ROWS, random_state=50)
    sub = sub.reset_index(drop=True)
    y = sub[J].values
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=50)
                  .split(sub, groups=sub['molecule_name']))
    out = {}
    for name, cols in experiments.items():
        X = sub[cols]
        model = make_model()
        fit(model, X.iloc[tr], y[tr], X.iloc[va], y[va])
        mae = np.mean(np.abs(y[va] - model.predict(X.iloc[va])))
        out[name] = dict(mae=mae, log_mae=np.log(mae))
        print(f'{T} | {name:16s} | MAE = {mae:7.3f} Hz | log-MAE = {np.log(mae):6.3f}')
    return out


results_E = {T: run(T) for T in TYPES_E}

summary_E = pd.DataFrame([{
    'type': T,
    'MAE a0+a1': o['dist + a0 + a1']['mae'],
    'MAE + path': o['+ path features']['mae'],
    'logMAE a0+a1': o['dist + a0 + a1']['log_mae'],
    'logMAE + path': o['+ path features']['log_mae']} for T, o in results_E.items()]).set_index('type')
summary_E['gain logMAE'] = summary_E['logMAE a0+a1'] - summary_E['logMAE + path']

fig, axes = plt.subplots(1, 2, figsize=(17, 5.5))
x = np.arange(len(summary_E))
w = 0.38
for ax, (c1, c2, ttl, yl) in zip(axes, [
        ('MAE a0+a1', 'MAE + path', 'MAE per type (2J and 3J)', 'MAE (Hz)'),
        ('logMAE a0+a1', 'logMAE + path', 'log-MAE per type (lower = better)', 'log-MAE')]):
    b1 = ax.bar(x - w / 2, summary_E[c1], w, label='dist + a0 + a1', color='#9aa0a6')
    b2 = ax.bar(x + w / 2, summary_E[c2], w, label='+ path features', color='#7a5cd6')
    for bars in (b1, b2):
        for b_ in bars:
            h = b_.get_height()
            ax.text(b_.get_x() + b_.get_width() / 2, h, f'{h:.2f}',
                    ha='center', va='bottom' if h >= 0 else 'top', fontsize=8)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(summary_E.index)
    ax.set_ylabel(yl)
    ax.set_title(ttl)
    ax.legend()
plt.tight_layout()
plt.show()

karplus = [T for T in ['3JHH', '3JHC'] if T in TYPES_E]
fig, axes = plt.subplots(1, len(karplus), figsize=(8 * len(karplus), 5), squeeze=False)
for ax, T in zip(axes[0], karplus):
    d3 = train_df[(train_df['type'] == T) & train_df['dih'].notna()]
    d3 = d3.sample(min(20000, len(d3)), random_state=50)
    ax.scatter(d3['dih'], d3[J], s=2, alpha=0.3)
    ax.set_xlabel('dihedral angle (deg)')
    ax.set_ylabel('J (Hz)')
    ax.set_title(f'{T}: J vs dihedral angle')
plt.tight_layout()
plt.show()

print('\n', summary_E.round(3).to_string())
if 'summary' in globals():
    full = summary['logMAE + features'].copy()
    for T in summary_E.index:
        full[T] = summary_E.loc[T, 'logMAE + path']
    print(f"\nMean log-MAE over all types (Step D for 1J, Step E for 2J/3J): {full.mean():.3f}")
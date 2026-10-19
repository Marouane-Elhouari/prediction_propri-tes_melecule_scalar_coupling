import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit

MAX_ROWS = 150_000
J        = 'scalar_coupling_constant'
FEATS    = ['n_bonds', 'n_H', 'n_C', 'n_N', 'n_O', 'n_F',
            'min_heavy_bl', 'mean_bl', 'mean_angle']

if 'a0_n_bonds' not in train_df.columns:
    train_df = add_atom_feats(train_df, col='atom_index_0', prefix='a0_')
    test_df  = add_atom_feats(test_df,  col='atom_index_0', prefix='a0_')

TYPES  = sorted(train_df['type'].unique())
a0_cols = ['a0_' + f for f in FEATS]
a1_cols = ['a1_' + f for f in FEATS]
experiments = {'dist only': ['dist'],
               'dist + a0 + a1': ['dist'] + a0_cols + a1_cols}

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

results = {T: run(T) for T in TYPES}

rows = []
for T, out in results.items():
    rows.append({'type': T,
                 'MAE dist only':  out['dist only']['mae'],
                 'MAE + features': out['dist + a0 + a1']['mae'],
                 'logMAE dist only':  out['dist only']['log_mae'],
                 'logMAE + features': out['dist + a0 + a1']['log_mae']})
summary = pd.DataFrame(rows).set_index('type')
summary['gain logMAE'] = summary['logMAE dist only'] - summary['logMAE + features']

fig, axes = plt.subplots(1, 2, figsize=(17, 5.5))
x = np.arange(len(summary)); w = 0.38
for ax, (c1, c2, ttl, yl) in zip(axes, [
        ('MAE dist only', 'MAE + features', 'MAE per type', 'MAE (Hz)'),
        ('logMAE dist only', 'logMAE + features', 'log-MAE per type (Kaggle metric, lower = better)', 'log-MAE')]):
    b1 = ax.bar(x - w / 2, summary[c1], w, label='dist only', color='#9aa0a6')
    b2 = ax.bar(x + w / 2, summary[c2], w, label='dist + a0 + a1 features', color='#7a5cd6')
    for bars in (b1, b2):
        for b_ in bars:
            h = b_.get_height()
            ax.text(b_.get_x() + b_.get_width() / 2, h, f'{h:.2f}',
                    ha='center', va='bottom' if h >= 0 else 'top', fontsize=8)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(summary.index)
    ax.set_ylabel(yl); ax.set_title(ttl); ax.legend()
plt.tight_layout()
plt.show()

print('\n', summary.round(3).to_string())
print(f"\nMean log-MAE : dist only = {summary['logMAE dist only'].mean():.3f} | "
      f"+ features = {summary['logMAE + features'].mean():.3f}")
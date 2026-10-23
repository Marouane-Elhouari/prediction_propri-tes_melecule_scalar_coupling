import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

EPOCHS = 30
LR = 1e-3
BASELINE_MAE = {'1JHC': 1.93}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

n_types = len(TYPES)
model = CouplingGNN(n_pair_types=n_types).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
mean_t = torch.tensor(MEAN, dtype=torch.float32, device=device)
std_t = torch.tensor(STD, dtype=torch.float32, device=device)
print('parameters:', sum(p.numel() for p in model.parameters()))


def train_one_epoch():
    model.train()
    total, count = 0.0, 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = F.l1_loss(pred, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * batch.y.numel()
        count += batch.y.numel()
    return total / count


@torch.no_grad()
def evaluate(loader):
    model.eval()
    err_sum = torch.zeros(n_types, device=device)
    cnt = torch.zeros(n_types, device=device)
    for batch in loader:
        batch = batch.to(device)
        t = batch.pair_type
        pred_hz = model(batch) * std_t[t] + mean_t[t]
        err = (pred_hz - batch.y_hz).abs()
        err_sum.index_add_(0, t, err)
        cnt.index_add_(0, t, torch.ones_like(err))
    present = cnt > 0
    mae = (err_sum[present] / cnt[present]).cpu().numpy()
    ids = torch.nonzero(present).flatten().cpu().numpy()
    return {TYPES[k]: float(m) for k, m in zip(ids, mae)}


@torch.no_grad()
def predict(loader):
    model.eval()
    P, Y = [], []
    for batch in loader:
        batch = batch.to(device)
        t = batch.pair_type
        P.append((model(batch) * std_t[t] + mean_t[t]).cpu())
        Y.append(batch.y_hz.cpu())
    return torch.cat(P).numpy(), torch.cat(Y).numpy()


history = {'train_loss': [], 'val_mae': [], 'val_logmae': []}
best = float('inf')

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    tr_loss = train_one_epoch()
    val = evaluate(val_loader)
    val_mae = float(np.mean(list(val.values())))
    val_log = float(np.mean([np.log(v) for v in val.values()]))
    scheduler.step(val_log)
    if val_log < best:
        best = val_log
        torch.save(model.state_dict(), 'best_gnn.pt')
    history['train_loss'].append(tr_loss)
    history['val_mae'].append(val_mae)
    history['val_logmae'].append(val_log)
    print(f'epoch {epoch:02d} | train L1 (normalised) {tr_loss:.4f} | val MAE {val_mae:.3f} Hz | '
          f'val log-MAE {val_log:.3f} | lr {optimizer.param_groups[0]["lr"]:.1e} | {time.time() - t0:.0f}s')

model.load_state_dict(torch.load('best_gnn.pt', map_location=device))
final = evaluate(val_loader)
print('\nbest model, validation MAE per type (Hz):')
for T, v in final.items():
    base = BASELINE_MAE.get(T)
    extra = f' | LightGBM baseline {base:.3f}' if base else ''
    print(f'  {T}: GNN {v:.3f} (log-MAE {np.log(v):.3f}){extra}')

fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
ep = np.arange(1, EPOCHS + 1)
axes[0].plot(ep, history['train_loss'], color='#7a5cd6', lw=2)
axes[0].set_xlabel('epoch')
axes[0].set_ylabel('train L1 (normalised)')
axes[0].set_title('Training loss')

axes[1].plot(ep, history['val_mae'], color='#7a5cd6', lw=2, label='GNN (validation)')
base_vals = [BASELINE_MAE[T] for T in final if T in BASELINE_MAE]
if base_vals:
    axes[1].axhline(np.mean(base_vals), color='gray', ls='--', label='LightGBM baseline')
axes[1].set_xlabel('epoch')
axes[1].set_ylabel('MAE (Hz)')
axes[1].set_title('Validation MAE')
axes[1].legend()

pred, true = predict(val_loader)
samp = np.random.RandomState(50).choice(len(true), min(15000, len(true)), replace=False)
axes[2].scatter(true[samp], pred[samp], s=2, alpha=0.3)
lo, hi = true.min(), true.max()
axes[2].plot([lo, hi], [lo, hi], color='red', lw=1)
axes[2].set_xlabel('actual J (Hz)')
axes[2].set_ylabel('predicted J (Hz)')
axes[2].set_title(f'GNN predicted vs actual | MAE = {np.mean(np.abs(pred - true)):.3f}')
plt.tight_layout()
plt.show()


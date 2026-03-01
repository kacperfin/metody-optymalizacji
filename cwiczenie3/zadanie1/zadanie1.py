from pathlib import Path
import cvxpy as cp
import scipy.io as sio
import matplotlib.pyplot as plt

script_dir = Path(__file__).parent
data = sio.loadmat(script_dir.parent / 'Data01.mat')
t, y = data['t'].flatten(), data['y'].flatten()
v = cp.Variable(len(y))

plots_dir = script_dir / 'wykresy'
plots_dir.mkdir(exist_ok=True)

# (3)

qs = [1, 5, 10, 20]
fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
fig.suptitle('Dopasowanie kawałkami stałego - (3)')

for i, q in enumerate(qs):
    ax = axes[i // 2, i % 2]
    prob = cp.Problem(cp.Minimize(cp.sum_squares(y - v)), [cp.norm(cp.diff(v), 1) <= q])
    prob.solve()
    ax.plot(t, y, '.', color='black', label='pomiar', alpha=0.15)
    ax.plot(t, v.value, label=f'q={q}', color='red', linewidth=2)
    ax.set_title(f'Parametr q = {q}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(plots_dir / 'zadanie1-q')

# (4) LASSO
taus = [1, 5, 10, 20]
fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
fig.suptitle('Dopasowanie kawałkami stałego - (4) LASSO')

for i, tau in enumerate(taus):
    ax = axes[i // 2, i % 2]
    prob = cp.Problem(cp.Minimize(cp.sum_squares(y - v) + tau * cp.norm(cp.diff(v), 1)))
    prob.solve()
    ax.plot(t, y, '.', color='black', label='pomiar', alpha=0.15)
    ax.plot(t, v.value, label=f'tau={tau}', color='blue', linewidth=2)
    ax.set_title(f'Parametr tau = {tau}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(plots_dir / 'zadanie1-lasso')

import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

txt = open("act_object50k_train.log", errors="ignore").read()
S, L, L1 = [], [], []
for s, loss, l1 in re.findall(r"step:(\S+)\s.*?\bloss:([\d.]+).*?\bl1_loss:([\d.]+)", txt):
    s = float(s[:-1]) * 1000 if s.endswith("K") else float(s)
    S.append(s); L.append(float(loss)); L1.append(float(l1))
S, L, L1 = map(np.array, (S, L, L1))
if len(S) == 5000: S = np.arange(1, 5001) * 10.0
else: print('WARNING: expected 5000 log points, got', len(S))
o = np.argsort(S); S, L, L1 = S[o], L[o], L1[o]
w = 100  # log_freq=10 -> 100 points = 1000 steps
sm = lambda a: np.convolve(a, np.ones(w) / w, mode="valid")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(S, L, color="#c9d2e3", lw=0.6, label="total loss (raw)")
ax.plot(S[w - 1:], sm(L), color="#1f4e9c", lw=2, label="total loss (1000-step avg)")
ax.plot(S[w - 1:], sm(L1), color="#d9822b", lw=2, ls="--", label="L1 loss (1000-step avg)")
ymax = np.nanpercentile(L, 99)
ax.set_ylim(0, ymax * 1.05)
for step, txt_ in [(10000, "10k: 0%"), (20000, "20k: 2%"), (50000, "50k: 22%")]:
    ax.axvline(step, color="gray", lw=0.8, ls=":")
    ax.text(step, ymax * 0.97, txt_, rotation=90, va="top", ha="right", fontsize=9, color="gray")
ax.set_xlabel("training step"); ax.set_ylabel("loss")
ax.set_title("ACT on libero_object, 50k steps (success at n_action_steps=10)")
ax.legend(frameon=False); ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig("act50k_loss.png", dpi=160)
print("points:", len(S), "| final smoothed total:", round(float(sm(L)[-1]), 3), "| final smoothed L1:", round(float(sm(L1)[-1]), 3))

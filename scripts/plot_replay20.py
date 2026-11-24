import re, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- training loss from the log ----
txt = open("stage2_replay20_train.log", errors="ignore").read()
steps, losses = [], []
for s, l in re.findall(r"step:(\S+)\s.*?\bloss:([\d.]+)", txt):
    s = float(s[:-1]) * 1000 if s.endswith("K") else float(s)
    steps.append(s); losses.append(float(l))
steps, losses = np.array(steps), np.array(losses)
w = 20  # log_freq=10, so 20 points = 200 steps
smooth = np.convolve(losses, np.ones(w) / w, mode="valid")

# ---- success numbers (successes, episodes), all at n_action_steps=10 ----
def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)

models = ["Stage 1\n(object)", "Stage 2\nno replay", "Stage 2\nreplay 20%"]
object_res = [(47, 50), (3, 50), (40, 50)]
long_res = [None, (26, 50), (30, 50)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})

ax1.plot(steps, losses, color="#b9c4d6", lw=0.8, label="per-log value")
ax1.plot(steps[w - 1:], smooth, color="#1f4e9c", lw=2, label="moving average (200 steps)")
ax1.set_xlabel("training step"); ax1.set_ylabel("flow-matching loss")
ax1.set_title("Stage 2 replay-20% — training loss")
ax1.legend(frameon=False); ax1.grid(alpha=0.25)

x = np.arange(len(models)); bw = 0.36
for off, res, label, col in [(-bw / 2, object_res, "libero_object", "#1f4e9c"),
                             (bw / 2, long_res, "libero_10", "#d9822b")]:
    for i, r in enumerate(res):
        if r is None:
            ax2.text(x[i] + off, 2, "n/a", ha="center", fontsize=9, color="gray")
            continue
        k, n = r; p = 100 * k / n; lo, hi = wilson(k, n)
        ax2.bar(x[i] + off, p, bw, color=col, label=label if i == (0 if col == "#1f4e9c" else 1) else None)
        ax2.errorbar(x[i] + off, p, yerr=[[p - lo], [hi - p]], color="black", capsize=3, lw=1)
        ax2.text(x[i] + off, hi + 2, f"{p:.0f}%", ha="center", fontsize=9)
ax2.set_xticks(x); ax2.set_xticklabels(models)
ax2.set_ylim(0, 110); ax2.set_ylabel("success rate (%), 50 episodes")
ax2.set_title("Success at n_action_steps=10 (95% Wilson CI)")
ax2.legend(frameon=False, loc="upper center", ncol=2); ax2.grid(axis="y", alpha=0.25)

plt.tight_layout()
plt.savefig("replay20_summary.png", dpi=160)
print("saved replay20_summary.png | log points:", len(steps), "| final smoothed loss:", round(float(smooth[-1]), 3))

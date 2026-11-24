import glob, sys, collections
import numpy as np
n_eps = int(sys.argv[1])            # episodes per task used in the eval
chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 10   # n_action_steps
d0 = sys.argv[3] if len(sys.argv) > 3 else "traces"
files = sorted(glob.glob(f"{d0}/ep_*.npz"))
print(f"{len(files)} episodes found")
rows = collections.defaultdict(list)
for i, f in enumerate(files):
    d = np.load(f); a = d["actions"]; T = len(a)
    grip = a[:, -1] > 0                       # LIBERO: +1 = close (assumed)
    toggles = int(np.sum(grip[1:] != grip[:-1]))
    first_close = int(np.argmax(grip)) if grip.any() else -1
    d_a = np.linalg.norm(np.diff(a[:, :6], axis=0), axis=1)
    idx = np.arange(1, T)
    at_b = d_a[idx % chunk == 0].mean() if np.any(idx % chunk == 0) else np.nan
    off_b = d_a[idx % chunk != 0].mean()
    jerk = at_b / (off_b + 1e-9)
    idle = float(np.mean(np.linalg.norm(a[:, :3], axis=1) < 0.05))
    rows[i // n_eps].append(dict(ok=bool(d["success"]), T=T, close=first_close,
                                 tog=toggles, jerk=jerk, idle=idle))
print("task | succ | steps(ok/fail) | 1st close | toggles | chunk-jerk | idle")
for t, rs in sorted(rows.items()):
    ok = [r for r in rs if r["ok"]]; bad = [r for r in rs if not r["ok"]]
    m = lambda L, k: np.mean([r[k] for r in L]) if L else float("nan")
    print(f"{t:4d} | {len(ok)}/{len(rs)} | {m(ok,'T'):6.0f}/{m(bad,'T'):6.0f} | "
          f"{m(rs,'close'):7.0f} | tog ok {m(ok,'tog'):5.1f} / fail {m(bad,'tog'):5.1f} | {m(rs,'jerk'):6.2f} | {m(rs,'idle'):.2f}")

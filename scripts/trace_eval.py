import atexit, os, functools
import numpy as np
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv

OUT = "traces"; os.makedirs(OUT, exist_ok=True)
state = {"ep": -1, "acts": [], "rews": [], "success": False}

def flush():
    if state["ep"] >= 0 and len(state["acts"]) > 5:
        np.savez(f"{OUT}/ep_{state['ep']:04d}.npz",
                 actions=np.array(state["acts"]), rewards=np.array(state["rews"]),
                 success=state["success"])
    state["acts"], state["rews"], state["success"] = [], [], False
atexit.register(flush)

def get_success(info):
    s = False
    if isinstance(info, dict):
        fi = info.get("final_info")
        if isinstance(fi, dict) and "is_success" in fi:
            s |= bool(np.any(fi["is_success"]))
        elif fi is not None:
            try:
                s |= any(isinstance(i, dict) and bool(i.get("is_success", False)) for i in fi)
            except TypeError:
                pass
        if "is_success" in info:
            s |= bool(np.any(info["is_success"]))
    return s

classes = [SyncVectorEnv, AsyncVectorEnv]
try:
    from lerobot.sim import LocalVectorEnv
    classes.append(LocalVectorEnv)
except Exception as e:
    print("no LocalVectorEnv:", e)
print("patching:", [c.__name__ for c in classes])
for cls in classes:
    orig_reset, orig_step = cls.reset, cls.step
    def mk_reset(f):
        @functools.wraps(f)
        def r(self, *a, **k):
            flush(); state["ep"] += 1
            return f(self, *a, **k)
        return r
    def mk_step(f):
        @functools.wraps(f)
        def s(self, actions, *a, **k):
            out = f(self, actions, *a, **k)
            state["acts"].append(np.array(actions)[0].copy())
            state["rews"].append(float(np.array(out[1])[0]))
            state["success"] |= get_success(out[4])
            return out
        return s
    cls.reset, cls.step = mk_reset(orig_reset), mk_step(orig_step)

from lerobot.scripts.lerobot_eval import main
main()

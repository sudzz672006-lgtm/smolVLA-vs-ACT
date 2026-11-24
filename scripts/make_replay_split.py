"""Build the stage 2 replay episode list: all libero_10 episodes + a random 20% of object episodes.
Usage: python scripts/make_replay_split.py splits/object_episodes.json splits/long10.json splits/long10_replay20.json
"""
import json, random, sys
obj_path, long_path, out_path = sys.argv[1:4]
random.seed(0)
obj, long = json.load(open(obj_path)), json.load(open(long_path))
replay = random.sample(obj, round(0.20 * len(obj)))
mix = sorted(set(long + replay))
json.dump(mix, open(out_path, "w"))
print(f"libero_10: {len(long)}  object replay: {len(replay)}  total: {len(mix)}")

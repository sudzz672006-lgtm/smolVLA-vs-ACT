import ast, re, sys
log = sys.argv[1]
txt = open(log).read()
m = re.search(r"Aggregated Metrics for per_task:\s*\n.*?(\[\{.*\}\])", txt, re.S)
rows = ast.literal_eval(m.group(1).split("\n")[0])

from libero.libero import benchmark
suite = benchmark.get_benchmark_dict()[rows[0]["task_group"]]()

for r in rows:
    tid = r["task_id"]
    print(f"\nTask {tid}: {suite.get_task(tid).language}  ->  {r['n_success']}/{r['n_episodes']}")
    for ok, v in zip(r["metrics"]["successes"], r["metrics"]["video_paths"]):
        print(f"   {'SUCCESS' if ok else 'FAIL   '}  {v}")

"""Live-style ECHO bench against a running llama-server, replicating josh's
protocol exactly: per prompt, erase slot 0, then /completion with the frozen
token ids, n_predict=192, cache_prompt=false, temperature=0; acceptance =
timings.draft_n_accepted / timings.draft_n; throughput = predicted t/s.

Offline hidden-state proxies failed to reproduce the live ordering (both the
prose held-out AND captures of these exact prompts) — generation-time
acceptance is a different measurement, so we measure generation.

Usage: bench_live.py <server-url> <prompts.json> <label>
"""
import json
import sys
import time
import urllib.request

URL, PROMPTS, LABEL = sys.argv[1], sys.argv[2], sys.argv[3]

prompts = json.load(open(PROMPTS))
by_class = {}
for p in prompts:
    by_class.setdefault(p["class"], []).append(p)


def post(path, payload, timeout=1800):
    req = urllib.request.Request(URL + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


results = {}
for cls in ("short", "medium", "long", "xlong"):
    acc_n = acc_d = 0
    tps = []
    for p in sorted(by_class.get(cls, []), key=lambda x: x["name"]):
        # erase slot 0 so nothing is reused (hybrid model: KV removal is fatal;
        # erase is the safe reset)
        try:
            post("/slots/0?action=erase", {})
        except Exception:
            pass
        t0 = time.time()
        d = post("/completion", {
            "prompt": p["tokens"],
            "n_predict": 192,
            "cache_prompt": False,
            "temperature": 0,
        })
        t = d.get("timings", {})
        dn, da = t.get("draft_n", 0), t.get("draft_n_accepted", 0)
        pred_ps = t.get("predicted_per_second", 0.0)
        acc_n += da
        acc_d += dn
        tps.append(pred_ps)
        print(f"  {p['name']:<10} {p['n_tokens']:>6} tok  "
              f"acc {da}/{dn} ({100.0 * da / max(dn, 1):.1f}%)  "
              f"{pred_ps:.1f} t/s  ({time.time() - t0:.0f}s)", flush=True)
    acc = 100.0 * acc_n / max(acc_d, 1)
    mtps = sum(tps) / max(len(tps), 1)
    results[cls] = (acc, mtps)
    print(f"{LABEL} {cls:<7}: {acc:.1f}% / {mtps:.1f} t/s", flush=True)

print("\nRESULT " + LABEL + " " +
      " ".join(f"{c}={results[c][0]:.1f}%/{results[c][1]:.1f}t/s"
               for c in ("short", "medium", "long", "xlong")), flush=True)

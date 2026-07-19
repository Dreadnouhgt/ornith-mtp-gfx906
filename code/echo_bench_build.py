"""Build a frozen ECHO-shaped benchmark.

Our other benchmark is software licence text, which looks nothing like what this
deployment serves: Discord conversations against a ship database, with a ~1.5k
token system prompt and XML tool calls. Draft acceptance is strongly
content-dependent, so tuning on licence prose does not necessarily tune for
production traffic.

System prompt and tool schemas come from the live slot prefix, so they are
exactly what production sends. Assistant turns are generated once by the served
model and then frozen: the benchmark replays identical token sequences forever,
which is what makes runs comparable.

Long conversations are built the way real ones get long - large tool payloads
(wiki page reads, multi-row database results) interleaved with answers - rather
than by generating tens of thousands of tokens of assistant prose, which would
be both slow and unrepresentative.

Output: echo_bench_prompts.json  [{name, class, n_tokens, tokens:[...]}, ...]
"""
import json
import random
import sys
import urllib.request

BASE = "http://localhost:8089"
CTX = "<scratchpad>/echo_ctx.json"
OUT = "<home>/llama-rocm-build/echo_bench_prompts.json"
random.seed(11)

# (class name, target prompt tokens, how many)
PLAN = [("short", 2_000, 3), ("medium", 8_000, 3), ("long", 16_000, 3), ("xlong", 30_000, 3)]


def erase_slot():
    """Any KV removal breaks this hybrid model's recurrent state - even keeping
    97.6% of the cache faults. Multi-turn re-rendering always diverges at the
    tail (the generation-prompt suffix is replaced by real content), so wipe the
    slot before every call rather than letting the server reuse it."""
    try:
        req = urllib.request.Request(BASE + "/slots/0?action=erase", data=b"{}",
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=60).read()
    except Exception:
        pass


def post(path, payload, timeout=900):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


ctx = json.load(open(CTX))
SYSTEM, TOOLS = ctx["system"], ctx["tools"]

QUESTIONS = [
    "ECHO what's the best mining loadout for a Cutter right now?",
    "ECHO compare the Corsair and the Vanguard destroyer for solo PvE",
    "ECHO which faction gives the best reputation rewards for hauling?",
    "ECHO what's the max ship size class in the game?",
    "ECHO how do I unlock the tier 3 engineering skill tree?",
    "ECHO what stations sell refined titanium and where are they?",
    "ECHO explain how shield hardening works against kinetic damage",
    "ECHO is the Frigate worth it over a Gunship for mid-game mining?",
    "ECHO plan me a mining route through the outer belt with fuel stops",
    "ECHO which missions give the most credits per hour at my level?",
]
FOLLOWUPS = [
    "what about for a bigger ship?", "how does that compare to the alternative?",
    "give me the exact numbers", "and if I'm playing solo?",
    "what's the cheapest way to do that?", "any downsides to that approach?",
    "pull the full wiki page on that", "what changed in the last patch?",
]
MANUF = ["Aegis Dynamics", "Roberts Industries", "Kraith Collective", "Nova Foundry"]
CLASSES = ["Cutter", "Gunship", "Corvette", "Frigate", "Destroyer", "Hauler"]


def ship_rows(n):
    return [{
        "name": f"VG-{random.randint(100,999)}", "class": random.choice(CLASSES),
        "size": random.choice(["Small", "Medium", "Large"]),
        "hull": random.randint(2000, 45000), "shield": random.randint(1000, 30000),
        "cargo": random.randint(50, 4000), "speed": random.randint(80, 420),
        "hardpoints": random.randint(2, 14), "manufacturer": random.choice(MANUF),
        "price_credits": random.randint(120_000, 48_000_000),
    } for _ in range(n)]


def wiki_page(paras):
    """Long-form wiki text, the way read_full_wiki_page returns it."""
    out = []
    for i in range(paras):
        out.append(
            f"== Section {i+1}: {random.choice(['Overview','Loadouts','Mining Yield','Faction Standing','Combat Role','Upgrades'])} ==\n"
            f"The {random.choice(CLASSES)} platform manufactured by {random.choice(MANUF)} occupies a "
            f"{random.choice(['contested','well established','narrow','dominant'])} position in the current meta. "
            f"Baseline hull integrity sits near {random.randint(2000,45000)} with shield capacity around "
            f"{random.randint(1000,30000)}, and pilots typically fit {random.randint(2,14)} hardpoints. "
            f"Yield per hour in the outer belt averages {random.randint(20,900)} units of refined ore, "
            f"though this varies with scanner tier and cargo capacity of {random.randint(50,4000)}. "
            f"Faction standing modifies station fees by up to {random.randint(2,35)} percent.")
    return "\n\n".join(out)


def tool_payload():
    kind = random.random()
    if kind < 0.45:
        return json.dumps({"results": ship_rows(random.randint(8, 40))})
    if kind < 0.8:
        return wiki_page(random.randint(4, 14))
    return json.dumps({"results": ship_rows(random.randint(3, 10)),
                       "notes": wiki_page(random.randint(2, 5))})


def n_tokens(messages):
    prompt = post("/apply-template", {"messages": messages, "tools": TOOLS})["prompt"]
    return len(post("/tokenize", {"content": prompt})["tokens"]), prompt


def build(target):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": random.choice(QUESTIONS)}]
    for cycle in range(60):
        cur, _ = n_tokens(msgs)
        if cur >= target:
            break
        # tool call -> large tool result -> generated answer
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{
            "type": "function", "function": {
                "name": random.choice(["query_ship_database", "read_full_wiki_page"]),
                "arguments": json.dumps({"query": random.choice(QUESTIONS)[:50]})}}]})
        msgs.append({"role": "tool", "content": tool_payload()})
        erase_slot()
        r = post("/v1/chat/completions", {"model": "Ornith-Q8", "messages": msgs,
                                          "max_tokens": 500, "temperature": 0.7})
        msgs.append({"role": "assistant",
                     "content": r["choices"][0]["message"].get("content") or ""})
        msgs.append({"role": "user", "content": random.choice(FOLLOWUPS)})
    return msgs


import os
out = json.load(open(OUT)) if os.path.exists(OUT) else []
have = {o["name"] for o in out}
print(f"resuming with {len(out)} prompts already built", flush=True)
for cls, target, count in PLAN:
    for i in range(count):
        if f"{cls}-{i}" in have:
            continue
        msgs = build(target)
        n, prompt = n_tokens(msgs)
        toks = post("/tokenize", {"content": prompt})["tokens"]
        out.append({"name": f"{cls}-{i}", "class": cls, "n_tokens": len(toks), "tokens": toks})
        print(f"{cls}-{i}: {len(toks)} tokens (target {target})", flush=True)
        json.dump(out, open(OUT, "w"))   # checkpoint as we go

print(f"\nwrote {len(out)} frozen prompts to {OUT}")

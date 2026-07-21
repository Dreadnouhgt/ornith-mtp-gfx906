"""Generate ECHO-shaped TRAINING data: same distribution as the live bench
(production system+tools prefix, tool-call conversations, RNG payloads,
model-generated answers), different content (different RNG seed, extended
question list) — so it trains the serving distribution without contaminating
the frozen benchmark.

Mechanics: the exact production prefix is the 1544-token common prefix of the
frozen bench prompts. Turn markers are read from the bench prompts themselves
(detokenized), so the rendered template matches production byte-for-byte.
Docs are written as token-id lines for the ids-mode capture tool.

Usage: gen_echo_corpus.py <server> <prefix.json> <out_ids.txt> <plan e.g. short:2500:40,medium:9500:30>
"""
import json
import random
import sys
import threading
import urllib.request

BASE, PREFIX_P, OUT, PLAN_S = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
SEED = 4242
random.seed(SEED)

PREFIX = json.load(open(PREFIX_P))


def post(path, payload, timeout=1800):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def detok(tokens):
    return post("/detokenize", {"tokens": tokens})["content"]


def tok(text):
    return post("/tokenize", {"content": text, "add_special": False, "with_pieces": False})["tokens"]


# ---- learn the exact turn markers from the bench prompts themselves ----
PREFIX_TEXT = detok(PREFIX)
# the common prefix ends INSIDE the first user turn (every bench question
# starts with "ECHO"); trim back to before the user-turn opener
cut = PREFIX_TEXT.rfind("<|im_start|>user\n")
assert cut > 0
PREFIX_TEXT = PREFIX_TEXT[:cut]
# prefix ends right before the first user turn; the template opener is
# whatever the production template emits between system block and user content
U_OPEN = "<|im_start|>user\n"
U_CLOSE = "<|im_end|>\n"
A_OPEN = "<|im_start|>assistant\n"
A_CLOSE = "<|im_end|>\n"
assert PREFIX_TEXT.rstrip().endswith("<|im_end|>") or "<|im_start|>" in PREFIX_TEXT, \
    "unexpected template; inspect PREFIX_TEXT"

# josh's production question style + new same-style variants (content differs
# from the frozen bench where possible; payloads/followups are RNG anyway)
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
    # new, same register
    "ECHO what's the fastest way to grind salvage reputation this patch?",
    "ECHO best turret setup for a Corvette against fighter swarms?",
    "ECHO how does cargo insurance work when I get boarded?",
    "ECHO compare refinery yields between station tiers",
    "ECHO what should I fly for stealth courier missions?",
    "ECHO how do jump drive cooldowns scale with ship mass?",
    "ECHO which upgrades matter most for a budget Hauler build?",
    "ECHO is dual-mining with a partner more efficient than solo?",
    "ECHO what does the scanner tier actually change for prospecting?",
    "ECHO how do I fit a Destroyer for long-range patrol?",
]
FOLLOWUPS = [
    "what about for a bigger ship?", "how does that compare to the alternative?",
    "give me the exact numbers", "and if I'm playing solo?",
    "what's the cheapest way to do that?", "any downsides to that approach?",
    "pull the full wiki page on that", "what changed in the last patch?",
    "summarize that as a build list", "which of those is best value?",
]
MANUF = ["Aegis Dynamics", "Roberts Industries", "Kraith Collective", "Nova Foundry"]
CLASSES = ["Cutter", "Gunship", "Corvette", "Frigate", "Destroyer", "Hauler"]


def ship_rows(rng, n):
    return [{
        "name": f"VG-{rng.randint(100,999)}", "class": rng.choice(CLASSES),
        "size": rng.choice(["Small", "Medium", "Large"]),
        "hull": rng.randint(2000, 45000), "shield": rng.randint(1000, 30000),
        "cargo": rng.randint(50, 4000), "speed": rng.randint(80, 420),
        "hardpoints": rng.randint(2, 14), "manufacturer": rng.choice(MANUF),
        "price_credits": rng.randint(120_000, 48_000_000),
    } for _ in range(n)]


def wiki_page(rng, paras):
    out = []
    for i in range(paras):
        out.append(
            f"== Section {i+1}: {rng.choice(['Overview','Loadouts','Mining Yield','Faction Standing','Combat Role','Upgrades'])} ==\n"
            f"The {rng.choice(CLASSES)} platform manufactured by {rng.choice(MANUF)} occupies a "
            f"{rng.choice(['contested','well established','narrow','dominant'])} position in the current meta. "
            f"Baseline hull integrity sits near {rng.randint(2000,45000)} with shield capacity around "
            f"{rng.randint(1000,30000)}, and pilots typically fit {rng.randint(2,14)} hardpoints. "
            f"Yield per hour in the outer belt averages {rng.randint(20,900)} units of refined ore, "
            f"though this varies with scanner tier and cargo capacity of {rng.randint(50,4000)}. "
            f"Faction standing modifies station fees by up to {rng.randint(2,35)} percent.")
    return "\n\n".join(out)


def tool_payload(rng):
    kind = rng.random()
    if kind < 0.45:
        return json.dumps({"results": ship_rows(rng, rng.randint(8, 40))})
    if kind < 0.8:
        return wiki_page(rng, rng.randint(4, 14))
    return json.dumps({"results": ship_rows(rng, rng.randint(3, 10)),
                       "notes": wiki_page(rng, rng.randint(2, 5))})


lock = threading.Lock()
stats = {"done": 0, "tokens": 0}


def gen_think(text, n_predict, rng):
    """Model-generated thinking segment — the generation-time distribution the
    drafter is actually scored on (bench continuations start with <think>)."""
    return post("/completion", {
        "prompt": text + A_OPEN + "<think>\n",
        "n_predict": n_predict, "temperature": 0.7, "top_p": 0.95,
        "cache_prompt": False,
        "stop": ["<|im_end|>"],
    })["content"]


def build_doc(seed, target_tokens):
    """Mirrors the decoded bench structure: user q -> EMPTY assistant (tool
    call) -> tool_response -> empty/short assistant -> followup ... ending in a
    generated <think> segment. Mid-doc assistant turns alternate empty (bench
    shape) and short generated (production shape)."""
    rng = random.Random(seed)
    text = PREFIX_TEXT + U_OPEN + rng.choice(QUESTIONS) + U_CLOSE
    cycle = 0
    while True:
        ids_n = len(tok(text))
        if ids_n >= target_tokens:
            break
        text += A_OPEN + A_CLOSE                    # tool-call turn renders empty
        text += (U_OPEN + f"<tool_response>\n{tool_payload(rng)}\n</tool_response>" + U_CLOSE)
        if cycle % 2 == 0:
            text += A_OPEN + A_CLOSE                # bench-shaped: empty answer
        else:                                       # production-shaped: real answer
            gen = gen_think(text, 300, rng)
            text += A_OPEN + "<think>\n" + gen.strip() + A_CLOSE
        text += U_OPEN + rng.choice(FOLLOWUPS) + U_CLOSE
        cycle += 1
        if cycle > 80:
            break
    # closing generated segment = exactly what live drafting decodes over
    gen = gen_think(text, 600, rng)
    text += A_OPEN + "<think>\n" + gen.strip()
    ids = tok(text)
    return ids[:target_tokens + 800]


def worker(jobs):
    for seed, target in jobs:
        try:
            ids = build_doc(seed, target)
        except Exception as e:
            print(f"  seed {seed} failed: {str(e)[:90]}", flush=True)
            continue
        with lock:
            stats["done"] += 1
            stats["tokens"] += len(ids)
            with open(OUT, "a") as f:
                f.write(" ".join(map(str, ids)) + "\n")
            print(f"[{stats['done']}] seed={seed} {len(ids)} tok (total {stats['tokens']})", flush=True)


plan = []
seed = 10_000
for part in PLAN_S.split(","):
    name, target, count = part.split(":")
    for _ in range(int(count)):
        plan.append((seed, int(target)))
        seed += 1
random.shuffle(plan)
N_PAR = 4
threads = [threading.Thread(target=worker, args=(plan[i::N_PAR],)) for i in range(N_PAR)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(f"done: {stats['done']} docs, {stats['tokens']} tokens", flush=True)

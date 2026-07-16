from __future__ import annotations

# IG = Information Geometry (informaciona geometrija) 

"""
inspiration / upgrade  <--->  inspiracija / nadogradnja


Dragan Stošić / dva rada LUCES / ESP32 osvetljenje: 

1. Empirijska IG: Fisher metric, Multi-Chart (kad signal padne prelaz chartova), Christoffel / Levi-Civita, Histerezis.
https://zenodo.org/records/20094759
(DOI 10.5281/zenodo.20094759) — Fisher, chartovi, Christoffel, histerezis.

2. Ceo experimentalni sloj (paper + data + PVS) — ovo je „journal-ready“ paket. 
isti Manifold + mikro-ekscitacija + Fisher-preconditioned kontrola (A/B −25% jitter) + PVS dokazi + senzorski CSV.
https://zenodo.org/records/20389804
(novija PDF verzija: https://zenodo.org/records/20393695)
Naslov: Excitation-Dependent Observability Geometry…
Sadrži: paper 15 str, 6 CSV (boot…), serial logovi, PVS dokazi, A/B Boot 291 (GEO −25% jitter).
"""


"""
Fisher metrika na porodici raspodela nad istorijom (npr. frekvencije / uslovne raspodele)
multi-chart kad „observabilnost“ padne (npr. drugačiji režim / era)
natural gradient (Fisher precondition) ako nešto optimizujem 
histerezis putanja kroz vreme
mikro-ekscitacija (loto ne možeš da „probudiš“ kao lampu); PVS dokazi.
"""



"""
0.5 BRIGHT + 0.5 CCT → next.

kombinovana ekscitacija (paper 2 / boot287 GEO night stil).

Isti BRIGHT (min/max) i CCT (cross-zone) Δρ skorovi kao v11.
  COMB: skor = 0.5·BRIGHT + 0.5·CCT
Ban last; jedna next. CSV ceo, seed=39.
"""



import csv
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

SEED = 39
FRONT_N = 39
FRONT_SELECT = 7
MIN_PAIR = 20
CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "loto7_4650_k56.csv"

np.random.seed(SEED)


def load_draws(csv_path: Path = CSV_PATH) -> np.ndarray:
    draws = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) < FRONT_SELECT:
                continue
            try:
                draw = sorted(int(x.strip()) for x in row[:FRONT_SELECT])
            except ValueError:
                continue
            if len(draw) == FRONT_SELECT and all(1 <= x <= FRONT_N for x in draw):
                if len(set(draw)) == FRONT_SELECT:
                    draws.append(draw)
    if not draws:
        raise ValueError(f"Nema validnih kola u {csv_path}")
    return np.array(draws, dtype=int)


def global_p(draws: np.ndarray) -> np.ndarray:
    cnt = Counter(draws.reshape(-1).tolist())
    n_slots = len(draws) * FRONT_SELECT
    return np.array([cnt.get(i, 0) / n_slots for i in range(1, FRONT_N + 1)], dtype=float)


def pair_transition_tables(draws: np.ndarray) -> tuple[dict, dict]:
    present = np.zeros((len(draws), FRONT_N), dtype=np.uint8)
    for i, d in enumerate(draws):
        for x in d.tolist():
            present[i, int(x) - 1] = 1
    pair_count: dict[tuple[int, int], int] = {}
    pair_next: dict[tuple[int, int], np.ndarray] = {}
    for t in range(len(draws) - 1):
        xs = np.where(present[t] == 1)[0]
        ys = np.where(present[t + 1] == 1)[0]
        for a, b in combinations(xs.tolist(), 2):
            key = (a, b) if a < b else (b, a)
            pair_count[key] = pair_count.get(key, 0) + 1
            if key not in pair_next:
                pair_next[key] = np.zeros(FRONT_N, dtype=np.float64)
            for yi in ys:
                pair_next[key][yi] += 1.0
    return pair_count, pair_next


def conditional_from_query_pairs(
    query: list[int],
    pair_count: dict,
    pair_next: dict,
    min_pair: int = MIN_PAIR,
) -> np.ndarray:
    carriers = sorted(int(x) - 1 for x in query)
    masses = []
    for a, b in combinations(carriers, 2):
        key = (a, b)
        c = pair_count.get(key, 0)
        if c < min_pair:
            continue
        masses.append(pair_next[key] / float(c))
    if not masses:
        for a, b in combinations(carriers, 2):
            key = (a, b)
            c = pair_count.get(key, 0)
            if c <= 0:
                continue
            masses.append(pair_next[key] / float(c))
    if not masses:
        mass = np.ones(FRONT_N, dtype=np.float64)
    else:
        mass = np.mean(np.stack(masses, axis=0), axis=0)
    mass = mass + 1e-6
    return mass / mass.sum()


def observability(p_cond: np.ndarray, p_glob: np.ndarray) -> float:
    return float(np.linalg.norm(p_cond - p_glob))


def zone(n: int) -> str:
    if n <= 13:
        return "low"
    if n <= 26:
        return "mid"
    return "high"


def score_bright(
    last: list[int],
    rho0: float,
    p_glob: np.ndarray,
    pair_count: dict,
    pair_next: dict,
    ban: set[int],
) -> dict[int, float]:
    pivots = {min(last), max(last)}
    out: dict[int, float] = {}
    for y in range(1, FRONT_N + 1):
        if y in ban:
            out[y] = -1e18
            continue
        deltas = []
        for x in pivots:
            q = sorted((set(last) - {x}) | {y})
            if len(q) != FRONT_SELECT:
                continue
            rho = observability(
                conditional_from_query_pairs(q, pair_count, pair_next), p_glob
            )
            deltas.append(rho - rho0)
        out[y] = float(np.mean(deltas)) if deltas else -1e18
    return out


def score_cct(
    last: list[int],
    rho0: float,
    p_glob: np.ndarray,
    pair_count: dict,
    pair_next: dict,
    ban: set[int],
) -> dict[int, float]:
    out: dict[int, float] = {}
    for y in range(1, FRONT_N + 1):
        if y in ban:
            out[y] = -1e18
            continue
        zy = zone(y)
        deltas = []
        for x in last:
            if zone(x) == zy:
                continue
            q = sorted((set(last) - {x}) | {y})
            if len(q) != FRONT_SELECT:
                continue
            rho = observability(
                conditional_from_query_pairs(q, pair_count, pair_next), p_glob
            )
            deltas.append(rho - rho0)
        out[y] = float(np.mean(deltas)) if deltas else -1e18
    return out


def combined_scores(
    sc_b: dict[int, float], sc_c: dict[int, float], ban: set[int]
) -> dict[int, float]:
    out = {}
    for n in range(1, FRONT_N + 1):
        if n in ban:
            out[n] = -1e18
        else:
            out[n] = 0.5 * float(sc_b[n]) + 0.5 * float(sc_c[n])
    return out


def _combo_fit(
    combo: list[int],
    score: dict[int, float],
    target_sum: float,
    pos_means: list[float],
    target_odd: float,
    ban: set[int],
) -> float:
    nums = sorted(combo)
    if any(x in ban for x in nums):
        return -1e18
    s = sum(score[x] for x in nums)
    s -= 0.08 * abs(sum(nums) - target_sum)
    s -= 0.04 * sum(abs(nums[i] - pos_means[i]) for i in range(FRONT_SELECT))
    odd = sum(1 for x in nums if x % 2)
    s -= 0.3 * abs(odd - target_odd)
    return s


def predict_next(draws: np.ndarray, score: dict[int, float], ban: set[int]) -> list[int]:
    ranked = sorted((n for n in score if n not in ban), key=lambda n: (-score[n], n))
    target_sum = float(draws.sum(axis=1).mean())
    pos_means = [float(draws[:, i].mean()) for i in range(FRONT_SELECT)]
    target_odd = float(np.mean([sum(1 for x in d if x % 2) for d in draws]))

    candidates = [sorted(ranked[:FRONT_SELECT])]
    for start in range(0, min(20, len(ranked) - FRONT_SELECT + 1)):
        candidates.append(sorted(ranked[start : start + FRONT_SELECT]))

    best, best_fit = None, -1e18
    for base in candidates:
        fit = _combo_fit(base, score, target_sum, pos_means, target_odd, ban)
        if fit > best_fit:
            best_fit, best = fit, list(base)
        for i in range(FRONT_SELECT):
            for repl in ranked[:30]:
                cand = sorted(set(base[:i] + base[i + 1 :] + [repl]))
                if len(cand) != FRONT_SELECT:
                    continue
                fit = _combo_fit(cand, score, target_sum, pos_means, target_odd, ban)
                if fit > best_fit:
                    best_fit, best = fit, cand
    return best if best is not None else sorted(ranked[:FRONT_SELECT])


def run_ig_02_v12(csv_path: Path = CSV_PATH) -> None:
    draws = load_draws(csv_path)
    last = [int(x) for x in draws[-1].tolist()]
    ban = set(last)
    p_glob = global_p(draws)
    pair_count, pair_next = pair_transition_tables(draws)
    rho0 = observability(
        conditional_from_query_pairs(last, pair_count, pair_next), p_glob
    )

    sc_b = score_bright(last, rho0, p_glob, pair_count, pair_next, ban)
    sc_c = score_cct(last, rho0, p_glob, pair_count, pair_next, ban)
    sc = combined_scores(sc_b, sc_c, ban)
    combo = predict_next(draws, sc, ban)

    print(f"CSV: {csv_path.name}")
    print(f"Kola: {len(draws)} | seed={SEED} | ig_02_v12 combined BRIGHT+CCT")
    print(f"last: {last} | rho0={rho0:.6f}")
    print()

    ranked = sorted(
        (
            (n, float(sc_b[n]), float(sc_c[n]), float(sc[n]))
            for n in range(1, FRONT_N + 1)
            if n not in ban
        ),
        key=lambda t: (-t[3], t[0]),
    )
    print("=== top12 COMB (B, C, 0.5B+0.5C) ===")
    print([(n, round(b, 6), round(c, 6), round(m, 6)) for n, b, c, m in ranked[:12]])
    print()
    print("=== next (ig_02_v12 combined) ===")
    print("next:", combo)
    print("overlap last:", sorted(set(combo) & ban))


if __name__ == "__main__":
    run_ig_02_v12()



"""
CSV: loto7_4650_k56.csv
Kola: 4650 | seed=39 | ig_02_v12 combined BRIGHT+CCT
last: [4, 5, 6, 11, 12, 18, 28] | rho0=0.009740

=== top12 COMB (B, C, 0.5B+0.5C) ===
[(13, 0.000405, 0.001287, 0.000846), (29, 0.000964, 0.000601, 0.000783), (19, 0.001035, 0.000446, 0.00074), (2, 0.000383, 0.001024, 0.000703), (36, 0.00047, 0.000362, 0.000416), (26, 0.000727, 1.3e-05, 0.00037), (3, 5.8e-05, 0.000492, 0.000275), (9, -1.4e-05,0.000331, 0.000158), (25, 3.2e-05, 0.000132, 8.2e-05), (35, 0.000275, -0.000131, 7.2e-05), (8, -0.00055, 0.000658, 5.4e-05), (38,0.000377, -0.000373, 2e-06)]

=== next (ig_02_v12 combined) ===
next: [8, x, 16, y, 25, z, 35]
overlap last: []
"""

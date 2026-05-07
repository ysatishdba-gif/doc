"""
Build per-CUI expansion lookup using ALL MRREL relations,
restricted to within the same semantic type as the source.

Stage 1 of 2: produces cui_expansion_lookup.pkl.
Stage 2 (run_topics.py) consumes that pickle and adds topic IDs.
Splitting the steps means we can re-run topic clustering with different
parameters without rebuilding the expensive expansion data.

For each CUI:
  - centrality = -log(reach / total), where reach is the number of unique
    same-semantic-type CUIs reachable via any MRREL relation.
  - expansion = list of same-semantic-type CUIs reachable from the source,
    sorted by hop distance, stopping past any neighbor whose centrality is
    more than CENTRALITY_DROP lower than the source.

Output pickle structure:
  {
    cui: {
      "own_label":     str,
      "semantic_type": str,
      "centrality":    float,
      "expansion": [
        {"cui": str, "label": str, "centrality": float, "rel": str},
        ...
      ]
    }
  }
"""

import math
import os
import pickle
import sys
import time
from collections import defaultdict, deque
from google.cloud import bigquery

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass


PROJECT = "your-gcp-project-id"
DATASET = "your-bigquery-dataset"
OUTPUT  = "cui_expansion_lookup.pkl"

# Bounding parameters — applied together to keep expansion sizes manageable.
#
# Why we need both:
#   With ALL MRREL relations, the within-semantic-type graph was densely
#   connected and centrality alone collapsed (every CUI reaching ~everyone
#   produced 10K+ expansions). The relation filter restores tree-like
#   structure so centrality is meaningful again as a bound.

# 1) Centrality drop — destination must be no more than this many units
#    less central than the source.
CENTRALITY_DROP = 0.5

# 2) Relation filter — only walk through these REL types. Drops RO
#    ("related other") which is what densely connects everything.
ALLOWED_RELS = {"PAR", "CHD", "RB", "RN", "SY", "SIB"}


# ─────────────────────────────────────────────────────────────
# BIGQUERY
# ─────────────────────────────────────────────────────────────

def query_concepts(client):
    sql = f"""
        SELECT DISTINCT
            c.CUI,
            c.STR AS label,
            LOWER(COALESCE(s.STY, 'unknown')) AS semantic_type
        FROM `{PROJECT}.{DATASET}.MRCONSO` c
        LEFT JOIN `{PROJECT}.{DATASET}.MRSTY` s ON c.CUI = s.CUI
        WHERE c.SAB='SNOMEDCT_US'
          AND c.LAT='ENG'
          AND c.ISPREF='Y'
          AND c.SUPPRESS='N'
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY c.CUI ORDER BY s.STY NULLS LAST
        ) = 1
    """
    print("  Querying concepts...")
    t = time.time()
    df = client.query(sql).to_dataframe()
    print(f"  -> {len(df):,} concepts ({time.time()-t:.1f}s)")
    return df


def query_relations(client):
    """Pull broader/narrower/synonym relations only. Excludes RO (related-
    other) which densely connects everything and breaks the centrality
    bound."""
    rels_csv = ", ".join(f"'{r}'" for r in sorted(ALLOWED_RELS))
    sql = f"""
        SELECT
            CUI1 AS cui_a,
            CUI2 AS cui_b,
            REL  AS rel
        FROM `{PROJECT}.{DATASET}.MRREL`
        WHERE SAB='SNOMEDCT_US'
          AND SUPPRESS='N'
          AND REL IN ({rels_csv})
    """
    print(f"  Querying relations (REL IN {sorted(ALLOWED_RELS)})...")
    t = time.time()
    df = client.query(sql).to_dataframe()
    print(f"  -> {len(df):,} relation rows ({time.time()-t:.1f}s)")
    return df


# ─────────────────────────────────────────────────────────────
# GRAPH
# ─────────────────────────────────────────────────────────────

def build_graph(concepts_df, relations_df):
    """Multi-relation graph: cui -> {neighbor_cui: rel_type}.

    All relations included. Auto-symmetric (each row produces edges in
    both directions) so expansion catches parents and children, broader
    and narrower, etc.
    """
    print("  Building multi-relation graph...")
    t = time.time()

    concepts_df = concepts_df.copy()
    concepts_df["semantic_type"] = (
        concepts_df["semantic_type"].fillna("unknown").str.lower()
    )
    concepts_df["full_label"] = (
        concepts_df["label"] + " (" + concepts_df["semantic_type"] + ")"
    )

    nodes = {}
    for r in concepts_df[["CUI", "full_label", "semantic_type"]].to_dict("records"):
        nodes[r["CUI"]] = {
            "label":         r["full_label"],
            "semantic_type": r["semantic_type"],
        }

    adj = defaultdict(dict)
    skipped = 0
    for r in relations_df.to_dict("records"):
        a, b, rel = r["cui_a"], r["cui_b"], r["rel"]
        if a not in nodes or b not in nodes:
            skipped += 1
            continue
        if a == b:
            continue
        if b not in adj[a]:
            adj[a][b] = rel
        if a not in adj[b]:
            adj[b][a] = rel

    print(f"  -> {len(nodes):,} nodes, "
          f"{sum(len(v) for v in adj.values()):,} directed edges  "
          f"(skipped {skipped:,} edges to non-SNOMED CUIs)  "
          f"({time.time()-t:.1f}s)")
    return nodes, adj


# ─────────────────────────────────────────────────────────────
# CENTRALITY (same-semantic-type reach)
# ─────────────────────────────────────────────────────────────

def compute_centrality(nodes, adj):
    """Reach is computed within the source's own semantic type only.

    BFS from each CUI, only following edges to neighbors of the same
    semantic type. centrality = -log(reach / total_same_type).

    This means a CUI's centrality reflects how peripheral it is within
    its own type, which is what we want since expansion is type-bounded.
    """
    print("  Computing same-type reach + centrality...")
    t = time.time()

    # Count CUIs per semantic type — used as the denominator for centrality
    type_counts = defaultdict(int)
    for n in nodes:
        type_counts[nodes[n]["semantic_type"]] += 1

    total = len(nodes)
    centrality = {}
    reach = {}

    for i, cui in enumerate(nodes):
        if i % 25000 == 0:
            eta = ((time.time()-t) / max(i, 1)) * (total - i)
            print(f"    {i:,}/{total:,}  ETA {eta:.0f}s", flush=True)

        src_type = nodes[cui]["semantic_type"]

        # BFS within the same semantic type
        seen = {cui}
        q = deque([cui])
        while q:
            n = q.popleft()
            for nb in adj.get(n, {}):
                if nb in seen:
                    continue
                if nodes.get(nb, {}).get("semantic_type") != src_type:
                    continue
                seen.add(nb)
                q.append(nb)

        r = len(seen)
        reach[cui] = r
        # Denominator is count of CUIs in this semantic type, not total
        denom = type_counts.get(src_type, total)
        centrality[cui] = -math.log(r / denom)

    print(f"  -> done ({time.time()-t:.1f}s)")
    return centrality, reach


# ─────────────────────────────────────────────────────────────
# EXPANSION
# ─────────────────────────────────────────────────────────────

def expand_cui(cui, nodes, adj, centrality, drop):
    """BFS within same semantic type, sorted by hop distance.

    Two bounds applied together:
      - Same semantic type as the source (drops cross-type neighbors)
      - Centrality drop (destination not too much more general than source)

    Each neighbor is checked: must be same semantic type AND have
    centrality >= source_centrality - drop. Otherwise rejected (and
    not walked through).
    """
    src_type = nodes[cui]["semantic_type"]
    src_centrality = centrality.get(cui)
    if src_centrality is None:
        return []

    threshold = src_centrality - drop

    seen = {cui}
    accepted = []  # list of (hop, neighbor, rel)

    # First hop neighbors
    current_layer = []
    for nb, rel in adj.get(cui, {}).items():
        if nb == cui or nb in seen:
            continue
        if nodes.get(nb, {}).get("semantic_type") != src_type:
            continue
        nb_centrality = centrality.get(nb)
        if nb_centrality is None or nb_centrality < threshold:
            continue
        seen.add(nb)
        accepted.append((1, nb, rel))
        current_layer.append(nb)

    hop = 1
    while current_layer:
        next_layer = []
        for n in current_layer:
            for nb, rel in adj.get(n, {}).items():
                if nb in seen:
                    continue
                if nodes.get(nb, {}).get("semantic_type") != src_type:
                    continue
                nb_centrality = centrality.get(nb)
                if nb_centrality is None or nb_centrality < threshold:
                    continue
                seen.add(nb)
                accepted.append((hop + 1, nb, rel))
                next_layer.append(nb)
        current_layer = next_layer
        hop += 1

    accepted.sort(key=lambda x: x[0])
    return accepted


def build_expansion_lookup(nodes, adj, centrality, drop):
    print(f"  Building expansion lookup (drop={drop}, same-type only)...")
    t = time.time()
    total = len(nodes)
    lookup = {}
    sizes = []

    for i, cui in enumerate(nodes):
        if i % 25000 == 0:
            eta = ((time.time()-t) / max(i, 1)) * (total - i)
            avg = sum(sizes[-1000:]) / max(len(sizes[-1000:]), 1) if sizes else 0
            print(f"    {i:,}/{total:,}  ETA {eta:.0f}s  "
                  f"avg expansion last 1k: {avg:.0f}", flush=True)

        accepted = expand_cui(cui, nodes, adj, centrality, drop)
        expansion = [
            {
                "cui":        nb_cui,
                "label":      nodes.get(nb_cui, {}).get("label", nb_cui),
                "centrality": round(centrality[nb_cui], 4),
                "rel":        rel,
            }
            for _hop, nb_cui, rel in accepted
        ]
        sizes.append(len(expansion))

        lookup[cui] = {
            "own_label":     nodes[cui]["label"],
            "semantic_type": nodes[cui]["semantic_type"],
            "centrality":    round(centrality[cui], 4),
            "expansion":     expansion,
        }

    print(f"  -> done ({time.time()-t:.1f}s)")
    if sizes:
        sizes_sorted = sorted(sizes)
        print(f"    expansion size stats:")
        print(f"      min:    {sizes_sorted[0]:,}")
        print(f"      median: {sizes_sorted[len(sizes_sorted)//2]:,}")
        print(f"      mean:   {sum(sizes)/len(sizes):.0f}")
        print(f"      p95:    {sizes_sorted[int(len(sizes_sorted)*0.95)]:,}")
        print(f"      max:    {sizes_sorted[-1]:,}")
        n_empty = sum(1 for s in sizes if s == 0)
        print(f"    empty expansions: {n_empty:,} ({n_empty/total*100:.1f}%)")

    return lookup



# ─────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = round(os.path.getsize(path) / 1e6, 1)
    print(f"  Saved {len(obj):,} entries -> {path}  ({size_mb} MB)")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    start = time.time()

    print(f"\n{'='*60}")
    print(f"  Multi-relation expansion builder (same-semantic-type)")
    print(f"  Project   : {PROJECT}")
    print(f"  Dataset   : {DATASET}")
    print(f"  Drop      : {CENTRALITY_DROP}")
    print(f"  Relations : {sorted(ALLOWED_RELS)}")
    print(f"{'='*60}\n")

    client = bigquery.Client(project=PROJECT)

    print("[1/4] Pulling concepts...")
    concepts_df = query_concepts(client)
    print()

    print("[2/4] Pulling relations...")
    relations_df = query_relations(client)
    print()

    print("[3/4] Building graph + centrality...")
    nodes, adj = build_graph(concepts_df, relations_df)
    centrality, reach = compute_centrality(nodes, adj)
    print()

    print("[4/4] Building expansion lookup...")
    lookup = build_expansion_lookup(nodes, adj, centrality, CENTRALITY_DROP)
    print()

    print("Saving pickle...")
    save_pickle(lookup, OUTPUT)
    print(f"\nTotal time: {(time.time()-start)/60:.1f} min")
    print(f"\nNext: run `python run_topics.py` to add topic IDs.")

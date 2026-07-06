"""
slim_temporal.py — read the big temporal formula JSON from GCS and keep only the
fields the runtime needs: name (match key), cui (output), and formula. Only
has_formula concepts are kept; no_formula items are ignored.

Output shape:   { name: [ {"cui": <code>, "formula": <str>}, ... ] }

If a name maps to more than one code, ALL of its entries are kept. Only
exact-duplicate rows (same name + cui) are collapsed. At match time the first
entry for a matched name supplies the output cui + formula.

Fill in CONFIG, then:  python slim_temporal.py
Requires google-cloud-storage + ADC creds.
"""

import json
from google.cloud import storage

# ───────────────────────── CONFIG (fill in) ─────────────────────────
PROJECT_ID = ""        # e.g. "my-gcp-project"
BUCKET     = ""        # your GCS bucket
SRC_PATH   = "Normalized_loinc_classes/temporal_norm_code_to_formula.json"
OUT_LOCAL  = "temporal_name_to_cui.json"
OUT_GCS    = ""        # e.g. "Normalized_loinc_classes/temporal_name_to_cui.json"; "" to skip
# ─────────────────────────────────────────────────────────────────────


def read_json_from_gcs(project_id, blob_path, bucket):
    blob = storage.Client(project=project_id).bucket(bucket).blob(blob_path)
    return json.loads(blob.download_as_text())


def write_json_to_gcs(project_id, blob_path, bucket, obj):
    blob = storage.Client(project=project_id).bucket(bucket).blob(blob_path)
    blob.upload_from_string(json.dumps(obj), content_type="application/json")


def slim(vocab):
    """Return ({name: [{cui, formula}, ...]}, has_formula_rows, total_entries,
    skipped, exact_dups, ignored_no_formula).

    Only has_formula concepts are kept; every no_formula item is ignored."""
    rows = vocab.get("has_formula") or []          # formula-bearing concepts only
    ignored_no_formula = len(vocab.get("no_formula") or [])
    out, seen = {}, {}
    total_entries = skipped = exact_dups = 0
    for r in rows:
        name = r.get("name")
        cui  = r.get("code")
        if not name or not cui:
            skipped += 1
            continue
        if cui in seen.get(name, set()):
            exact_dups += 1            # same name+cui -> collapse
            continue
        out.setdefault(name, []).append({"cui": cui, "formula": r.get("formula")})
        seen.setdefault(name, set()).add(cui)
        total_entries += 1
    return out, len(rows), total_entries, skipped, exact_dups, ignored_no_formula


def main():
    vocab = read_json_from_gcs(PROJECT_ID, SRC_PATH, BUCKET)
    out, total, entries, skipped, exact_dups, ignored = slim(vocab)
    multi = {n: v for n, v in out.items() if len(v) > 1}
    with_formula = sum(1 for v in out.values() for e in v if e.get("formula"))

    print(f"has_formula rows read       : {total:,}")
    print(f"no_formula rows IGNORED     : {ignored:,}")
    print(f"unique names (match keys)   : {len(out):,}")
    print(f"total entries (cui)         : {entries:,}")
    print(f"entries WITH formula        : {with_formula:,} / {entries:,}")
    print(f"names with >1 entry         : {len(multi):,} (all kept)")
    print(f"exact-duplicate rows dropped: {exact_dups:,}")
    print(f"skipped (no name/cui)       : {skipped:,}")
    print("sample (a multi-entry name if any):")
    for name, v in (list(multi.items())[:5] or list(out.items())[:5]):
        print(f"   {name!r}: {v}")

    # If every formula is null, the has_formula rows don't store it under
    # 'formula'. Show the real keys so you can fix the r.get('formula') below.
    if entries and with_formula == 0:
        sample_keys = list((vocab.get("has_formula") or [{}])[0].keys())
        print("\n!!! WARNING: every formula came out null.")
        print(f"!!! has_formula row keys are: {sample_keys}")
        print("!!! The formula is under a different key — change r.get('formula').")

    with open(OUT_LOCAL, "w") as f:
        json.dump(out, f)
    print(f"\nsaved -> {OUT_LOCAL}")
    if OUT_GCS:
        write_json_to_gcs(PROJECT_ID, OUT_GCS, BUCKET, out)
        print(f"uploaded -> gs://{BUCKET}/{OUT_GCS}")


if __name__ == "__main__":
    main()

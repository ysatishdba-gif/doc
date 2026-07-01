# ── run one query. retrieval-signals toggle: True = on, False = off ──
out = minimal_pipeline(
    pipeline,
    "elevated psa in last 3 years",
    include_retrieval_signals=True,
)
out

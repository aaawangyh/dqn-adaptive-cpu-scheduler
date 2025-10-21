
# Methodology

We simulate an N‑core CPU with a global ready queue and DVFS levels. State encodes queue stats,
per‑core context, DVFS one‑hot, and a thermal proxy. Actions are small dispatch patterns × DVFS index.
Reward penalizes latency, deadline misses, energy proxy, and context switches.

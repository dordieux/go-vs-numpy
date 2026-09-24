"""Cross-language parity check: for each (op, n), every implementation's
checksum must agree within a relative tolerance.

Usage: python verify.py results/all.jsonl
"""

import json
import sys
from collections import defaultdict

RTOL = 1e-9
# The checksum is a naive sum of n values of magnitude ~1e2; when it lands
# near zero by cancellation, the meaningful comparison scale is n * element
# magnitude * ulp, not the checksum itself. 1e-12 per element is ~4500 ulps
# of 100.0 — loose enough for summation-order differences, still tight
# enough that any real kernel bug blows straight through it.
ATOL_PER_ELEM = 1e-12


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "results/all.jsonl"
    groups: dict[tuple, list] = defaultdict(list)
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            groups[(row["op"], row["n"])].append(row)

    failures = 0
    for (op, n), rows in sorted(groups.items()):
        ref = rows[0]["checksum"]
        for row in rows[1:]:
            got = row["checksum"]
            scale = max(abs(ref), abs(got), 1e-300)
            if abs(got - ref) > RTOL * scale + ATOL_PER_ELEM * n:
                failures += 1
                print(
                    f"MISMATCH {op} n={n}: {rows[0]['lang']}/{rows[0]['impl']}"
                    f"={ref!r} vs {row['lang']}/{row['impl']}={got!r}"
                )
    if failures:
        print(f"{failures} mismatches")
        sys.exit(1)
    print(f"parity OK: {len(groups)} (op, n) groups agree within rtol={RTOL}")


if __name__ == "__main__":
    main()

import math
from typing import *


def parse_chunk_decay(
    pattern: str,
    chunk_count: int,
    snaps_count: int,
    fulls_count: int,
):
    assert fulls_count <= snaps_count
    
    n = snaps_count - fulls_count
    if n <= 0:
        return []

    if pattern.split(":")[0].lower() == "auto":
        _, s = pattern.split(":")
        s = float(s)

        s = max(0.0, min(s, 1.0))
        m = math.ceil(chunk_count * s)
        
        if m <= 0:
            return []
        
        if n == 1:
            return [m]
        
        s = m ** (1.0 / (n - 1))

        outs = [1]
        for i in range(n - 1):
            outs.append(outs[-1] * s)
        outs = [min(math.ceil(x), m) for x in reversed(outs)]
        return outs
    
    outs = []
    for s in pattern.split(","):
        s = s.strip()
        if not s:
            continue
        s = float(s)
        assert 0 < s <= 1, f"invalid chunk decay {s}"
        assert not outs or outs[-1] >= s, f"chunk decay must be decreasing in order"
        outs.append(s)
    
    outs = outs[:n]
    assert len(outs) == n, f"chunk decay must have {n} values"

    outs = [min(math.ceil(chunk_count * s), chunk_count) for s in outs]
    return outs


if __name__ == "__main__":
    print(parse_chunk_decay("0.5, 0.25, 0.125, 0.1, 0.08, 0.07, 0.06", 100, 8, 2))
    print(parse_chunk_decay("auto:0.5", 100, 8, 2))
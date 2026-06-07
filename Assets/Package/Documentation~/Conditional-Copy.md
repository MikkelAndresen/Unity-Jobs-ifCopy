# Conditional Copy (Stage 3)

This is the stage that actually moves bytes. By the time it runs, every parallel iteration has all the information it needs to produce its part of `dst[]` independently and contiguously:

- `indices[index]` — the 64-bit "which elements passed" mask for this batch.
- `counts[index - 1]` — the destination offset where this batch's writes begin.
- `index * 64` — the source offset where this batch's input begins.

Implemented in `ParallelConditionalCopyJob<T, W>` (`ParallelConditionalCopy.cs:230`).

## Three execution paths

Inside `Execute(int index)` (`ParallelConditionalCopy.cs:249`), the dispatch goes like this:

| Condition | Path | Why |
|---|---|---|
| `bitCount == 64` | **Fast** — single `data.Write(dst, src, 64+ext)` | Whole batch passes; one large memcpy is unbeatable. |
| `bitCount` small relative to `runCount` | **`ExecuteSingle`** — gather into stack buffer, single write | Fragmented bitmask means lots of tiny memcpys hurt more than gathering. |
| Otherwise | **`ExecuteBatched`** — one memcpy per run, unrolled by 2 | Few but substantial runs; per-run memcpy wins. |

The dispatch uses the **average run length** as the discriminator. The math:

```csharp
int runCount = math.countbits(n & ~(n << 1));    // number of runs of 1s in n
// avgRun = bitCount / runCount  (avoided; multiply instead)
if (extension == 0 && bitCount < runCount * SingleThreshold)
    ExecuteSingle(...)
else
    ExecuteBatched(...)
```

`SingleThreshold` is a private const (default 3). Tune it per writer — write-combined / GPU destinations should bias it higher because small writes to WC memory are disproportionately expensive.

### Why `n & ~(n << 1)` counts runs

A run starts at bit `k` iff bit `k` is 1 *and* bit `k-1` is 0. `n << 1` shifts the bit pattern up by one position, so `(n << 1)` at position `k` holds the original bit `k-1`. Inverting gives "bit `k-1` was 0". ANDing with `n` keeps only the run-start positions. Popcount → number of runs. One `andn` + one `popcnt`, both single-cycle on modern x86.

## The fast path

```csharp
if (bitCount == 64)
{
    data.Write(dstStartIndex, srcStartIndex, 64 + extension);
    return;
}
```

`data.Write(dstIndex, srcIndex, length)` on `DataRW<T>` is a `UnsafeUtility.MemCpy` (`DataRW.cs:112`). Burst inlines the call and emits the platform's best memcpy — for moderately large `length`, this saturates memory bandwidth.

The `+ extension` term comes from the cross-batch run merging logic, covered below.

## `ExecuteSingle` — gather then write

```csharp
Span<T> temp = stackalloc T[bitCount];   // bitCount is in [1, 63]
int i = 0, t = 0;
while (n != 0)
{
    int tzcnt = math.tzcnt(n);
    t += tzcnt;
    temp[i] = data.Read(srcStartIndex + t + i);
    i++;
    n = (n >> tzcnt) >> 1;   // two-step shift; see below
}
data.Write(dstStartIndex, temp, i);
```

(`ParallelConditionalCopy.cs:330`)

Walk-through using `n = 0b10110`:

| iter | tzcnt | t (after +=) | reads `src[start + t+i]` | n after shift |
|---|---|---|---|---|
| 0 | 1 | 1 | `src[start + 1]` | `0b101` |
| 1 | 0 | 1 | `src[start + 2]` | `0b10` |
| 2 | 1 | 2 | `src[start + 4]` | `0` |

The `t + i` index trick: `t` is the running count of zero bits consumed so far, `i` is the running count of one bits consumed. Their sum is the position in `n` of the bit we're about to read. Cute and branchless.

When `ExecuteSingle` wins:
- Highly fragmented bitmasks (many length-1 runs) — the test case `TestDataType.Odd` is exactly this.
- Write-combined / GPU destinations — one bulk write is much friendlier than many small ones (small writes to WC memory cause partial cache-line flushes).
- Tiny `T` where the per-`data.Write` overhead dominates a tiny memcpy.

When `ExecuteSingle` *cannot* be used:
- When `extension > 0` (cross-batch run merging is in play). `ExecuteSingle` only collects elements from the current batch's stack buffer; it can't fold elements from following batches into its bulk write. The dispatch at the top of `Execute` forces `ExecuteBatched` when `extension > 0`.

## `ExecuteBatched` — per-run memcpys

```csharp
while (n != 0)
{
    // Iteration 1
    int tz1 = math.tzcnt(n);
    t += tz1;
    int rl1 = math.tzcnt(~(n >> tz1));                    // run length
    int wl1 = (t + rl1) >= 64 ? rl1 + extension : rl1;    // include extension iff trailing
    data.Write(dstStartIndex + i, srcStartIndex + t, wl1);
    i += wl1;
    t += rl1;
    n = ((n >> tz1) >> 1) >> (rl1 - 1);

    if (n == 0) break;

    // Iteration 2 — same body
    ...
}
```

(`ParallelConditionalCopy.cs:387`)

For each run of 1-bits in `n`:
1. `tzcnt(n)` skips to the start of the run.
2. `tzcnt(~(n >> tzcnt))` measures the length of the run by counting trailing 1s of the shifted value (which are leading 0s of its inverse, i.e. trailing 0s of `~(n >> tzcnt)`).
3. One memcpy issues for that run.
4. Shift past the run and continue.

The loop is **manually unrolled by 2**: two run iterations per physical loop trip, with a mid-loop `if (n == 0) break;` to handle odd run counts. This halves the loop-control overhead, which matters because most batches have only 2–6 runs total.

### The `wl1 = (t + rl1) >= 64 ? rl1 + extension : rl1` trick

Only the run that reaches bit 63 of the batch picks up the cross-batch extension. The check `(t + rl) >= 64` (where `t` has just been advanced by the trailing-zero count) identifies that run uniquely. When `extension == 0` the branch is dead and Burst either folds it away or replaces it with a cmov; the cost is negligible.

## Cross-batch run merging

This is the trickiest piece of the file. The goal: if a run of passing elements starts in one batch and continues into the next, write all of it in *one* memcpy instead of two separate ones. This is the optimization that makes the dense case faster than a plain memcpy of the whole array on `Half`-like patterns — the runs in stage-3 are large enough to saturate memory bandwidth, and they get coalesced across batch boundaries.

### The ownership invariant

> A run is *owned* by the iteration whose batch contains the run's first set bit. That iteration writes the entire run (including any cross-batch extension). Every other batch the run passes through skips it.

The invariant is maintained by two symmetric local rules:

**Trailing-extend rule** (`ParallelConditionalCopy.cs:271`):
```csharp
if ((n >> 63) != 0 && index + 1 < indices.Length && (indices[index + 1].Value & 1UL) != 0)
    extension = ScanForwardExtension(index);
```
If our trailing run reaches bit 63 *and* the next batch's bit 0 is 1, the run crosses the boundary. `ScanForwardExtension` walks forward through subsequent batches, summing their leading-1 counts, stopping at the first batch whose leading run doesn't span the full 64 bits.

**Leading-skip rule** (`ParallelConditionalCopy.cs:259`):
```csharp
if (index > 0 && (n & 1UL) != 0 && (indices[index - 1].Value >> 63) != 0)
{
    int leadingOnes = math.tzcnt(~n);
    if (leadingOnes >= 64) return;
    n &= ~((1UL << leadingOnes) - 1UL);
    dstStartIndex += leadingOnes;
}
```
If our leading bit is 1 *and* the previous batch's trailing bit is 1, our leading run is the tail of someone else's owned run. Clear those bits from `n` and advance `dstStartIndex` past the slots they'd otherwise have occupied.

The two rules are dual: they evaluate the same boolean predicate at each batch boundary, so adjacent iterations always agree about who owns the run.

### Why this is race-free

For any destination slot in `dst[]`, the run containing it has a unique first-set-bit, hence a unique owning batch. That batch's iteration writes the slot; all other iterations whose batch overlaps the run either skip their leading 1s (per the leading-skip rule) or are an all-passthrough batch in the middle of the chain (whose `n` becomes 0 after the mask and exits without writing anything).

Worked example — chain `A → A+1 (all 1s) → A+2 (leading run of `k`)`:

| Iter | n after mask | dstStart adjustment | Writes |
|---|---|---|---|
| A | (unchanged) | (unchanged) | `[counts[A-1] + offset, counts[A-1] + offset + r_A + 64 + k)` |
| A+1 | 0 | — | nothing (early return) |
| A+2 | `n & ~((1<<k)-1)` | `+= k` | starts at `counts[A+1] + k`, i.e. exactly where A's write ended |

No overlap, no gap, no atomic.

### Why the scan is unbounded

`ScanForwardExtension` (`ParallelConditionalCopy.cs:303`) keeps walking as long as the next batch's leading-1 count is 64 — i.e. as long as the chain of passthroughs continues. There's no fixed cap. In the worst case (a single huge run spanning every batch), one iteration does the entire scan + one large memcpy while every other iteration early-outs. That degrades parallelism for that specific run but is still correct and total-work-optimal.

If pathological "all-1s" input is a real concern, you can cap the scan, but the leading-skip rule would then need to learn about the cap (e.g. via "every Nth batch is a forced boundary that never skips its leading 1s"), or the invariant breaks. The unbounded version was chosen because the typical case has reasonably-bounded run lengths and the bookkeeping for a capped version isn't worth it.

## Bitwise tricks worth remembering

### Two-step shifts (`ParallelConditionalCopy.cs:339`, `ParallelConditionalCopy.cs:398`)

```csharp
n = (n >> tzcnt) >> 1;                          // ExecuteSingle
n = ((n >> tz1) >> 1) >> (rl1 - 1);             // ExecuteBatched
```

C# masks `ulong` shift counts by 63: `n >> 64` evaluates as `n >> 0 == n`, *not* `0`. The natural-looking single-shift `n >>= tzcnt + 1` is therefore wrong when `tzcnt == 63` — it leaves `n` unchanged and the loop runs forever. Splitting the shift into pieces that are each provably < 64 sidesteps the issue without a branch.

(There may have been a latent infinite-loop bug in the original `ExecuteSingle` for any input where the only set bit was bit 63. Burst's LLVM backend has different shift semantics from C# and may have masked it differently, which is probably why it wasn't caught.)

### Clearing the low `k` bits

```csharp
n &= ~((1UL << k) - 1UL);   // clears bits [0, k)
```

`(1 << k) - 1` is the mask of the low `k` bits; its bitwise NOT is the mask of everything else. Guarded by `k < 64` because `1UL << 64` is undefined-ish in C#.

### Counting run starts

```csharp
int runCount = math.countbits(n & ~(n << 1));
```

`(n << 1)` aligns bit `k-1` over bit `k`, so `~(n << 1)` at position `k` is "bit `k-1` was 0". AND with `n` keeps only the run-start positions; popcount gives the run count.

## Why this beats a flat memcpy

A plain `Array.Copy` / `UnsafeUtility.MemCpy` of the full `src` has to read and write every byte. The conditional copy with cross-batch merging reads every byte (it has to inspect them for validation), but on dense input it still issues memcpys that approach the size of contiguous runs in the predicate's pass-set — which for runs longer than a cache-line worth of `T` is bandwidth-equivalent to the flat copy, *but* writes fewer total bytes (only the ones that passed). On extremely dense inputs (`bitCount` near 64 most of the time), the cross-batch merge collapses many batches into a single very large memcpy that the prefetcher and store-buffer handle better than the flat copy's segmented stream.

The flat memcpy is the bound on raw write throughput, but it has no opportunity to write less. The conditional copy gets to write less *while* preserving most of the memcpy's bandwidth profile — which is what makes it competitive (and sometimes faster) on benchmarks.

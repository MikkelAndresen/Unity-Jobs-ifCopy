# Prefix Sum (Stage 2)

After Stage 1, `counts[k]` holds the number of elements that passed in batch `k`. Stage 3 needs to know *where in `dst[]` to start writing* for each batch. That destination offset is the cumulative sum of all prior batches' counts — a prefix sum (a.k.a. inclusive scan).

This stage performs that scan in place over `counts[]`. After it runs, `counts[k]` no longer means "passes in batch `k`"; it means "total passes in batches `0..k` inclusive". Stage 3 then uses `counts[k-1]` as batch `k`'s destination start.

Implemented in `PrefixSum.Schedule` (`ParallelConditionalCopy.cs:113`).

## Why parallelize this at all?

A prefix sum is inherently dependent — `counts[k]` depends on `counts[k-1]`. For small arrays, a sequential scan is the right answer (it's a single tight loop with one add and one store per element). For large arrays, that loop becomes a bottleneck even when the rest of the pipeline is parallel.

`counts.Length == src.Length / 64`. For an `src` of a few million elements, `counts` has tens of thousands of entries — large enough that a parallel scan starts beating the sequential one. So the code picks based on size:

```csharp
int numBlocks = (counts.Length + BlockSize - 1) / BlockSize;
if (numBlocks <= 1) { /* sequential fast path */ }
else                { /* 3-phase parallel scan */ }
```

`BlockSize = 1024` (`ParallelConditionalCopy.cs:111`). That means the fast path covers `counts` up to length 1024, which is `src` up to ~65 K elements. Above that, the parallel scan kicks in.

## The sequential fast path

`SequentialPrefixSumJob` (`ParallelConditionalCopy.cs:154`):

```csharp
int sum = 0;
for (int i = 0; i < counts.Length; i++)
{
    sum += counts[i];
    counts[i] = sum;
}
totalCount.Value = sum;
```

Inclusive scan, side-effects `totalCount` with the grand total. Nothing surprising.

## The 3-phase parallel scan

This is the standard "scan-by-blocks" algorithm. Three jobs, executed in sequence:

```
counts: [c0 c1 c2 ... c1023 | c1024 c1025 ... c2047 | ... ]
         └───── block 0 ────┘└───── block 1 ──────┘
```

### Phase 1 — Parallel partial scans (`ParallelPartialPrefixSumJob`, line 172)

Each block gets one parallel iteration. The iteration does a sequential scan *within* the block, treating block-local positions, and writes the block's grand total to `blockTotals[blockIndex]`.

After this phase:
- `counts[0..1023]` holds the prefix sums for block 0 (already globally correct because block 0 starts at zero).
- `counts[1024..2047]` holds prefix sums for block 1, but starting from zero — they need to be shifted by `blockTotals[0]` to be globally correct.
- And so on.
- `blockTotals[k]` holds the sum of block `k` (i.e., `counts[k*BlockSize + BlockSize - 1]` *before* it gets globalized — equivalent to the partial scan's last value).

### Phase 2 — Sequential scan over block totals (`BlockPrefixSumJob`, line 193)

A single sequential job that runs an inclusive scan over `blockTotals[]` and writes the grand total to `totalCount`. After this:
- `blockTotals[k]` holds the cumulative total of blocks `0..k` inclusive.
- `totalCount` is the grand total.

`blockTotals.Length == numBlocks`, which is `counts.Length / 1024`. For million-element `counts` that's only a thousand entries — fast to scan sequentially.

### Phase 3 — Parallel finalize (`FinalizePrefixSumJob`, line 211)

Each iteration takes one block (skipping block 0, which is already globally correct) and adds the cumulative offset of the *previous* blocks to every element in its block:

```csharp
int offset = blockTotals[blockIndex];   // cumulative total of blocks 0..blockIndex
int start = (blockIndex + 1) * BlockSize;
int end = math.min(start + BlockSize, counts.Length);
for (int i = start; i < end; i++)
    counts[i] += offset;
```

Note the indexing: `blockIndex` in this job corresponds to block `(blockIndex + 1)` in the original array. The loop is scheduled with `length = numBlocks - 1`. `blockTotals[blockIndex]` is exactly what block `(blockIndex + 1)` needs added — the cumulative total of all blocks *before* it.

After this phase, `counts[]` holds the global inclusive prefix sum.

## Allocation lifetime

`blockTotals` is allocated as `TempJob` inside `Schedule` and disposed via `blockTotals.Dispose(handle)` so the disposal is scheduled to run after the finalize job completes. No manual cleanup is needed.

## Why inclusive (not exclusive)?

Stage 3 reads `counts[index - 1]` as batch `index`'s start, with the special case `index == 0 → 0`. That makes it an exclusive prefix sum at the call site, derived from the inclusive sum stored in `counts[]`. We could store exclusive instead, but inclusive gives us `totalCount` for free (it's `counts[counts.Length - 1]`) and avoids an off-by-one in the scan loop.

## Trade-off summary

| | Sequential | 3-phase parallel |
|---|---|---|
| Jobs scheduled | 1 | 3 |
| Scratch allocation | 0 | 1× `int[numBlocks]` |
| Work | O(N) | O(N) total, ~O(N/blocks) wall-clock |
| Wins below | ~1024 counts (~65 K src) | above that |

`BlockSize = 1024` is a tuned constant — small enough that lots of blocks exist on million-element inputs, large enough that the per-block setup is amortized.

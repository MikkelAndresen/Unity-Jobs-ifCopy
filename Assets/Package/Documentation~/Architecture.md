# Architecture

## What this package does

Given a source array `src` of `T` and a predicate `V : IBatchValidator<T>`, copy every element that passes the predicate into a destination array `dst`, contiguously and in source order. This is *stream compaction* (sometimes called "filter" or "if-copy"). The package does it in parallel with Burst-compiled jobs and no atomics on the hot path.

## Why this problem is hard in parallel

The sequential version is trivial:

```csharp
int w = 0;
for (int i = 0; i < src.Length; i++)
    if (validator.Validate(i, src[i])) dst[w++] = src[i];
```

The hard part for a parallel version is the shared `w` cursor. Naive approaches either:
- Use an atomic increment per pass → serializes on the cache line, kills throughput.
- Allocate per-thread temporary lists and concatenate → unpredictable memory traffic, ordering hazards.

Both also lose source ordering, which we want to preserve.

The trick this package uses is to **split the problem into a metadata pass and a write pass**, with a prefix sum in between. After the metadata pass every parallel writer knows *exactly* where its output goes — no atomics, no contention, source order preserved.

## The three-stage pipeline

```
src[0..N]                     (input)
   │
   │  Stage 1: ParallelIndexingSumJob  (one iteration per 64 elements, parallel)
   ▼
indices[0..N/64]              (BitField64 per 64-element batch — bit i = "src[batch*64+i] passes")
counts[0..N/64]               (popcount per batch — how many elements pass in each batch)
   │
   │  Stage 2: PrefixSum  (3-phase parallel prefix scan, or sequential for small N)
   ▼
counts[0..N/64]               (now holds prefix sums — counts[i] = total passes in batches 0..i)
totalCount                    (total number of elements that passed)
   │
   │  Stage 3: ParallelConditionalCopyJob  (one iteration per batch, parallel)
   ▼
dst[0..totalCount]            (output, packed in source order)
```

Each stage is a Burst-compiled job (`Assets/Package/Runtime/ParallelConditionalCopy.cs`). The schedule wiring lives in `NativeCollectionExtensions.IfCopyToParallel` (`Assets/Package/Runtime/NativeCollectionExtensions.cs:131`).

### Stage 1 — Indexing (`ParallelIndexingSumJob`)

Each parallel iteration takes one 64-element slice of `src`, calls `IBatchValidator<T>.Validate(slice)` to get a `BitField64`, popcounts it, and writes both results out. The remainder (when `N % 64 != 0`) is handled by a single `IJob` afterwards (`ParallelConditionalCopy.cs:50`).

Output: `indices[]` (one BitField64 per batch) and `counts[]` (popcount per batch).

See [Bitfield-Indexing.md](Bitfield-Indexing.md).

### Stage 2 — Prefix Sum (`PrefixSum`)

Converts `counts[]` from "passes per batch" into "destination offsets per batch". After this stage, `counts[i-1]` is the destination index at which batch `i`'s output begins.

For small inputs (`numBlocks <= 1`) this is a single sequential scan job. For larger inputs it's a 3-phase parallel algorithm: per-block partial scans, a sequential scan over block totals, and a parallel finalize pass that adds block offsets back.

See [Prefix-Sum.md](Prefix-Sum.md).

### Stage 3 — Copy (`ParallelConditionalCopyJob`)

Each parallel iteration reads its batch's `BitField64`, knows its destination offset (from `counts[]`), and writes the passing elements. The job has three internal execution paths and merges runs across batch boundaries to minimize the number of memcpy calls.

See [Conditional-Copy.md](Conditional-Copy.md).

## The design tradeoff

Stages 1 and 2 are pure overhead — they don't move any actual element data, just metadata. Stage 3 is where real bytes get copied. The bet is that the metadata is so cheap (popcounts, bitfields, integer scans) and so cache-friendly that the savings in Stage 3 — no atomics, perfectly contiguous writes, large memcpys — pay for the metadata several times over.

Empirically, for the test cases in `CopyTestBehaviour.cs`, this beats a plain `memcpy` of the whole array on dense inputs, and beats list-based filters by a wide margin on sparse-to-mid inputs. The optimizations in Stage 3 (Single vs Batched dispatch, cross-batch run merging) are what make the "dense" case competitive with raw memcpy.

## Entry points

Most users should call the extension method (`Assets/Package/Runtime/NativeCollectionExtensions.cs`):

```csharp
JobHandle handle = src.IfCopyToParallel<T, MyValidator>(dst, out var counter);
handle.Complete();
int actuallyCopied = counter.Value;
counter.Dispose();
```

If you call repeatedly with the same `src`/`dst` shape, allocate `indices` and `counts` once and pass them in to avoid per-call `TempJob` allocations — there's an overload for that, plus a `CopyHandler<T, V>` wrapper at `NativeCollectionExtensions.cs:17` that owns the scratch.

There's also a `NativeList<T>` overload at `NativeCollectionExtensions.cs:181` that resizes the destination list to the actually-copied count at the end of the chain.

## Key types

| Type | Role |
|---|---|
| `IBatchValidator<T>` | Validates 64 elements at a time → `BitField64`. The whole pipeline keys off this. |
| `IIndexWriter<T>` / `IIndexReader<T>` | Abstracts the dst (and src) so the copy job can target a `NativeArray<T>`, a GPU `ComputeBuffer` (via `BeginWrite`), or anything else. |
| `DataRW<T>` | The standard implementation of those interfaces — backs both src and dst with `NativeArray<T>` plus raw pointers for fast unmanaged memcpy. |
| `ParallelConditionalCopyJob<T, W>` | Stage 3. Generic over both `T` and the writer `W`, so Burst can monomorphize and inline. |

## Files at a glance

```
Assets/Package/Runtime/
  ParallelConditionalCopy.cs       Stages 1, 2, 3 (jobs)
  NativeCollectionExtensions.cs    Schedule wiring + CopyHandler
  DataRW.cs                        IIndexWriter<T>/IIndexReader<T> implementation
  CopyJobInterfaces.cs             Validator + writer/reader interfaces
```

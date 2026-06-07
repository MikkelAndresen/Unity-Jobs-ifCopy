# Bitfield Indexing (Stage 1)

This stage turns `src[]` into two parallel arrays of metadata:

- `indices[]` — one `BitField64` per 64 source elements. Bit `i` is set iff `src[batch*64 + i]` passes the validator.
- `counts[]` — one `int` per batch, holding the popcount of that batch's `BitField64`.

Implemented in `ParallelIndexingSumJob` (`ParallelConditionalCopy.cs:16`).

## Why a bitfield instead of a list of indices?

The obvious alternative is "produce a `NativeList<int>` of passing indices." That works but has worse properties:

1. **Atomic-free output.** A `NativeList<int>` would need either an atomic counter (per-element contention) or per-thread sublists (concatenation work). A bitfield is written at a fixed offset per batch — no contention at all.
2. **Dense metadata.** 64 elements of metadata fit in 8 bytes. A list of `int` indices would be 4× larger in the dense case and unbounded in length, hurting cache locality in Stage 3.
3. **Decodable in a tight loop.** `tzcnt` lets the copy job walk just the set bits in a few instructions per bit, with no random-access reads of an "indices" list.
4. **Friendly to batch validators.** Many useful predicates (range checks, sign checks, flag tests) can be evaluated on a SIMD lane of 4–16 elements at a time, producing a small bitmask that gets merged into the BitField64 directly. `IBatchValidator<T>.Validate(in NativeSlice<T>)` exposes this.

The cost is some bookkeeping (Stage 2) to turn batch popcounts into write offsets, but that's a pass over `N/64` ints — much cheaper than the alternative dst-write contention.

## The validator interface

```csharp
public interface IBatchValidator<T> : IValidator<T> where T : unmanaged
{
    BitField64 Validate(in NativeSlice<T> elements);  // elements.Length == 64
}
```

The slice is always 64 elements long for the parallel path. The remainder path (below) uses the simpler per-element `IValidator<T>.Validate(int, T)` for the tail.

Example implementation (from `CopyTestBehaviour.cs:201`):

```csharp
public BitField64 Validate(in NativeSlice<float3x4> elements)
{
    var bits = new BitField64();
    Hint.Assume(elements.Length == 64);   // lets Burst unroll/vectorize
    for (int i = 0; i < 64; i++)
        bits.SetBits(i, elements[i].c0.x > 0);
    return bits;
}
```

The `Hint.Assume(elements.Length == 64)` is load-bearing: Burst uses it to drop bounds checks and to vectorize the inner loop when `T` is small enough.

## The parallel iteration

```csharp
[SkipLocalsInit]
public void Execute(int index)
{
    int dataIndex = index * 64;
    Hint.Assume(src.Length >= dataIndex + 64);   // bounds hint — see below
    var slice = src.Slice(dataIndex, 64);
    var bits = del.Validate(slice);
    counts[index] = math.countbits(bits.Value);
    indices[index] = bits;
}
```

(`ParallelConditionalCopy.cs:34`)

A few non-obvious choices:

- **`src.Length >= dataIndex + 64` hint.** The job is scheduled with `length = src.Length / 64` (integer division — see `Schedule` at line 82), so this assertion is mathematically true for every iteration. The hint communicates that to Burst so it can elide the bounds check inside `src.Slice(...)`.
- **`SkipLocalsInit`.** Skips the C# default-zero-init of locals. Cheap, but worth it on a job called once per 64 elements.
- **Popcount written separately.** We write both the bitfield and its popcount. The popcount is later overwritten by the prefix sum (Stage 2 mutates `counts[]` in place). We could compute the popcount lazily in Stage 3 instead, but doing it here:
  1. Keeps it in cache (the bitfield is already in a register).
  2. Gives the prefix sum a contiguous array to scan without touching `indices[]`.

## The remainder

If `src.Length % 64 != 0`, the last batch is incomplete. `ParallelIndexingSumJob.Schedule` (`ParallelConditionalCopy.cs:72`) schedules the main parallel pass over `src.Length / 64` full batches, then a single `RemainderValidationJob` (`ParallelConditionalCopy.cs:50`) for the tail:

```csharp
public void Execute()
{
    int remainderCount = src.Length % 64;
    int dataStartIndex = src.Length - remainderCount;
    var bits = new BitField64();
    for (int i = 0; i < remainderCount; i++)
        bits.SetBits(i, del.Validate(dataStartIndex + i, src[dataStartIndex + i]));
    counts[indices.Length - 1] = math.countbits(bits.Value);
    indices[indices.Length - 1] = bits;
}
```

Key points:

- The remainder bitfield only has its low `remainderCount` bits potentially set; the high bits are zero. This Just Works for Stage 3 — those zero bits look like "elements that didn't pass" and get skipped.
- It uses the per-element validator (`IValidator<T>.Validate(int, T)`), not the batch one, because the slice is partial and most batch validators assume a full 64.
- The schedule attaches it as a dependency of the prefix sum, so the tail bitfield is in place before any offset math.

## What Stage 1 hands off

After this stage:
- `indices[k]` describes which of the elements `src[k*64 .. k*64+63]` pass.
- `counts[k]` says how many of them pass (popcount of `indices[k]`).

`counts[]` is the input to Stage 2.

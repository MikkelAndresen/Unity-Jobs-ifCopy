using System;
using Unity.Burst;
using Unity.Burst.CompilerServices;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

/// <summary>
/// This job is meant to pack booleans into <see cref="indices"/>.
/// Then you can use <see cref="ParallelConditionalCopyJob{T,W}"/> to write to a destination array based on the <see cref="indices"/> array.
/// It also counts the bits set and assigns them to <see cref="counts"/>.
/// </summary>
/// <typeparam name="T"></typeparam>
/// <typeparam name="V"></typeparam>
[BurstCompile(CompileSynchronously = true, OptimizeFor = OptimizeFor.Performance), GenerateTestsForBurstCompatibility]
public struct ParallelIndexingSumJob<T, V> : IJobParallelFor, IConditionalIndexingJob<T, V>
	where T : unmanaged where V : IBatchValidator<T>
{
	[ReadOnly] public V del;
	[ReadOnly] public NativeArray<T> src;
	[WriteOnly] public NativeArray<BitField64> indices;

	[WriteOnly] public NativeArray<int> counts;

	public ParallelIndexingSumJob(NativeArray<T> src, NativeArray<BitField64> indices, NativeArray<int> counts,
		V del = default)
	{
		this.src = src;
		this.indices = indices;
		this.counts = counts;
		this.del = del;
	}

	[SkipLocalsInit]
	public void Execute(int index)
	{
		int dataIndex = index * 64;
		Hint.Assume(src.Length >= dataIndex + 64); // bounds hint

		var slice = src.Slice(dataIndex, 64);
		var bits = del.Validate(slice);
		counts[index] = math.countbits(bits.Value);
		indices[index] = bits;
	}

	/// <summary>
	/// Validates the remainder elements that don't fill a complete 64-bit batch.
	/// The prefix sum is handled separately by <see cref="PrefixSum"/>.
	/// </summary>
	[BurstCompile(CompileSynchronously = true, OptimizeFor = OptimizeFor.Performance), GenerateTestsForBurstCompatibility]
	private struct RemainderValidationJob : IJob
	{
		[ReadOnly] public V del;
		[ReadOnly] public NativeArray<T> src;
		[WriteOnly] public NativeArray<BitField64> indices;
		[NativeDisableParallelForRestriction] public NativeArray<int> counts;

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
	}

	public static JobHandle Schedule(
		NativeArray<T> src,
		NativeArray<BitField64> indices,
		NativeArray<int> counts,
		NativeReference<int> totalCount,
		int innerBatchCount = 10,
		JobHandle dependsOn = default,
		V del = default)
	{
		int remainder = src.Length % 64;
		int length = src.Length / 64;
		var handle =
			new ParallelIndexingSumJob<T, V>(src, indices, counts, del).Schedule(length, innerBatchCount, dependsOn);

		if (remainder > 0)
			handle = new RemainderValidationJob
			{
				src = src,
				indices = indices,
				counts = counts,
				del = del,
			}.Schedule(handle);

		handle = PrefixSum.Schedule(counts, totalCount, handle);

		return handle;
	}
}

/// <summary>
/// Parallel prefix sum over an integer array.
/// For small arrays (single block), uses a simple sequential scan.
/// For larger arrays, uses a 3-phase parallel algorithm:
/// 1. Partial prefix sums within each block (parallel)
/// 2. Sequential prefix sum over block totals
/// 3. Add block offsets to finalize global prefix sums (parallel)
/// </summary>
public static class PrefixSum
{
	private const int BlockSize = 1024;

	public static JobHandle Schedule(NativeArray<int> counts, NativeReference<int> totalCount, JobHandle dependsOn)
	{
		int numBlocks = (counts.Length + BlockSize - 1) / BlockSize;

		if (numBlocks <= 1)
		{
			return new SequentialPrefixSumJob
			{
				counts = counts,
				totalCount = totalCount,
			}.Schedule(dependsOn);
		}

		var blockTotals = new NativeArray<int>(numBlocks, Allocator.TempJob);

		// Partial prefix sums within each block
		var handle = new ParallelPartialPrefixSumJob
		{
			counts = counts,
			blockTotals = blockTotals,
		}.Schedule(numBlocks, 1, dependsOn);

		// Sequential prefix sum over block totals
		handle = new BlockPrefixSumJob
		{
			blockTotals = blockTotals,
			totalCount = totalCount,
		}.Schedule(handle);

		// Add block offsets to all elements except block 0
		handle = new FinalizePrefixSumJob
		{
			counts = counts,
			blockTotals = blockTotals,
		}.Schedule(numBlocks - 1, 1, handle);

		blockTotals.Dispose(handle);
		return handle;
	}

	[BurstCompile(CompileSynchronously = true, OptimizeFor = OptimizeFor.Performance)]
	private struct SequentialPrefixSumJob : IJob
	{
		public NativeArray<int> counts;
		public NativeReference<int> totalCount;

		public void Execute()
		{
			int sum = 0;
			for (int i = 0; i < counts.Length; i++)
			{
				sum += counts[i];
				counts[i] = sum;
			}
			totalCount.Value = sum;
		}
	}

	[BurstCompile(CompileSynchronously = true, OptimizeFor = OptimizeFor.Performance)]
	private struct ParallelPartialPrefixSumJob : IJobParallelFor
	{
		[NativeDisableParallelForRestriction] public NativeArray<int> counts;
		[WriteOnly] public NativeArray<int> blockTotals;

		public void Execute(int blockIndex)
		{
			int start = blockIndex * BlockSize;
			int end = math.min(start + BlockSize, counts.Length);

			int sum = 0;
			for (int i = start; i < end; i++)
			{
				sum += counts[i];
				counts[i] = sum;
			}
			blockTotals[blockIndex] = sum;
		}
	}

	[BurstCompile(CompileSynchronously = true, OptimizeFor = OptimizeFor.Performance)]
	private struct BlockPrefixSumJob : IJob
	{
		public NativeArray<int> blockTotals;
		public NativeReference<int> totalCount;

		public void Execute()
		{
			int sum = 0;
			for (int i = 0; i < blockTotals.Length; i++)
			{
				sum += blockTotals[i];
				blockTotals[i] = sum;
			}
			totalCount.Value = sum;
		}
	}

	[BurstCompile(CompileSynchronously = true, OptimizeFor = OptimizeFor.Performance)]
	private struct FinalizePrefixSumJob : IJobParallelFor
	{
		[NativeDisableParallelForRestriction] public NativeArray<int> counts;
		[ReadOnly] public NativeArray<int> blockTotals;

		public void Execute(int blockIndex)
		{
			// blockIndex is 0-based but represents block (blockIndex + 1) since block 0 is skipped
			int offset = blockTotals[blockIndex]; // cumulative total of blocks 0..blockIndex
			int start = (blockIndex + 1) * BlockSize;
			int end = math.min(start + BlockSize, counts.Length);

			for (int i = start; i < end; i++)
				counts[i] += offset;
		}
	}
}

[BurstCompile(CompileSynchronously = true, OptimizeFor = OptimizeFor.Performance), GenerateTestsForBurstCompatibility]
public struct ParallelConditionalCopyJob<T, W> : IJobParallelFor, IConditionalCopyJob<T, W> where T : unmanaged
	where W : struct, IIndexWriter<T>, IIndexReader<T>
{
	// Average run length below which ExecuteSingle (gather-then-bulk-write) is preferred
	// over ExecuteBatched (per-run memcpy). Bias higher for write-combined / GPU writers.
	private const int SingleThreshold = 3;

	public W data;
	[ReadOnly] public NativeArray<int> counts;
	[ReadOnly] private NativeArray<BitField64> indices;

	public ParallelConditionalCopyJob(
		W data,
		NativeArray<BitField64> indices,
		NativeArray<int> counts)
	{
		this.data = data;
		this.counts = counts;
		this.indices = indices;
	}

	// Terminology: a "run" is a maximal contiguous span of set bits in the mask,
	// which corresponds to a contiguous span of source elements to copy. Runs can
	// straddle 64-bit batch boundaries; the cross-batch handoff logic below
	// assigns each straddling run to the batch that owns its first bit.
	[SkipLocalsInit]
	public void Execute(int index)
	{
		Hint.Assume(counts.Length > 0);
		Hint.Assume(indices.Length > 0);

		ulong n = indices[index].Value;
		int srcStartIndex = index * 64;
		int dstStartIndex = index == 0 ? 0 : counts[index - 1];

		// Cross-batch handoff (leading side): if our leading 1s continue a run owned by the
		// previous batch, that batch writes them as part of its extension. Skip them here.
		if (CurBatchWrittenByPrevRun(in indices))
		{
			int leadingOnes = math.tzcnt(~n);
			if (leadingOnes >= 64) return; // entire batch consumed by continuation
			n &= ~((1UL << leadingOnes) - 1UL);
			dstStartIndex += leadingOnes;
		}

		if (n == 0) return;

		int bitCount = math.countbits(n);

		// Cross-batch handoff (trailing side): if our trailing run reaches bit 63 and the
		// next batch starts with a 1, we own a cross-batch run — compute the extension.
		int extension = 0;
		if (ShouldExtendCurBatch(in indices))
			extension = ScanForwardExtension(index);

		// Fast path: entire (remaining) batch is contiguous 1s.
		if (bitCount == 64)
		{
			data.Write(dstStartIndex, srcStartIndex, 64 + extension);
			return;
		}

		// Pick path by average run length. n & ~(n << 1) isolates run-start bits; popcount
		// of that = number of runs. ExecuteSingle is forced off when extension > 0 because
		// it can't carry a write into the following batches.
		int runCount = math.countbits(n & ~(n << 1));

		if (extension == 0 && bitCount < runCount * SingleThreshold)
			ExecuteSingle(n, bitCount, dstStartIndex, srcStartIndex);
		else
			ExecuteBatched(n, dstStartIndex, srcStartIndex, extension);

		bool CurBatchWrittenByPrevRun(in NativeArray<BitField64> batches)
		{
			return index > 0 &&
			       (n & 1UL) != 0 && // First bit of current batch is set
			       (batches[index - 1].Value >> 63) != 0; // Last bit of previous batch is set
		}

		bool ShouldExtendCurBatch(in NativeArray<BitField64> batches)
		{
			return (n >> 63) != 0 && // Last bit of current batch is set
			       index + 1 < batches.Length && // Not last batch
			       (batches[index + 1].Value & 1UL) != 0; // First bit of next batch set
		}
	}

	private int ScanForwardExtension(int fromIndex)
	{
		int total = 0;
		for (int i = fromIndex + 1; i < indices.Length; i++)
		{
			ulong next = indices[i].Value;
			if ((next & 1UL) == 0) break;
			int leading = math.tzcnt(~next);
			total += leading;
			if (leading < 64) break;
		}
		return total;
	}

	[SkipLocalsInit]
	private unsafe void ExecuteSingle(ulong n, int bitCount, int dstStartIndex, int srcStartIndex)
	{
		Hint.Assume(bitCount > 0);
		Hint.Assume(dstStartIndex >= 0);
		Hint.Assume(srcStartIndex >= 0);

		Span<T> temp = stackalloc T[bitCount];

#if UNITY_BURST_EXPERIMENTAL_PREFETCH_INTRINSIC
		data.PrefetchSrc(srcStartIndex + bitCount);
		data.PrefetchDst(dstStartIndex + bitCount);
#endif

		int i = 0;
		int t = 0;
		while (n != 0)
		{
			int tzcnt = math.tzcnt(n);
			t += tzcnt;
			temp[i] = data.Read(srcStartIndex + t + i);

			i++;
			// Two-step shift avoids C#'s shift-mask edge: `n >> 64` would become `n >> 0`
			// when tzcnt == 63, causing an infinite loop.
			n = (n >> tzcnt) >> 1;
		}

		data.Write(dstStartIndex, temp, i);
	}

	[SkipLocalsInit]
	public void ExecuteBatched(int index)
	{
		Hint.Assume(counts.Length > 0);
		Hint.Assume(indices.Length > 0);

		ulong n = indices[index].Value;
		if (n == 0) return;
		int dstStartIndex = index == 0 ? 0 : counts[index - 1];
		int srcStartIndex = index * 64;

		int extension = 0;
		if ((n >> 63) != 0 && index + 1 < indices.Length && (indices[index + 1].Value & 1UL) != 0)
			extension = ScanForwardExtension(index);

		ExecuteBatched(n, dstStartIndex, srcStartIndex, extension);
	}

	[SkipLocalsInit]
	private void ExecuteBatched(ulong n, int dstStartIndex, int srcStartIndex, int extension)
	{
		Hint.Assume(dstStartIndex >= 0);
		Hint.Assume(srcStartIndex >= 0);

#if UNITY_BURST_EXPERIMENTAL_PREFETCH_INTRINSIC
		data.PrefetchSrc(srcStartIndex + 64 + extension);
		data.PrefetchDst(dstStartIndex + 64 + extension);
#endif

		// Run-length loop, unrolled by 2 to halve loop-control overhead. Only the run that
		// reaches bit 63 picks up the cross-batch extension; for all other runs (and when
		// extension == 0) the branch resolves to the cheap fall-through.
		int i = 0;
		int t = 0;
		while (n != 0)
		{
			// Iteration 1
			int tz1 = math.tzcnt(n);
			t += tz1;
			int rl1 = math.tzcnt(~(n >> tz1));
			int wl1 = (t + rl1) >= 64 ? rl1 + extension : rl1;
			data.Write(dstStartIndex + i, srcStartIndex + t, wl1);
			i += wl1;
			t += rl1;
			// Two-step shift keeps each shift count < 64 — branchless replacement for the
			// `shift >= 64 ? 0 : n >> shift` guard.
			n = ((n >> tz1) >> 1) >> (rl1 - 1);

			if (n == 0) break;

			// Iteration 2
			int tz2 = math.tzcnt(n);
			t += tz2;
			int rl2 = math.tzcnt(~(n >> tz2));
			int wl2 = (t + rl2) >= 64 ? rl2 + extension : rl2;
			data.Write(dstStartIndex + i, srcStartIndex + t, wl2);
			i += wl2;
			t += rl2;
			n = ((n >> tz2) >> 1) >> (rl2 - 1);
		}
	}
}

[BurstCompile(CompileSynchronously = true, OptimizeFor = OptimizeFor.Performance)]
public struct ConditionalCopyJob<T, V> : IJobParallelFor where T : unmanaged where V : IValidator<T>
{
	[ReadOnly] public NativeArray<T> src;

	public NativeList<T>.ParallelWriter dst;
	public V Validator;

	public unsafe void Execute(int index)
	{
		Hint.Assume(src.Length > 0);
		Hint.Assume(dst.ListData->Capacity > index);
		var value = src[index];

		// Check condition
		if (Validator.Validate(index, value))
			dst.AddNoResize(value);
	}
}

public static class FilterCopy<T, V> where T : unmanaged where V : unmanaged, IValidator<T>
{
	public static JobHandle Schedule(NativeArray<T> src, NativeList<T> dst, NativeList<int> indices,
		int innerLoopBatchCount, JobHandle dependsOn = default)
	{
		var handle = new FilterJob { src = src, Validator = default }.ScheduleAppend(indices, src.Length, dependsOn);
		handle = new UpdateListLengthJob { list = dst, indices = indices }.Schedule(handle);
		handle = new IndicesCopyJob { src = src, indices = indices, dst = dst }.Schedule(indices, innerLoopBatchCount, handle);
		return handle;
	}

	[BurstCompile(CompileSynchronously = true, OptimizeFor = OptimizeFor.Performance)]
	private struct IndicesCopyJob : IJobParallelForDefer
	{
		[ReadOnly] public NativeArray<T> src;
		[ReadOnly] public NativeList<int> indices;

		[WriteOnly, NativeDisableParallelForRestriction]
		public NativeList<T> dst;

		public void Execute(int index)
		{
			Hint.Assume(src.Length > 0);
			Hint.Assume(indices.Length > 0);
			int srcIndex = indices[index];
			Hint.Assume(srcIndex >= 0);

			dst[index] = src[srcIndex];
		}
	}

	[BurstCompile(CompileSynchronously = true, OptimizeFor = OptimizeFor.Performance)]
	private struct UpdateListLengthJob : IJob
	{
		[WriteOnly] public NativeList<T> list;
		[ReadOnly] public NativeList<int> indices;

		public void Execute() => list.Length = indices.Length;
	}

	[BurstCompile(CompileSynchronously = true, OptimizeFor = OptimizeFor.Performance)]
	private struct FilterJob : IJobFilter
	{
		[ReadOnly] public NativeArray<T> src;
		public V Validator;

		public bool Execute(int index) => Validator.Validate(index, src[index]);
	}
}

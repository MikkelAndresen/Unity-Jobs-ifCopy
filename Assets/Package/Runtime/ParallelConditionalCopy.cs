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

		// Step 1: Partial prefix sums within each block
		var handle = new ParallelPartialPrefixSumJob
		{
			counts = counts,
			blockTotals = blockTotals,
		}.Schedule(numBlocks, 1, dependsOn);

		// Step 2: Sequential prefix sum over block totals
		handle = new BlockPrefixSumJob
		{
			blockTotals = blockTotals,
			totalCount = totalCount,
		}.Schedule(handle);

		// Step 3: Add block offsets to all elements except block 0
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

	[SkipLocalsInit]
	public void Execute(int index)
	{
		ulong n = indices[index].Value;
		var bitCount = math.countbits(n);

		if (bitCount == 0) return;

		// Fast path: entire batch passes — single contiguous memcpy
		if (bitCount == 64)
		{
			int dstStart = index == 0 ? 0 : counts[index - 1];
			data.Write(dstStart, index * 64, 64);
			return;
		}

		ExecuteSingle(index);
	}

	[SkipLocalsInit]
	public unsafe void ExecuteSingle(int index)
	{
		Hint.Assume(counts.Length > 0);
		Hint.Assume(indices.Length > 0);

		// We need to start write index of the src data which we can get from counts
		int dstStartIndex = index == 0 ? 0 : counts[index - 1];
		int srcStartIndex = index * 64;
		Hint.Assume(dstStartIndex >= 0);
		Hint.Assume(srcStartIndex >= 0);

		ulong n = indices[index].Value;
		var bitCount = math.countbits(n);
		if (bitCount == 0)
			return;
		Hint.Assume(bitCount > 0);

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
			n >>= tzcnt + 1;
		}

		data.Write(dstStartIndex, temp, i);
	}

	[SkipLocalsInit]
	public void ExecuteBatched(int index)
	{
		Hint.Assume(counts.Length > 0);
		Hint.Assume(indices.Length > 0);

		// We need to start write index of the src data which we can get from counts
		int dstStartIndex = index == 0 ? 0 : counts[index - 1];
		int srcStartIndex = index * 64;
		Hint.Assume(dstStartIndex >= 0);
		Hint.Assume(srcStartIndex >= 0);

		ulong n = indices[index].Value;
		var bitCount = math.countbits(n);
		if (bitCount == 0)
			return;
		Hint.Assume(bitCount > 0);

#if UNITY_BURST_EXPERIMENTAL_PREFETCH_INTRINSIC
		data.PrefetchSrc(srcStartIndex + bitCount);
		data.PrefetchDst(dstStartIndex + bitCount);
#endif
		// Batched run-length loop: copies consecutive set-bit runs in a single call
		int i = 0;
		int t = 0;
		while (n != 0)
		{
			int tzcnt = math.tzcnt(n);
			t += tzcnt;
			int runLength = math.tzcnt(~(n >> tzcnt));

			data.Write(dstStartIndex + i, srcStartIndex + t, runLength);

			i += runLength;
			t += runLength;
			int shift = tzcnt + runLength;
			n = shift >= 64 ? 0 : n >> shift;
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

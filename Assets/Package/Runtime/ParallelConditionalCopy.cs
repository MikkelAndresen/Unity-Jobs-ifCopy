using System;
using Unity.Burst;
using Unity.Burst.CompilerServices;
using Unity.Burst.Intrinsics;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

/// <summary>
/// This job is meant to pack booleans into <see cref="indices"/>.
/// Then you can use <see cref="ParallelConditionalCopyJob{T,W}"/> to write to a destination array based on the <see cref="indices"/> array.
/// It also counts the bits set and assigns them to <see cref="counts"/>.
/// </summary>
/// <typeparam name="T"></typeparam>
/// <typeparam name="V"></typeparam>
[BurstCompile(CompileSynchronously = true), GenerateTestsForBurstCompatibility]
public struct ParallelIndexingSumJob<T, V> : IJobParallelFor, IConditionalIndexingJob<T, V>
	where T : unmanaged where V : IValidator<T>
{
	[ReadOnly] public V del;
	[ReadOnly] public NativeArray<T> src;
	[WriteOnly] public NativeArray<BitField64> indices;

	[WriteOnly] public NativeArray<int> counts;
	// private static readonly ProfilerMarker conditionIndexingSumJobMarker = new ProfilerMarker(nameof(ConditionIndexingSumJob<T, M>));

	public ParallelIndexingSumJob(NativeArray<T> src, NativeArray<BitField64> indices, NativeArray<int> counts,
		V del = default)
	{
		this.src = src;
		this.indices = indices;
		this.counts = counts;
		this.del = del;
	}

	public void Execute(int index)
	{
		//conditionIndexingSumJobMarker.Begin();

		BitField64 bits = new BitField64(0);
		int dataIndex = index * 64;

		for (int i = 0; i < 64; i++)
		{
			bool v = del.Validate(dataIndex + i, src[dataIndex + i]);
			// This one seems to generate less instructions, but not vectorized. The performance was the same as the line below however.
			bits.SetBits(i, v);
			// This generates more vectorized instructions with the same performance, I'm guessing the power cost is higher for this line though.
			// bits.Value |= (del.Validate(src[dataIndex + i]) ? 1ul : 0ul) << i;
		}

		counts[index] = math.countbits(bits.Value);
		indices[index] = bits;

		//conditionIndexingSumJobMarker.End();
	}

	// private static readonly ProfilerMarker remainderJobMarker = new ProfilerMarker(nameof(RemainderSumJob));
	/// <summary>
	/// This job sets the bits and sums the last element of <see cref="indices"/>.
	/// It also will count all the bits at the end and store the count so far in <see cref="counts"/>.
	/// </summary>
	[BurstCompile(CompileSynchronously = true), GenerateTestsForBurstCompatibility]
	private struct RemainderSumJob : IJob
	{
		[ReadOnly] public V del;
		[ReadOnly] public NativeArray<T> src;
		[WriteOnly] public NativeArray<BitField64> indices;
		public NativeArray<int> counts;
		public NativeReference<int> totalCount;
		private BitField64 bits;

		public void Execute()
		{
			//remainderJobMarker.Begin();

			int remainderCount = src.Length % 64;
			int dataStartIndex = src.Length - remainderCount;
			bits.Clear();

			for (int i = 0; i < remainderCount; i++)
				bits.SetBits(i, del.Validate(dataStartIndex + i, src[dataStartIndex + i]));

			counts[indices.Length - 1] = math.countbits(bits.Value);
			indices[indices.Length - 1] = bits;

			// Lastly we want to count all of them together 
			for (int i = 0; i < counts.Length; i++)
			{
				totalCount.Value += counts[i];
				// We store the count so far because we can use it later
				counts[i] = totalCount.Value;
			}

			//remainderJobMarker.End();
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
		// The job only supports writing whole 64 bit batches, so we floor here and then run the remainder elsewhere
		int length = (int)math.floor(src.Length / 64f);
		var handle =
			new ParallelIndexingSumJob<T, V>(src, indices, counts, del).Schedule(length, innerBatchCount, dependsOn);
		if (remainder > 0)
			handle = new RemainderSumJob
			{
				src = src,
				indices = indices,
				counts = counts,
				totalCount = totalCount,
				del = del,
			}.Schedule(handle);

		return handle;
	}
}

[BurstCompile(CompileSynchronously = true), GenerateTestsForBurstCompatibility]
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

	// TODO Insert another job to find contiguous ranges between each batch
	// This can then be used to produce fewer threads and larger copy blocks
	
	public void Execute(int index) => ExecuteBatched(index);
	// public void Execute(int index) => ExecuteSingle(index);

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
		// Span<int> temp = stackalloc int[bitCount];

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
			// temp[i] = srcStartIndex + t + i;
			// data.Write(dstStartIndex + i, srcStartIndex + t + i);
	
			i++;
			n >>= tzcnt + 1;
		}

		// for (int j = 0; j < bitCount; j++)
		// {
		// 	data.Write(dstStartIndex + j, temp[j]);
		// }
		data.Write(dstStartIndex, temp, i);
	}

	public unsafe void ExecuteBatched(int index)
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

		// Span<T> temp = stackalloc T[64];
		// data.CopyTo(srcStartIndex, 64, temp);
		
		// data.ReadAsSpan(srcStartIndex, 64).CopyTo(temp);
		// var arr = NativeArrayUnsafeUtility.ConvertExistingDataToNativeArray(temp, Allocator.None);
		// data.Read(srcStartIndex, 64).CopyTo(arr);
		
#if UNITY_BURST_EXPERIMENTAL_PREFETCH_INTRINSIC
		data.PrefetchSrc(srcStartIndex + bitCount);
		data.PrefetchDst(dstStartIndex + bitCount);
#endif
		// fixed (T* ptr = temp)
		// {
			// Batched run-length loop: copies consecutive set-bit runs in a single call
			int i = 0;
			int t = 0;
			while (n != 0)
			{
				int tzcnt = math.tzcnt(n);
				t += tzcnt;
				int runLength = math.tzcnt(~(n >> tzcnt));

				data.Write(dstStartIndex + i, srcStartIndex + t, runLength);

				// var read = data.Read(srcStartIndex + t, runLength);
				// var read = temp[t];
				// UnsafeUtility.MemCpy(ptr + i, read.GetUnsafeReadOnlyPtr(), data.Stride * runLength);
				
				i += runLength;
				t += runLength;
				int shift = tzcnt + runLength;
				n = shift >= 64 ? 0 : n >> shift;
			}
		
			// data.Write(dstStartIndex, temp, i);
		// }
	}
}

[BurstCompile]
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
		handle = new IndicesCopyJob { src = src, indices = indices, dst = dst }.Schedule(indices, innerLoopBatchCount,
			handle);
		return handle;
	}

	[BurstCompile]
	private unsafe struct IndicesCopyJob : IJobParallelForDefer
	{
		[ReadOnly] public NativeArray<T> src;
		[ReadOnly] public NativeList<int> indices;

		[WriteOnly, NativeDisableParallelForRestriction]
		public NativeList<T> dst;

		public IndicesCopyJob(NativeArray<T> src, NativeList<int> indices, NativeList<T> dst)
		{
			this.src = src;
			this.indices = indices;
			this.dst = dst;
		}

		public void Execute(int index)
		{
			Hint.Assume(src.Length > 0);
			Hint.Assume(indices.Length > 0);
			int srcIndex = indices[index];
			Hint.Assume(srcIndex >= 0);

			dst[index] = src[srcIndex];
		}
	}

	[BurstCompile]
	private struct UpdateListLengthJob : IJob
	{
		[WriteOnly] public NativeList<T> list;
		[ReadOnly] public NativeList<int> indices;

		public void Execute() => list.Length = indices.Length;
	}

	[BurstCompile]
	private struct FilterJob : IJobFilter
	{
		[ReadOnly] public NativeArray<T> src;
		public V Validator;

		public bool Execute(int index) => Validator.Validate(index, src[index]);
	}
}
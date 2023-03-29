using System;
using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Burst.Intrinsics;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.UIElements;
using static Unity.Burst.Intrinsics.X86.Avx;
using static Unity.Burst.Intrinsics.X86.Sse;
using static Unity.Burst.Intrinsics.X86.Sse2;
using static Unity.Burst.Intrinsics.X86.Sse3;
using static Unity.Burst.Intrinsics.X86.Sse4_1;
using static Unity.Burst.Intrinsics.X86.Sse4_2;
using Random = Unity.Mathematics.Random;
#if UNITY_BURST_EXPERIMENTAL_PREFETCH_INTRINSIC
using Unity.Burst.Intrinsics;
#endif

public interface IValidator<in T> where T : unmanaged
{
	bool Validate(T element);
}

public interface IIndexWriter
{
	public void Write(int dstIndex, int srcIndex);
	public void Write(int dstIndex, int srcIndex, int srcRange);
	public void Prefetch(int dstIndex, int srcIndex);
}

public interface IIndexWriter<T> : IIndexWriter where T : unmanaged { }

public interface IConditionalCopyJob<T, W> where T : unmanaged where W : struct, IIndexWriter<T> { }

public interface IConditionalIndexingJob<T, M> where T : unmanaged where M : IValidator<T> { }

public unsafe struct GenericWriter<T> : IIndexWriter<T> where T : unmanaged
{
	[ReadOnly, NativeDisableParallelForRestriction]
	private NativeArray<T> src;

	[WriteOnly, NativeDisableParallelForRestriction]
	private NativeArray<T> dst;

	[ReadOnly, NativeDisableUnsafePtrRestriction]
	private readonly T* srcPtr;

	[WriteOnly, NativeDisableUnsafePtrRestriction]
	private readonly T* dstPtr;

	public GenericWriter(NativeArray<T> src, NativeArray<T> dst)
	{
		this.dst = dst;
		this.src = src;
		srcPtr = (T*)src.GetUnsafeReadOnlyPtr();
		dstPtr = (T*)dst.GetUnsafeReadOnlyPtr();
	}

	public NativeArray<T> GetSrc() => src;
	public NativeArray<T> GetDst() => dst;

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public void Write(int dstIndex, int srcIndex) => dst[dstIndex] = src[srcIndex];

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public void Write(int dstIndex, int srcIndex, int srcRange)
	{
		// Prefetch(dstIndex, srcIndex);
		for (int i = 0; i < srcRange; i++)
			dstPtr[dstIndex + i] = srcPtr[srcIndex + i];
	}

	public void Prefetch(int dstIndex, int srcIndex /*, Common.ReadWrite rw, Common.Locality locality*/)
	{
#if UNITY_BURST_EXPERIMENTAL_PREFETCH_INTRINSIC
		Common.Prefetch(srcPtr + srcIndex, Common.ReadWrite.Read, Common.Locality.LowTemporalLocality);
		Common.Prefetch(dstPtr + dstIndex, Common.ReadWrite.Write, Common.Locality.HighTemporalLocality);
#endif
	}
}

#region Parallel Indexing, single copy

/// <summary>
/// This job is meant to pack booleans into <see cref="indices"/>.
/// Then you can use <see cref="ConditionalCopyMergeJob{T,W}"/> to write to a destination array based on the <see cref="indices"/> array.
/// </summary>
/// <typeparam name="T"></typeparam>
/// <typeparam name="V"></typeparam>
[BurstCompile, BurstCompatible]
public struct ConditionIndexingJob<T, V> : IJobParallelFor, IConditionalIndexingJob<T, V> where T : unmanaged where V : IValidator<T>
{
	[ReadOnly] private V del;
	[ReadOnly] private NativeArray<T> src;

	[WriteOnly] public NativeArray<BitField64> indices;
	// private static readonly ProfilerMarker conditionIndexingJobMarker = new ProfilerMarker(nameof(ConditionIndexingJob<T, M>));

	public ConditionIndexingJob(NativeArray<T> src, NativeArray<BitField64> indices, V del = default)
	{
		this.src = src;
		this.indices = indices;
		this.del = del;
	}

	public void Execute(int index)
	{
		//conditionIndexingJobMarker.Begin();

		BitField64 bits = new BitField64(0);
		//byte* bitsPtr = (byte*)&bits.Value;
		//const int iterations = 64 / 4;
		int dataIndex = index * 64;

		for (int i = 0; i < 64; i++)
			bits.SetBits(i, del.Validate(src[dataIndex + i]));
		indices[index] = bits;

		//conditionIndexingJobMarker.End();
	}

	// private static readonly ProfilerMarker remainderJobMarker = new ProfilerMarker(nameof(RemainderJob));
	[BurstCompile, BurstCompatible]
	private struct RemainderJob : IJob
	{
		[ReadOnly] public V del;
		[ReadOnly] public NativeArray<T> src;
		[WriteOnly] public NativeArray<BitField64> indices;
		private BitField64 bits;

		public void Execute()
		{
			//remainderJobMarker.Begin();

			int remainderCount = src.Length % 64;
			int dataStartIndex = src.Length - remainderCount;
			bits.Clear();

			for (int i = 0; i < remainderCount; i++)
				bits.SetBits(i, del.Validate(src[dataStartIndex + i]));

			indices[indices.Length - 1] = bits;

			//remainderJobMarker.End();
		}
	}

	public static JobHandle Schedule(
		NativeArray<T> src,
		NativeArray<BitField64> indices,
		out ConditionIndexingJob<T, V> job,
		int innerBatchCount = 10,
		V del = default)
	{
		int remainder = src.Length % 64;
		// The job only supports writing whole 64 bit batches, so we floor here and then run the remainder elsewhere
		int length = (int)math.floor(src.Length / 64f);
		job = new ConditionIndexingJob<T, V>(src, indices);

		var handle = job.Schedule(length, innerBatchCount);
		if (remainder > 0)
			handle = new RemainderJob
			{
				src = src,
				indices = indices,
				del = del
			}.Schedule(handle);

		return handle;
	}
}

[BurstCompile, BurstCompatible]
public struct ConditionalCopyMergeJob<T, W> : IJobFor, IConditionalCopyJob<T, W> where T : unmanaged where W : struct, IIndexWriter<T>
{
	public W writer;

	private NativeArray<BitField64> indices;
	// private readonly int indicesLength;
	// private readonly long sizeOfT;
	// private static readonly ProfilerMarker conditionalCopyMergeJobMarker = new ProfilerMarker(nameof(ConditionalCopyMergeJob<T, W>));

	private NativeReference<int> counter;

	public ConditionalCopyMergeJob(
		W writer,
		NativeArray<BitField64> indices,
		NativeReference<int> count)
	{
		this.writer = writer;
		// indicesLength = indices.Length;
		counter = count;
		this.indices = indices;
		// sizeOfT = UnsafeUtility.SizeOf(typeof(T));
		//cumulativeTzcnt = default;
		//consecutiveStartIndex = default;
	}

	public void Execute(int index)
	{
		ExecuteNoBatches(index);
		// ExecuteBatched(index);
	}

	private void ExecuteNoBatches(int index)
	{
		//conditionalCopyMergeJobMarker.Begin();

		int countSoFar = counter.Value;
		int srcStartIndex = index * 64;

		ulong n = indices[index].Value;
		int t = 0;
		//int popCount = math.countbits(n);
		//for (int i = 0; i < popCount; i++)
		//{
		//	int tzcnt = math.tzcnt(n);
		//	t += tzcnt;
		//	//Debug.Log($"Writing from index {srcStartIndex + t + i}");
		//	// MemCpy is slower for some reason
		//	//UnsafeUtility.MemCpy(dstPtr + countSoFar, srcPtr + srcStartIndex + t + i, sizeOfT);
		//	dst[countSoFar] = src[srcStartIndex + t + i];
		//	i++;
		//	countSoFar++;
		//	n >>= tzcnt + 1;
		//}
		//if (math.countbits(n) == 64)
		//{
		//	// This is more expensive for some reason than the while loop
		//	UnsafeUtility.MemCpy(dstPtr + countSoFar, srcPtr + srcStartIndex + t + 64, sizeOfT);
		//	counter.Value += 64;
		//}
		//else
		//{
		int i = 0;
		while (n != 0)
		{
			int tzcnt = math.tzcnt(n);
			t += tzcnt;
			//Debug.Log($"Writing from index {srcStartIndex + t + i}");

			// MemCpy is slower for some reason
			//UnsafeUtility.MemCpy(dstPtr + countSoFar, srcPtr + srcStartIndex + t + i, sizeOfT);

			writer.Write(countSoFar, srcStartIndex + t + i);

			countSoFar++;
			i++;

			n >>= tzcnt + 1;
		}

		counter.Value = countSoFar;
		//}

		//conditionalCopyMergeJobMarker.End();
	}

	private void ExecuteBatched(int index)
	{
		//conditionalCopyMergeJobMarker.Begin();

		int countSoFar = counter.Value;
		int srcStartIndex = index * 64;
		//int consecutiveStartIndex = 0;
		int consecutiveStartIndex = 0;

		ulong n = indices[index].Value;
		int cumulativeTzcnt = 0;

		int i = 0;
		int consecutiveCount = default;

		// TODO Try converting this to a for loop using popcount
		while (n != 0)
		{
			int tzcnt = math.tzcnt(n);
			cumulativeTzcnt += tzcnt;

			if (tzcnt != 0 && consecutiveCount > 0)
			{
				// Debug.Log("Start: " + consecutiveStartIndex);
				//writer.Write(countSoFar, srcStartIndex + t + i);

				for (int j = 0; j < consecutiveCount; j++)
					writer.Write(dstIndex: countSoFar + j, srcIndex: consecutiveStartIndex + j);
				countSoFar += consecutiveCount;

				// This works, more or less
				consecutiveStartIndex = cumulativeTzcnt + i + srcStartIndex;
				consecutiveCount = 0;
			}

			consecutiveCount++;
			i++;

			n >>= tzcnt + 1;
		}

		if (consecutiveCount != 0)
		{
			if (consecutiveStartIndex == 0)
				consecutiveStartIndex = srcStartIndex;

			// Debug.Log("Start Remainder: " + consecutiveStartIndex);
			for (int j = 0; j < consecutiveCount; j++)
				writer.Write(dstIndex: countSoFar + j, srcIndex: consecutiveStartIndex + j);
			countSoFar += consecutiveCount;
		}

		counter.Value = countSoFar;
		//conditionalCopyMergeJobMarker.End();
	}

	public static JobHandle Schedule<M>(ConditionIndexingJob<T, M> indexingJob,
		W writer,
		NativeReference<int> counter,
		JobHandle dependsOn = default) where M : IValidator<T>
	{
		//GenericWriter<T> writer = new GenericWriter<T>() { src = indexingJob.src, dst = dst };
		ConditionalCopyMergeJob<T, W> job = new ConditionalCopyMergeJob<T, W>(writer, indexingJob.indices, counter);
		var handle = job.Schedule(indexingJob.indices.Length, dependsOn);

		return handle;
	}
}

#endregion

// This parallel setup is a fair bit faster, but also consumes far more of the CPU so probably not worth it in many cases

#region Parallel indexing, Single Sum, parallel copy

/// <summary>
/// This job is meant to pack booleans into <see cref="indices"/>.
/// Then you can use <see cref="ConditionalCopyMergeJob{T,W}"/> to write to a destination array based on the <see cref="indices"/> array.
/// It also counts the bits set and assigns them to <see cref="counts"/>.
/// </summary>
/// <typeparam name="T"></typeparam>
/// <typeparam name="V"></typeparam>
[BurstCompile, BurstCompatible]
public struct ParallelIndexingSumJob<T, V> : IJobParallelFor, IConditionalIndexingJob<T, V> where T : unmanaged where V : IValidator<T>
{
	[ReadOnly] public V del;
	[ReadOnly] public NativeArray<T> src;
	[WriteOnly] public NativeArray<BitField64> indices;

	[WriteOnly] public NativeArray<int> counts;
	// private static readonly ProfilerMarker conditionIndexingSumJobMarker = new ProfilerMarker(nameof(ConditionIndexingSumJob<T, M>));

	public ParallelIndexingSumJob(NativeArray<T> src, NativeArray<BitField64> indices, NativeArray<int> counts, V del = default)
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

		for (int i = 0; i < 64; i += 4)
		{
			
			bits.SetBits(i, del.Validate(src[dataIndex + i]));
			bits.SetBits(i + 1, del.Validate(src[dataIndex + i + 1]));
			bits.SetBits(i + 2, del.Validate(src[dataIndex + i + 2]));
			bits.SetBits(i + 3, del.Validate(src[dataIndex + i + 3]));
		}
		//
		// for (int i = 0; i < 64; i++)
		// {
		// 	bool v = del.Validate(src[dataIndex + i]);
		// 	// This one seems to generate less instructions, but not vectorized. The performance was the same as the line below however.
		// 	bits.SetBits(i, v);
		// 	// This generates more vectorized instructions with the same performance, I'm guessing the power cost is higher for this line though.
		// 	// bits.Value |= (del.Validate(src[dataIndex + i]) ? 1ul : 0ul) << i;
		// }

		counts[index] = math.countbits(bits.Value);
		indices[index] = bits;

		//conditionIndexingSumJobMarker.End();
	}

	// private static readonly ProfilerMarker remainderJobMarker = new ProfilerMarker(nameof(RemainderSumJob));
	/// <summary>
	/// This job sets the bits and sums the last element of <see cref="indices"/>.
	/// It also will count all the bits at the end and store the count so far in <see cref="counts"/>.
	/// </summary>
	[BurstCompile, BurstCompatible]
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
				bits.SetBits(i, del.Validate(src[dataStartIndex + i]));

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
		out ParallelIndexingSumJob<T, V> job,
		int innerBatchCount = 10,
		V del = default)
	{
		int remainder = src.Length % 64;
		// The job only supports writing whole 64 bit batches, so we floor here and then run the remainder elsewhere
		int length = (int)math.floor(src.Length / 64f);
		job = new ParallelIndexingSumJob<T, V>(src, indices, counts);

		var handle = job.Schedule(length, innerBatchCount);
		if (remainder > 0)
			handle = new RemainderSumJob
			{
				src = src,
				indices = indices,
				counts = counts,
				totalCount = totalCount,
				del = del
			}.Schedule(handle);

		return handle;
	}
}

[BurstCompile, BurstCompatible]
public unsafe struct ParallelConditionalCopyJob<T, W> : IJobParallelFor, IConditionalCopyJob<T, W> where T : unmanaged where W : struct, IIndexWriter<T>
{
	public W writer;
	[ReadOnly] public NativeArray<int> counts;

	[NativeDisableUnsafePtrRestriction, ReadOnly]
	private readonly BitField64* bitsPtr;

	// private readonly int indicesLength;
	// private static readonly ProfilerMarker parallelCopyJobMarker = new ProfilerMarker(nameof(ParallelConditionalCopyJob<T, W>));

	public ParallelConditionalCopyJob(
		W writer,
		NativeArray<BitField64> indices,
		NativeArray<int> counts)
	{
		this.writer = writer;
		// indicesLength = indices.Length;
		this.counts = counts;
		bitsPtr = (BitField64*)indices.GetUnsafeReadOnlyPtr();
	}

	public void Execute(int index)
	{
		//parallelCopyJobMarker.Begin();

		// We need to start write index of the src data which we can get from counts
		int dstStartIndex = index == 0 ? 0 : counts[math.max(0, index - 1)];

		int srcStartIndex = index * 64;
		ulong n = bitsPtr[index].Value;

		// writer.Prefetch(dstStartIndex, srcStartIndex);

		int i = 0;
		int t = 0;
		while (n != 0)
		{
			int tzcnt = math.tzcnt(n);
			t += tzcnt;
			//Debug.Log($"Writing from index {srcStartIndex + t + i}");
			writer.Write(dstStartIndex + i, srcStartIndex + t + i);
			i++;
			n >>= tzcnt + 1;
		}

		//parallelCopyJobMarker.End();
	}

	public static JobHandle Schedule<V>(ParallelIndexingSumJob<T, V> indexingJob,
		W writer,
		int innerBatchCount = 10,
		JobHandle dependsOn = default) where V : IValidator<T>
	{
		ParallelConditionalCopyJob<T, W> job = new ParallelConditionalCopyJob<T, W>(writer, indexingJob.indices, indexingJob.counts);
		var handle = job.Schedule(indexingJob.indices.Length, innerBatchCount, dependsOn);

		return handle;
	}
}

#endregion
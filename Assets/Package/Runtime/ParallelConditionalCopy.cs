using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Burst.Intrinsics;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine.Assertions;

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

public static class NativeCollectionExtensions
{
	/// <summary>
	/// Copies data filtered from <param name="src"></param> to <param name="dst"></param> using <see cref="IValidator{T}"/> and parallel for jobs.
	/// There is one step in the job chain which just uses a <see cref="IJobFor"/> to count how many should be written.
	/// The count written is accesible from <param name="counter"></param> which must be disposed of after use.
	/// </summary>
	/// <param name="src"></param>
	/// <param name="dst"></param>
	/// <param name="counter"></param>
	/// <param name="indexingBatchCount"></param>
	/// <param name="writeBatchCount"></param>
	/// <param name="dependsOn"></param>
	/// <param name="indices"></param>
	/// <param name="counts"></param>
	/// <typeparam name="T"></typeparam>
	/// <typeparam name="V"></typeparam>
	/// <returns></returns>
	public static JobHandle IfCopyToParallel<T, V>(this NativeArray<T> src, NativeArray<T> dst,
		out NativeReference<int> counter,
		int indexingBatchCount = 64,
		int writeBatchCount = 64,
		JobHandle dependsOn = default,
		NativeArray<BitField64> indices = default,
		NativeArray<int> counts = default) where T : unmanaged where V : unmanaged, IValidator<T>
	{
		GenericWriter<T> writer = new GenericWriter<T>(src, dst);
		Assert.IsTrue(dst.Length >= src.Length, "Assert Failed: dst.Length < src.Length");
		
		bool tempBits = !indices.IsCreated;
		if (tempBits)
			indices = new NativeArray<BitField64>((int)math.ceil(src.Length / 64f), Allocator.TempJob);
		bool tempCounts = !counts.IsCreated;
		if (tempCounts)
			counts = new NativeArray<int>(src.Length, Allocator.TempJob);
		counter = new NativeReference<int>(0, Allocator.TempJob);
		
		ParallelConditionalCopyJob<T, GenericWriter<T>> copyJob = new ParallelConditionalCopyJob<T, GenericWriter<T>>(writer, indices, counts);
		int indicesLength = indices.Length;

		var handle = ParallelIndexingSumJob<T, V>.Schedule(src, indices, counts, counter, indexingBatchCount, dependsOn);
		handle = copyJob.Schedule(indicesLength, writeBatchCount, handle);

		if(tempBits)
			indices.Dispose(handle);
		if (tempCounts)
			counts.Dispose(handle);
		
		return handle;
	}
	
	public static JobHandle IfCopyToParallel<T, V>(this NativeArray<T> src, NativeList<T> dst,
		int indexingBatchCount = 64,
		int writeBatchCount = 64,
		JobHandle dependsOn = default,
		NativeArray<BitField64> indices = default,
		NativeArray<int> counts = default) where T : unmanaged where V : unmanaged, IValidator<T>
	{
		Assert.IsTrue(dst.Capacity >= src.Length, "Assert Failed: dst.Capacity < src.Length");
		dst.ResizeUninitialized(dst.Capacity);
		
		var handle = src.IfCopyToParallel<T, V>(dst.AsArray(), out var counter, indexingBatchCount, writeBatchCount, dependsOn, indices, counts);
		// Set length to the counted length instead of capacity
		handle = new AssignJobLengthJob<T>() { dst = dst, count = counter }.Schedule(handle);
		counter.Dispose(handle);
		
		return handle;
	}

	private struct AssignJobLengthJob<T> : IJob where T : unmanaged
	{
		public NativeList<T> dst;
		public NativeReference<int> count;
		
		public void Execute() => dst.ResizeUninitialized(count.Value);
	}
}

[BurstCompatible, BurstCompile]
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

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public void Write(int dstIndex, int srcIndex)
	{
		Prefetch(dstIndex, srcIndex);
		dst[dstIndex] = src[srcIndex];
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public void Write(int dstIndex, int srcIndex, int srcRange)
	{
		for (int i = 0; i < srcRange; i++)
			dstPtr[dstIndex + i] = srcPtr[srcIndex + i];
	}

	public readonly void Prefetch(int dstIndex, int srcIndex)
	{
#if UNITY_BURST_EXPERIMENTAL_PREFETCH_INTRINSIC
		Common.Prefetch(srcPtr + srcIndex, Common.ReadWrite.Read, Common.Locality.LowTemporalLocality);
		Common.Prefetch(dstPtr + dstIndex, Common.ReadWrite.Write, Common.Locality.HighTemporalLocality);
#endif
	}
}

/// <summary>
/// This job is meant to pack booleans into <see cref="indices"/>.
/// Then you can use <see cref="ParallelConditionalCopyJob{T,W}"/> to write to a destination array based on the <see cref="indices"/> array.
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
		
		for (int i = 0; i < 64; i++)
		{
			bool v = del.Validate(src[dataIndex + i]);
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
		int innerBatchCount = 10,
		JobHandle dependsOn = default,
		V del = default)
	{
		int remainder = src.Length % 64;
		// The job only supports writing whole 64 bit batches, so we floor here and then run the remainder elsewhere
		int length = (int)math.floor(src.Length / 64f);
		var job = new ParallelIndexingSumJob<T, V>(src, indices, counts);

		var handle = job.Schedule(length, innerBatchCount, dependsOn);
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

[BurstCompile, BurstCompatible]
public unsafe struct ParallelConditionalCopyJob<T, W> : IJobParallelFor, IConditionalCopyJob<T, W> where T : unmanaged where W : struct, IIndexWriter<T>
{
	public W writer;
	[ReadOnly] public NativeArray<int> counts;

	[NativeDisableUnsafePtrRestriction, ReadOnly]
	private readonly BitField64* bitsPtr;

	public ParallelConditionalCopyJob(
		W writer,
		NativeArray<BitField64> indices,
		NativeArray<int> counts)
	{
		this.writer = writer;
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
		
		int i = 0;
		int t = 0;
		while (n != 0)
		{
			int tzcnt = math.tzcnt(n);
			t += tzcnt;
			writer.Write(dstStartIndex + i, srcStartIndex + t + i);
			i++;
			n >>= tzcnt + 1;
		}
		
		// The following code is slightly slower than the above code
		// int total_set_bits = math.countbits(n);
		// for (int i = 0, bitIndex = 0; i < total_set_bits; ++bitIndex)
		// {
		// 	if ((n & (1UL << bitIndex)) != 0)
		// 	{
		// 		writer.Write(dstStartIndex + i, srcStartIndex + bitIndex);
		// 		++i;
		// 	}
		// }
		//parallelCopyJobMarker.End();
	}
}
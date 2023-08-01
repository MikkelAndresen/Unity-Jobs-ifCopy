using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;

/// <summary>
/// This job is meant to pack booleans into <see cref="indices"/>.
/// Then you can use <see cref="ParallelConditionalCopyJob{T,W}"/> to write to a destination array based on the <see cref="indices"/> array.
/// It also counts the bits set and assigns them to <see cref="counts"/>.
/// </summary>
/// <typeparam name="T"></typeparam>
/// <typeparam name="V"></typeparam>
[BurstCompile, GenerateTestsForBurstCompatibility]
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
	[BurstCompile, GenerateTestsForBurstCompatibility]
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

[BurstCompile, GenerateTestsForBurstCompatibility]
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
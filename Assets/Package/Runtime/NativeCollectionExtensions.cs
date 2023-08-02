using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine.Assertions;

// ReSharper disable once UnusedType.Global
public static class NativeCollectionExtensions
{
	/// <summary>
	/// This class exists to make it easier to use the <see cref="IfCopyToParallel"/> function without having to manually allocated both the indices and counts arrays.
	/// Whenever you call either of the copy methods, a temp <see cref="NativeReference{T}"/> will be allocated. In order to release it you need to also call either <see cref="ReleaseTempMemory"/> or <see cref="CountCopied"/>.
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <typeparam name="V"></typeparam>
	public unsafe class CopyHandler<T, V> : IDisposable where T : unmanaged where V : unmanaged, IValidator<T>
	{
		public NativeArray<T> src;
		public NativeArray<T> dst;
		public int indexingBatchCount;
		public int writeBatchCount;
		public V validator;

		private int lastCount;

		/// <summary>
		/// Returns the value of the <see cref="NativeReference{T}"/> counter used when the <see cref="IfCopyToParallel"/> function is run.
		/// Returns zero if nothing was copied or the function was never run.
		/// This property is a bit special because when you access it, it also disposes the temp native reference counter.
		/// The value of the disposed counter is cached.
		/// You can also dispose of the temp memory using <see cref="ReleaseTempMemory"/>
		/// </summary>
		public int CountCopied
		{
			get
			{
				if (!counter.IsCreated) return lastCount;
				lastCount = counter.Value;
				counter.Dispose();
				return lastCount;
			}
		}

		public void ReleaseTempMemory()
		{
			if (!counter.IsCreated)
				return;
			lastCount = counter.Value;
			counter.Dispose();
		}

		private NativeArray<BitField64> indices;
		private NativeArray<int> counts;
		private NativeReference<int> counter;
		private readonly T* srcReadOnlyPtr;

		public CopyHandler(
			int srcLength,
			NativeArray<T> dst,
			int indexingBatchCount = 64,
			int writeBatchCount = 64,
			V validator = default)
		{
			src = new NativeArray<T>(srcLength, Allocator.Persistent);
			srcReadOnlyPtr = (T*)src.GetUnsafeReadOnlyPtr();

			this.dst = dst;
			this.indexingBatchCount = indexingBatchCount;
			this.writeBatchCount = writeBatchCount;
			this.validator = validator;
			int indicesLength = (int)math.ceil(src.Length / 64f);

			indices = new NativeArray<BitField64>(indicesLength, Allocator.Persistent);
			counts = new NativeArray<int>(indicesLength, Allocator.Persistent);
		}

		public JobHandle IfCopyToParallel(JobHandle dependsOn = default) => src.IfCopyToParallel(dst, out counter, indexingBatchCount, writeBatchCount, dependsOn, indices, counts, validator);
		public JobHandle IfCopyToParallelUnsafe(T* dstReadOnlyPtr, JobHandle dependsOn = default) => src.IfCopyToParallel(dst, srcReadOnlyPtr, dstReadOnlyPtr, out counter, indexingBatchCount, writeBatchCount, dependsOn, indices, counts, validator);

		public void Dispose()
		{
			src.Dispose();
			indices.Dispose();
			counts.Dispose();
			if (counter.IsCreated)
				counter.Dispose();
		}
	}

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
	/// <param name="validator"></param>
	/// <typeparam name="T"></typeparam>
	/// <typeparam name="V"></typeparam>
	/// <returns></returns>
	public unsafe static JobHandle IfCopyToParallel<T, V>(
		this NativeArray<T> src,
		NativeArray<T> dst,
		out NativeReference<int> counter,
		int indexingBatchCount = 64,
		int writeBatchCount = 64,
		JobHandle dependsOn = default,
		NativeArray<BitField64> indices = default,
		NativeArray<int> counts = default,
		V validator = default) where T : unmanaged where V : unmanaged, IValidator<T> => IfCopyToParallel(
		src, dst, (T*)src.GetUnsafeReadOnlyPtr(), (T*)dst.GetUnsafeReadOnlyPtr(),
		out counter, indexingBatchCount, writeBatchCount, dependsOn, indices, counts, validator);

	public unsafe static JobHandle IfCopyToParallel<T, V>(
		this NativeArray<T> src,
		NativeArray<T> dst,
		T* srcReadOnlyPtr,
		T* dstReadOnlyPtr,
		out NativeReference<int> counter,
		int indexingBatchCount = 64,
		int writeBatchCount = 64,
		JobHandle dependsOn = default,
		NativeArray<BitField64> indices = default,
		NativeArray<int> counts = default,
		V validator = default) where T : unmanaged where V : unmanaged, IValidator<T>
	{
		Assert.IsTrue(dst.Length >= src.Length, "Assert Failed: dst.Length < src.Length");
		int indicesLength = (int)math.ceil(src.Length / 64f);

		bool tempBits = indices.IsCreated;
		var tempIndicesArray = tempBits ? new NativeArray<BitField64>(indicesLength, Allocator.TempJob) : default;

		bool tempCounts = !counts.IsCreated;
		if (tempCounts)
			counts = new NativeArray<int>(src.Length, Allocator.TempJob);
		counter = new NativeReference<int>(0, Allocator.TempJob);

		GenericWriter<T> writer = new GenericWriter<T>(src, dst, srcReadOnlyPtr, dstReadOnlyPtr);
		ParallelConditionalCopyJob<T, GenericWriter<T>> copyJob = new ParallelConditionalCopyJob<T, GenericWriter<T>>(writer, indices, counts);

		var handle = ParallelIndexingSumJob<T, V>.Schedule(src, indices, counts, counter, indexingBatchCount, dependsOn, validator);
		handle = copyJob.Schedule(indicesLength, writeBatchCount, handle);

		if (tempIndicesArray.IsCreated)
			tempIndicesArray.Dispose(handle);
		if (tempCounts)
			counts.Dispose(handle);

		return handle;
	}

	public static JobHandle IfCopyToParallel<T, V>(this NativeArray<T> src, NativeList<T> dst,
		int indexingBatchCount = 64,
		int writeBatchCount = 64,
		JobHandle dependsOn = default,
		NativeArray<BitField64> indices = default,
		NativeArray<int> counts = default,
		V validator = default) where T : unmanaged where V : unmanaged, IValidator<T>
	{
		Assert.IsTrue(dst.Capacity >= src.Length, "Assert Failed: dst.Capacity < src.Length");
		dst.ResizeUninitialized(dst.Capacity);

		var handle = src.IfCopyToParallel<T, V>(dst.AsArray(), out var counter, indexingBatchCount, writeBatchCount, dependsOn, indices, counts, validator);
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
using System;
using UnityEngine;
using Unity.Mathematics;
using Unity.Collections;
using Unity.Jobs;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Profiling;

public class CopyTestBehaviour : CopyTestBehaviour<float3x4, CopyTestBehaviour.GreaterThanZeroDel>
{
	public struct GreaterThanZeroDel : IValidator<float3x4>
	{
		public bool Validate(float3x4 element) => element.c0.x > 0;
	}

	private enum TestDataType
	{
		None,
		All,
		Odd,
		Half
	}

	[SerializeField] private TestDataType dataGenMethod = TestDataType.Odd;

	protected override float3x4 GetData(int i) =>
		dataGenMethod switch
		{
			TestDataType.None => -1,
			TestDataType.All  => 1,
			TestDataType.Odd  => i % 2 == 0 ? -1 : 1,
			TestDataType.Half => i > 50 ? 1 : -1,
			_                 => default,
		};

	protected override ParallelIndexingSumJob<float3x4, GreaterThanZeroDel> GetJob()
	{
		return new ParallelIndexingSumJob<float3x4, GreaterThanZeroDel>();
	}
}

public abstract class CopyTestBehaviour<T, V> : MonoBehaviour where T : unmanaged where V : unmanaged, IValidator<T>
{
	[SerializeField] private int dataLength = 100;
	[SerializeField] private bool useGPUBuffer;
	[SerializeField] private int indexingBatchCount = 2;
	[SerializeField] private int writeBatchCount = 2;
	[SerializeField] private bool completeInLateUpdate = false;
	[SerializeField] private bool useScheduleUtility;

	private NativeArray<T> src;
	private NativeArray<T> dstData;
	private NativeReference<int> counter;
	private NativeArray<int> counts;
	private NativeArray<BitField64> bits;
	private ComputeBuffer gpuBuffer;
	protected abstract ParallelIndexingSumJob<T, V> GetJob();

	protected virtual void Start()
	{
		src = new NativeArray<T>(dataLength, Allocator.Persistent);
		bits = new NativeArray<BitField64>((int)math.ceil(dataLength / 64f), Allocator.Persistent);
		dstData = new NativeArray<T>(dataLength, Allocator.Persistent);
		counter = new NativeReference<int>(0, Allocator.Persistent);
		counts = new NativeArray<int>(bits.Length, Allocator.Persistent);

		for (int i = 0; i < src.Length; i++)
			src[i] = GetData(i);

		if (useGPUBuffer)
			gpuBuffer = new ComputeBuffer(dataLength, UnsafeUtility.SizeOf(typeof(T)), ComputeBufferType.Default,
				ComputeBufferMode.SubUpdates);
	}

	protected abstract T GetData(int i);

	private static readonly ProfilerMarker indexingSumJobMarker = new ProfilerMarker(nameof(ParallelIndexingSumJob<T, V>));
	private static readonly ProfilerMarker parallelCopyJobMarker = new ProfilerMarker(nameof(ParallelConditionalCopyJob<T, GenericWriter<T>>));

	private JobHandle handle;

	private void Update()
	{
		counter.Value = 0;
		NativeArray<T> dst = useGPUBuffer ? gpuBuffer.BeginWrite<T>(0, dataLength) : dstData;
		var writer = new GenericWriter<T>(src, dst);
		
		if (useScheduleUtility)
		{
			handle = writer.Schedule<V>(indexingBatchCount, writeBatchCount);
			if (!completeInLateUpdate)
				handle.Complete();
		}
		else
		{
			indexingSumJobMarker.Begin();
			handle = ParallelIndexingSumJob<T, V>.Schedule(src, bits, counts, counter, out var indexSumJob, indexingBatchCount);
			if (!completeInLateUpdate)
				handle.Complete();
			indexingSumJobMarker.End();

			parallelCopyJobMarker.Begin();
			handle = ParallelConditionalCopyJob<T, GenericWriter<T>>.Schedule(indexSumJob, writer, writeBatchCount, handle);
			if (!completeInLateUpdate)
				handle.Complete();
			parallelCopyJobMarker.End();
		}

		if (useGPUBuffer && !completeInLateUpdate)
			gpuBuffer.EndWrite<float3x4>(dataLength);
	}

	private void LateUpdate()
	{
		if (!completeInLateUpdate)
			return;

		handle.Complete();
		if (useGPUBuffer)
			gpuBuffer.EndWrite<float3x4>(dataLength);
	}

	private void OnDestroy()
	{
		src.Dispose();
		bits.Dispose();
		dstData.Dispose();
		counter.Dispose();
		counts.Dispose();
		if (useGPUBuffer)
			gpuBuffer.Dispose();
	}
}
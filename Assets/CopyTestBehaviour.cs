using System;
using UnityEngine;
using Unity.Mathematics;
using Unity.Collections;
using Unity.Jobs;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Profiling;

public class CopyTestBehaviour : MonoBehaviour
{
	[SerializeField] private int dataLength = 100;
	[SerializeField] private bool useGPUBuffer;
	[SerializeField] private int indexingBatchCount = 2;
	[SerializeField] private int writeBatchCount = 2;
	[SerializeField] private bool completeInLateUpdate = false;
	[SerializeField] private bool useScheduleUtility;

	private NativeArray<float> src;
	private NativeArray<float> dstData;
	private NativeReference<int> counter;
	private NativeArray<int> counts;
	private NativeArray<BitField64> bits;
	private ComputeBuffer gpuBuffer;

	protected virtual void Start()
	{
		src = new NativeArray<float>(dataLength, Allocator.Persistent);
		bits = new NativeArray<BitField64>((int)math.ceil(dataLength / 64f), Allocator.Persistent);
		dstData = new NativeArray<float>(dataLength, Allocator.Persistent);
		counter = new NativeReference<int>(0, Allocator.Persistent);
		counts = new NativeArray<int>(bits.Length, Allocator.Persistent);

		for (int i = 0; i < src.Length; i++)
			src[i] = GetData(i);

		if (useGPUBuffer)
			gpuBuffer = new ComputeBuffer(dataLength, UnsafeUtility.SizeOf(typeof(float)), ComputeBufferType.Default,
				ComputeBufferMode.SubUpdates);
	}

	public struct GreaterThanZeroDel : IValidator<float>
	{
		public bool Validate(float element) => element > 0;
	}

	private enum TestDataType
	{
		None,
		All,
		Odd,
		Half
	}

	[SerializeField] private TestDataType dataGenMethod = TestDataType.Odd;

	protected float GetData(int i) =>
		dataGenMethod switch
		{
			TestDataType.None => -1,
			TestDataType.All  => 1,
			TestDataType.Odd  => i % 2 == 0 ? -1 : 1,
			TestDataType.Half => i > 50 ? 1 : -1,
			_                 => default,
		};

	private static readonly ProfilerMarker indexingSumJobMarker = new ProfilerMarker(nameof(ParallelIndexingSumJob<float, GreaterThanZeroDel>));
	private static readonly ProfilerMarker parallelCopyJobMarker = new ProfilerMarker(nameof(ParallelConditionalCopyJob<float, GenericWriter<float>>));

	private JobHandle handle;

	private void Update()
	{
		counter.Value = 0;
		NativeArray<float> dst = useGPUBuffer ? gpuBuffer.BeginWrite<float>(0, dataLength) : dstData;
		var writer = new GenericWriter<float>(src, dst);

		if (useScheduleUtility)
		{
			handle = writer.Schedule<GreaterThanZeroDel>(indexingBatchCount, writeBatchCount);
			if (!completeInLateUpdate)
				handle.Complete();
		}
		else
		{
			indexingSumJobMarker.Begin();
			handle = ParallelIndexingSumJob<float, GreaterThanZeroDel>.Schedule(src, bits, counts, counter, out var indexSumJob, indexingBatchCount);
			if (!completeInLateUpdate)
				handle.Complete();
			indexingSumJobMarker.End();

			parallelCopyJobMarker.Begin();
			handle = ParallelConditionalCopyJob<float, GenericWriter<float>>.Schedule(indexSumJob, writer, writeBatchCount, handle);
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
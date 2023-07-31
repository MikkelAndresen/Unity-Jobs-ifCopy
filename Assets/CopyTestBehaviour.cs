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
	[SerializeField] private TestDataType dataGenMethod = TestDataType.Odd;

	private NativeArray<float> src;
	private NativeList<float> dstData;
	private NativeReference<int> counter;
	private NativeArray<int> counts;
	private NativeArray<BitField64> indices;
	private ComputeBuffer gpuBuffer;
	private JobHandle handle;

	private static readonly ProfilerMarker indexingSumJobMarker = new ProfilerMarker(nameof(ParallelIndexingSumJob<float, GreaterThanZeroDel>));
	private static readonly ProfilerMarker parallelCopyJobMarker = new ProfilerMarker(nameof(ParallelConditionalCopyJob<float, GenericWriter<float>>));

	void Start()
	{
		src = new NativeArray<float>(dataLength, Allocator.Persistent);
		indices = new NativeArray<BitField64>((int)math.ceil(dataLength / 64f), Allocator.Persistent);
		dstData = new NativeList<float>(dataLength, Allocator.Persistent);
		counter = new NativeReference<int>(0, Allocator.Persistent);
		counts = new NativeArray<int>(indices.Length, Allocator.Persistent);

		for (int i = 0; i < src.Length; i++)
			src[i] = GetData(i);

		if (useGPUBuffer)
			gpuBuffer = new ComputeBuffer(dataLength, UnsafeUtility.SizeOf(typeof(float)), ComputeBufferType.Default,
				ComputeBufferMode.SubUpdates);
		// For testing
		dstData.Length = dataLength;
	}

	private void Update()
	{
		counter.Value = 0;
		var writer = useGPUBuffer ? 
			new GenericWriter<float>(src, gpuBuffer.BeginWrite<float>(0, dataLength)) : 
			new GenericWriter<float>(src, dstData);

		if (useScheduleUtility)
		{
			handle = src.IfCopyToParallel<float, GreaterThanZeroDel>(dstData, indexingBatchCount, writeBatchCount);
			if (!completeInLateUpdate)
				handle.Complete();
		}
		else
		{
			var copyJob = new ParallelConditionalCopyJob<float, GenericWriter<float>>(writer, indices, counts);

			indexingSumJobMarker.Begin();
			handle = ParallelIndexingSumJob<float, GreaterThanZeroDel>.Schedule(src, indices, counts, counter, indexingBatchCount);
			if (!completeInLateUpdate)
				handle.Complete();
			indexingSumJobMarker.End();
			
			parallelCopyJobMarker.Begin();
			handle = copyJob.Schedule(indices.Length, writeBatchCount, handle);
			
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
	
	protected float GetData(int i) =>
		dataGenMethod switch
		{
			TestDataType.None => -1,
			TestDataType.All => 1,
			TestDataType.Odd => i % 2 == 0 ? -1 : 1,
			TestDataType.Half => i > 50 ? 1 : -1,
			_ => default,
		};
	
	private void OnDestroy()
	{
		src.Dispose();
		indices.Dispose();
		dstData.Dispose();
		counter.Dispose();
		counts.Dispose();
		if (useGPUBuffer)
			gpuBuffer.Dispose();
	}
	
	private enum TestDataType
	{
		None,
		All,
		Odd,
		Half
	}
	
	public struct GreaterThanZeroDel : IValidator<float>
	{
		public bool Validate(float element) => element > 0;
	}
}
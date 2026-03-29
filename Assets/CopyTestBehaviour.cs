using Unity.Burst;
using UnityEngine;
using Unity.Mathematics;
using Unity.Collections;
using Unity.Jobs;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Profiling;

/// <summary>
/// This class can be used to run the code in update to see prolonged performance in the profiler.
/// </summary>
public class CopyTestBehaviour : MonoBehaviour
{
	[SerializeField] private int dataLength = 100;
	[SerializeField] private bool useGPUBuffer;
	[SerializeField] private int indexingBatchCount = 2;
	[SerializeField] private int writeBatchCount = 2;
	[SerializeField] private bool completeInLateUpdate = false;
	[SerializeField] private bool useScheduleUtility;
	[SerializeField] private bool useScheduleUtilityPreAllocatedCollections;
	[SerializeField] private TestDataType dataGenMethod = TestDataType.Odd;
	[SerializeField] private bool runAndMeasureBasicCopyJob;
	[SerializeField] private bool runAndMeasureConditionalCopyJob;
	[SerializeField] private bool runAndMeasureFilterJob;

	private NativeArray<float3x4> src;
	private NativeList<float3x4> dstData;
	private NativeReference<int> counter;
	private NativeReference<int> tempCounter;
	private NativeArray<int> counts;
	private NativeArray<BitField64> indices;
	private NativeList<int> filterIndices;
	private ComputeBuffer gpuBuffer;
	private JobHandle handle;

	private static readonly ProfilerMarker indexingSumJobMarker =
		new (nameof(ParallelIndexingSumJob<float3x4, GreaterThanZeroDel>));
	private static readonly ProfilerMarker parallelCopyJobMarker =
		new (nameof(ParallelConditionalCopyJob<float3x4, DataRW<float3x4>>));
	private static readonly ProfilerMarker basicCopyJobMarker =
		new (nameof(CopyJob<float3x4>));
	private static readonly ProfilerMarker conditionalListCopyJobMarker =
		new (nameof(ConditionalCopyJob<float3x4, GreaterThanZeroDel>));
	private static readonly ProfilerMarker filterJobMarker =
		new (nameof(FilterCopy<float3x4, GreaterThanZeroDel>));

	private void Start()
	{
		src = new NativeArray<float3x4>(dataLength, Allocator.Persistent);
		indices = new NativeArray<BitField64>((int)math.ceil(dataLength / 64f), Allocator.Persistent);
		dstData = new NativeList<float3x4>(dataLength, Allocator.Persistent);
		counts = new NativeArray<int>(indices.Length, Allocator.Persistent);
		counter = new NativeReference<int>(Allocator.Persistent);
		filterIndices = new NativeList<int>(dataLength, Allocator.Persistent);

		for (int i = 0; i < src.Length; i++)
			src[i] = GetData(i);

		gpuBuffer = new ComputeBuffer(dataLength, UnsafeUtility.SizeOf(typeof(float3x4)), ComputeBufferType.Default,
			ComputeBufferMode.SubUpdates);
	}

	private void Update()
	{
		counter.Value = 0;

		// Reset for next test
		dstData.Resize(0, NativeArrayOptions.UninitializedMemory);
		filterIndices.Resize(0, NativeArrayOptions.UninitializedMemory);
		
		if (runAndMeasureConditionalCopyJob)
		{
			conditionalListCopyJobMarker.Begin();
			new ConditionalCopyJob<float3x4, GreaterThanZeroDel> { src = src, dst = dstData.AsParallelWriter(), Validator = default }.Schedule(src.Length, writeBatchCount).Complete();
			conditionalListCopyJobMarker.End();	
		}
		
		// Reset for next test
		dstData.Resize(0, NativeArrayOptions.UninitializedMemory);
		filterIndices.Resize(0, NativeArrayOptions.UninitializedMemory);
		
		if (runAndMeasureFilterJob)
		{
			filterJobMarker.Begin();
			FilterCopy<float3x4, GreaterThanZeroDel>.Schedule(src, dstData, filterIndices, writeBatchCount).Complete();
			filterJobMarker.End();
		}
		
		// Reset for next test
		dstData.Resize(src.Length, NativeArrayOptions.UninitializedMemory);

		if (runAndMeasureBasicCopyJob)
		{
			basicCopyJobMarker.Begin();
			new CopyJob<float3x4> { src = src, dst = dstData.AsArray() }.Schedule().Complete();
			basicCopyJobMarker.End();	
		}
		
		// Reset for next test
		
		if (useScheduleUtility)
		{
			if (useGPUBuffer) // Array
			{
				var dst = gpuBuffer.BeginWrite<float3x4>(0, dataLength);
				handle = src.IfCopyToParallel<float3x4, GreaterThanZeroDel>(dst, out tempCounter, indexingBatchCount, writeBatchCount, default,
					useScheduleUtilityPreAllocatedCollections ? indices : default,
					useScheduleUtilityPreAllocatedCollections ? counts : default);
				tempCounter.Dispose(handle);
			}
			else // List
			{
				handle = src.IfCopyToParallel<float3x4, GreaterThanZeroDel>(dstData, indexingBatchCount, writeBatchCount, default,
					useScheduleUtilityPreAllocatedCollections ? indices : default,
					useScheduleUtilityPreAllocatedCollections ? counts : default);
			}
			
			if (!completeInLateUpdate)
				handle.Complete();
		}
		else
		{
			var writer = useGPUBuffer ? 
				new DataRW<float3x4>(src, gpuBuffer.BeginWrite<float3x4>(0, dataLength)) : 
				new DataRW<float3x4>(src, dstData.AsArray());
			
			var copyJob = new ParallelConditionalCopyJob<float3x4, DataRW<float3x4>>(writer, indices, counts);

			indexingSumJobMarker.Begin();
			handle = ParallelIndexingSumJob<float3x4, GreaterThanZeroDel>.Schedule(src, indices, counts, counter, indexingBatchCount);
			if (!completeInLateUpdate)
				handle.Complete();
			indexingSumJobMarker.End();
			
			parallelCopyJobMarker.Begin();
			handle = copyJob.Schedule(indices.Length, writeBatchCount, handle);
			
			if (!completeInLateUpdate)
				handle.Complete();
			parallelCopyJobMarker.End();
		}

		if (!completeInLateUpdate)
			EndGPUWrite();
	}

	private void LateUpdate()
	{
		if (!completeInLateUpdate)
			return;

		handle.Complete();
		EndGPUWrite();
	}

	private void EndGPUWrite()
	{
		if (useGPUBuffer)
			gpuBuffer.EndWrite<float3x4>(tempCounter.IsCreated ? tempCounter.Value : dataLength);
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
		if (tempCounter.IsCreated)
			tempCounter.Dispose();
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
	
	public struct GreaterThanZeroDel : IValidator<float3x4>
	{
		public bool Validate(int index, float3x4 element) => element.c0.x > 0;
	}

	[BurstCompile(CompileSynchronously = true)]
	private struct CopyJob<T> : IJob where T : unmanaged
	{
		[ReadOnly] public NativeArray<T> src;
		[WriteOnly] public NativeArray<T> dst;

		public void Execute() => src.CopyTo(dst);
	}
}
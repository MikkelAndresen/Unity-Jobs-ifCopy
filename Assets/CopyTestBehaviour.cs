using UnityEngine;
using Unity.Mathematics;
using Unity.Collections;
using Unity.Jobs;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Profiling;

public class CopyTestBehaviour : MonoBehaviour
{
	private struct GreaterThanZeroDel : IValidator<float3x4>
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

	[SerializeField]
	private int dataLength = 100;
	[SerializeField]
	private bool useParallelCopy;
	[SerializeField]
	private bool testCopier;
	[SerializeField]
	private int testCopyRange = 10;
	[SerializeField]
	private bool useGPUBuffer;
	[SerializeField]
	private int indexingBatchCount = 2;
	[SerializeField]
	private int copyBatchCount = 2;
	[SerializeField]
	private TestDataType dataGenMethod = TestDataType.Odd;

	private NativeArray<float3x4> src;
	private NativeArray<float3x4> dstData;
	private NativeReference<int> counter;
	private NativeArray<int> counts;
	private NativeArray<BitField64> bits;
	private ComputeBuffer gpuBuffer;

	private void Start()
	{
		src = new NativeArray<float3x4>(dataLength, Allocator.Persistent);
		bits = new NativeArray<BitField64>((int)math.ceil(dataLength / 64f), Allocator.Persistent);
		dstData = new NativeArray<float3x4>(dataLength, Allocator.Persistent);
		counter = new NativeReference<int>(0, Allocator.Persistent);
		counts = new NativeArray<int>(bits.Length, Allocator.Persistent);
		
		for (int i = 0; i < src.Length; i++)
			src[i] = GetData(i);

		if (useGPUBuffer)
			gpuBuffer = new ComputeBuffer(dataLength, UnsafeUtility.SizeOf(typeof(float3x4)), ComputeBufferType.Default, ComputeBufferMode.SubUpdates);
	}

	private float3x4 GetData(int i) => 
	dataGenMethod switch
	{
		TestDataType.None => -1,
		TestDataType.All => 1,
		TestDataType.Odd => i % 2 == 0 ? -1 : 1,
		TestDataType.Half => i > 50 ? 1 : -1,
		_ => default,
	};

	private static readonly ProfilerMarker indexingJobMarker = new ProfilerMarker(nameof(ConditionIndexingJob<float3x4, GreaterThanZeroDel>));
	private static readonly ProfilerMarker mergeJobMarker = new ProfilerMarker(nameof(ConditionalCopyMergeJob<float3x4, GenericWriter<float3x4>>));
	private static readonly ProfilerMarker indexingSumJobMarker = new ProfilerMarker(nameof(ParallelIndexingSumJob<float3x4, GreaterThanZeroDel>));
	private static readonly ProfilerMarker parallelCopyJobMarker = new ProfilerMarker(nameof(ParallelConditionalCopyJob<float3x4, GenericWriter<float3x4>>));
	// private static readonly ProfilerMarker testCopyMarker = new ProfilerMarker(nameof(SimdFloatWriter<float3x4>));
	// private static readonly ProfilerMarker simpleTestCopyMarker = new ProfilerMarker(nameof(GenericSimpleWriter<float3x4>));
	
	private void Update()
	{
		counter.Value = 0;
		NativeArray<float3x4> dst = useGPUBuffer ? gpuBuffer.BeginWrite<float3x4>(0, dataLength) : dstData;
		GenericWriter<float3x4> writer = new GenericWriter<float3x4>(src, dst);
		
		if (useParallelCopy)
		{
			indexingSumJobMarker.Begin();
			ParallelIndexingSumJob<float3x4, GreaterThanZeroDel>.Schedule(src, bits, counts, counter, out var indexSumJob, indexingBatchCount).Complete();
			indexingSumJobMarker.End();
			parallelCopyJobMarker.Begin();
			ParallelConditionalCopyJob<float3x4, GenericWriter<float3x4>>.Schedule(indexSumJob, writer, copyBatchCount).Complete();
			parallelCopyJobMarker.End();
		} else
		{
			indexingJobMarker.Begin();
			ConditionIndexingJob<float3x4, GreaterThanZeroDel>.Schedule(src, bits, out var indexingJob, indexingBatchCount).Complete();
			indexingJobMarker.End();
			mergeJobMarker.Begin();
			ConditionalCopyMergeJob<float3x4, GenericWriter<float3x4>>.Schedule(indexingJob, writer, counter).Complete();
			mergeJobMarker.End();
		}

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
		gpuBuffer.Dispose();
	}
}
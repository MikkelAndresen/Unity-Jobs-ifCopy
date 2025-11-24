using NUnit.Framework;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.PerformanceTesting;
using UnityEngine;

public class PerformanceTest
{
	private NativeArray<float3x4> src;
	private NativeList<float3x4> dstData;
	private NativeReference<int> tempCounter;
	private NativeArray<int> counts;
	private NativeArray<BitField64> indices;
	private ComputeBuffer gpuBuffer;
	private const int IndexingBatchCount = 1024;
	private const int WriteBatchCount = 16384;
	private const int DataLength = 10_000_000;

	[SetUp]
	public void SetUp()
	{
		Unity.Jobs.LowLevel.Unsafe.JobsUtility.JobDebuggerEnabled = false;
		Unity.Jobs.LowLevel.Unsafe.JobsUtility.JobCompilerEnabled = true;
		
		src = new NativeArray<float3x4>(DataLength, Allocator.Persistent);
		indices = new NativeArray<BitField64>((int)math.ceil(DataLength / 64f), Allocator.Persistent);
		dstData = new NativeList<float3x4>(DataLength, Allocator.Persistent);
		counts = new NativeArray<int>(indices.Length, Allocator.Persistent);
		const TestDataType testDataType = TestDataType.All;
		for (int i = 0; i < src.Length; i++)
			src[i] = GetData(testDataType, i);

		gpuBuffer = new ComputeBuffer(DataLength, UnsafeUtility.SizeOf(typeof(float3x4)), ComputeBufferType.Default,
			ComputeBufferMode.SubUpdates);
	}

	[TearDown]
	public void TearDown()
	{
		if (src.IsCreated)
			src.Dispose();
		if (dstData.IsCreated)
			dstData.Dispose();
		if (tempCounter.IsCreated)
			tempCounter.Dispose();
		if (counts.IsCreated)
			counts.Dispose();
		if (indices.IsCreated)
			indices.Dispose();
		if(gpuBuffer != null && gpuBuffer.IsValid())
			gpuBuffer.Dispose();
	}
	
	[Test, Performance]
	public void CPUMemNoAlloc()
	{
		Measure.Method(() =>
		{
			var handle = src.IfCopyToParallel<float3x4, GreaterThanZeroDel>(dstData, IndexingBatchCount, WriteBatchCount, default,
				indices,
				counts);
			handle.Complete();
		}).WarmupCount(3).MeasurementCount(20).Run();
	}
	
	[Test, Performance]
	public void CPUMem()
	{
		Measure.Method(() =>
		{
			var handle = src.IfCopyToParallel<float3x4, GreaterThanZeroDel>(dstData, IndexingBatchCount, WriteBatchCount);
			handle.Complete();
		}).WarmupCount(3).MeasurementCount(20).Run();
	}

	[Test, Performance]
	public void GPUMemNoAlloc()
	{
		Measure.Method(() =>
		{
			var dst = gpuBuffer.BeginWrite<float3x4>(0, DataLength);
			var handle = src.IfCopyToParallel<float3x4, GreaterThanZeroDel>(dst, out tempCounter, IndexingBatchCount, WriteBatchCount, default,
				indices,
				counts);
			handle.Complete();
			gpuBuffer.EndWrite<float3x4>(tempCounter.Value);
			tempCounter.Dispose();
		}).WarmupCount(3).MeasurementCount(20).Run();
	}
	
	[Test, Performance]
	public void GPUMem()
	{
		Measure.Method(() =>
		{
			var dst = gpuBuffer.BeginWrite<float3x4>(0, DataLength);
			var handle = src.IfCopyToParallel<float3x4, GreaterThanZeroDel>(dst, out tempCounter, IndexingBatchCount, WriteBatchCount);
			handle.Complete();
			gpuBuffer.EndWrite<float3x4>(tempCounter.Value);
			tempCounter.Dispose();
		}).WarmupCount(3).MeasurementCount(20).Run();
	}
	
	[Test, Performance]
	public void GPUMemNoCondition()
	{
		Measure.Method(() =>
		{
			var dst = gpuBuffer.BeginWrite<float3x4>(0, DataLength);
			var handle = new CopyJob<float3x4>() { src = src, dst = dst }.Schedule();
			handle.Complete();
			gpuBuffer.EndWrite<float3x4>(DataLength);
		}).WarmupCount(3).MeasurementCount(20).Run();
	}
	
	[Test, Performance]
	public void CPUMemNoCondition()
	{
		Measure.Method(() =>
		{
			var handle = new CopyJob<float3x4>() { src = src, dst = dstData.AsArray() }.Schedule();
			handle.Complete();
		}).WarmupCount(3).MeasurementCount(20).Run();
	}
	
	public struct GreaterThanZeroDel : IValidator<float3x4>
	{
		public bool Validate(int index, float3x4 element) => element.c0.x > 0;
	}
	
	private enum TestDataType
	{
		None,
		All,
		Odd,
		Half
	}
	
	private static float GetData(TestDataType dataGenMethod, int i) =>
		dataGenMethod switch
		{
			TestDataType.None => -1,
			TestDataType.All => 1,
			TestDataType.Odd => i % 2 == 0 ? -1 : 1,
			TestDataType.Half => i > 50 ? 1 : -1,
			_ => default,
		};
	
	[BurstCompile(CompileSynchronously = true)]
	private struct CopyJob<T> : IJob where T : unmanaged
	{
		public NativeArray<T> src, dst;
		
		public void Execute() => src.CopyTo(dst);
	}
}
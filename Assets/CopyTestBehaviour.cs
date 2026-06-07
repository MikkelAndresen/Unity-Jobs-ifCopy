using System.Runtime.CompilerServices;
using Unity.Burst;
using UnityEngine;
using Unity.Mathematics;
using Unity.Collections;
using Unity.Jobs;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Profiling;

/// <summary>
/// Abstract base used to run conditional-copy jobs in Update so prolonged performance shows up in the profiler.
/// Concrete subclasses pick the element stride (and a matching validator) so the benchmark can sweep element size
/// — the predicate:copy cost ratio shifts dramatically across strides and changes which technique wins.
/// </summary>
public abstract class CopyTestBehaviour<T, TValidator> : MonoBehaviour
	where T : unmanaged
	where TValidator : unmanaged, IBatchValidator<T>
{
	[SerializeField] private int dataLength = 100;
	[SerializeField] private bool useGPUBuffer;
	[SerializeField] private int indexingBatchCount = 1024;
	[SerializeField] private int writeBatchCount = 1024;
	[SerializeField] private bool completeInLateUpdate = false;
	[SerializeField] private bool useScheduleUtility;
	[SerializeField] private bool useScheduleUtilityPreAllocatedCollections;
	[SerializeField] private TestDataType dataGenMethod = TestDataType.Odd;
	[SerializeField] private int segmentSpacing = 64;
	[SerializeField] private int segmentLength = 8;
	[SerializeField] private bool runAndMeasureBasicCopyJob;
	[SerializeField] private bool runAndMeasureConditionalCopyJob;
	[SerializeField] private bool runAndMeasureFilterJob;

	private NativeArray<T> src;
	private NativeList<T> dstData;
	private NativeReference<int> counter;
	private NativeReference<int> tempCounter;
	private NativeArray<int> counts;
	private NativeArray<BitField64> indices;
	private NativeList<int> filterIndices;
	private ComputeBuffer gpuBuffer;
	private JobHandle handle;

	private static readonly ProfilerMarker indexingSumJobMarker =
		new($"ParallelIndexingSumJob<{typeof(T).Name}>");
	private static readonly ProfilerMarker parallelCopyJobMarker =
		new($"ParallelConditionalCopyJob<{typeof(T).Name}>");
	private static readonly ProfilerMarker basicCopyJobMarker =
		new($"CopyJob<{typeof(T).Name}>");
	private static readonly ProfilerMarker conditionalListCopyJobMarker =
		new($"ConditionalCopyJob<{typeof(T).Name}>");
	private static readonly ProfilerMarker filterJobMarker =
		new($"FilterCopy<{typeof(T).Name}>");

	private void Start()
	{
		src = new NativeArray<T>(dataLength, Allocator.Persistent);
		indices = new NativeArray<BitField64>((int)math.ceil(dataLength / 64f), Allocator.Persistent);
		dstData = new NativeList<T>(dataLength, Allocator.Persistent);
		counts = new NativeArray<int>(indices.Length, Allocator.Persistent);
		counter = new NativeReference<int>(Allocator.Persistent);
		filterIndices = new NativeList<int>(dataLength, Allocator.Persistent);

		for (int i = 0; i < src.Length; i++)
			src[i] = MakeElement(i, ShouldKeep(i));

		gpuBuffer = new ComputeBuffer(dataLength, UnsafeUtility.SizeOf<T>(), ComputeBufferType.Default,
			ComputeBufferMode.SubUpdates);
	}

	private void Update()
	{
		counter.Value = 0;

		dstData.Resize(0, NativeArrayOptions.UninitializedMemory);
		filterIndices.Resize(0, NativeArrayOptions.UninitializedMemory);

		if (runAndMeasureConditionalCopyJob)
		{
			conditionalListCopyJobMarker.Begin();
			new ConditionalCopyJob<T, TValidator> { src = src, dst = dstData.AsParallelWriter(), Validator = default }
				.Schedule(src.Length, writeBatchCount).Complete();
			conditionalListCopyJobMarker.End();
		}

		dstData.Resize(0, NativeArrayOptions.UninitializedMemory);
		filterIndices.Resize(0, NativeArrayOptions.UninitializedMemory);

		if (runAndMeasureFilterJob)
		{
			filterJobMarker.Begin();
			FilterCopy<T, TValidator>.Schedule(src, dstData, filterIndices, writeBatchCount).Complete();
			filterJobMarker.End();
		}

		dstData.Resize(src.Length, NativeArrayOptions.UninitializedMemory);

		if (runAndMeasureBasicCopyJob)
		{
			basicCopyJobMarker.Begin();
			new CopyJob<T> { src = src, dst = dstData.AsArray() }.Schedule().Complete();
			basicCopyJobMarker.End();
		}

		if (useScheduleUtility)
		{
			parallelCopyJobMarker.Begin();

			if (useGPUBuffer)
			{
				var dst = gpuBuffer.BeginWrite<T>(0, dataLength);
				handle = src.IfCopyToParallel<T, TValidator>(dst, out tempCounter, indexingBatchCount, writeBatchCount, default,
					useScheduleUtilityPreAllocatedCollections ? indices : default,
					useScheduleUtilityPreAllocatedCollections ? counts : default);
				tempCounter.Dispose(handle);
			}
			else
			{
				handle = src.IfCopyToParallel<T, TValidator>(dstData, indexingBatchCount, writeBatchCount, default,
					useScheduleUtilityPreAllocatedCollections ? indices : default,
					useScheduleUtilityPreAllocatedCollections ? counts : default);
			}

			if (!completeInLateUpdate)
				handle.Complete();
			parallelCopyJobMarker.End();
		}
		else
		{
			var writer = useGPUBuffer
				? new DataRW<T>(src, gpuBuffer.BeginWrite<T>(0, dataLength))
				: new DataRW<T>(src, dstData.AsArray());

			var copyJob = new ParallelConditionalCopyJob<T, DataRW<T>>(writer, indices, counts);

			parallelCopyJobMarker.Begin();
			indexingSumJobMarker.Begin();
			handle = ParallelIndexingSumJob<T, TValidator>.Schedule(src, indices, counts, counter, indexingBatchCount);
			if (!completeInLateUpdate)
				handle.Complete();
			indexingSumJobMarker.End();

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

		parallelCopyJobMarker.Begin();
		handle.Complete();
		parallelCopyJobMarker.End();
		EndGPUWrite();
	}

	private void EndGPUWrite()
	{
		if (useGPUBuffer)
			gpuBuffer.EndWrite<T>(tempCounter.IsCreated ? tempCounter.Value : dataLength);
	}

	private bool ShouldKeep(int i) =>
		dataGenMethod switch
		{
			TestDataType.None => false,
			TestDataType.All => true,
			TestDataType.Odd => i % 2 != 0,
			TestDataType.Half => i > 50,
			TestDataType.Segments => (i % math.max(segmentSpacing, 1)) < segmentLength,
			_ => false,
		};

	/// <summary>Produce a concrete element whose validator-visible state matches <paramref name="keep"/>.</summary>
	protected abstract T MakeElement(int i, bool keep);

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
		Half,
		Segments
	}

	[BurstCompile(CompileSynchronously = true)]
	private struct CopyJob<TElem> : IJob where TElem : unmanaged
	{
		[ReadOnly] public NativeArray<TElem> src;
		[WriteOnly] public NativeArray<TElem> dst;

		public void Execute() => src.CopyTo(dst);
	}
}

/// <summary>
/// Generic predicate that checks the first byte of an element. Lets struct strides reuse the same "keep" encoding
/// (first byte = 0 to skip, non-zero to keep) without writing per-type SIMD code. Scalar only — the AVX/SSE/NEON
/// path lives on the byte-specific validator since stride-1 is what those intrinsics naturally compress.
/// </summary>
[BurstCompile]
public unsafe struct FirstByteNonZeroValidator<T> : IBatchValidator<T> where T : unmanaged
{
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public bool Validate(int index, T element) => *(byte*)&element != 0;

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public BitField64 Validate(in NativeSlice<T> elements)
	{
		ulong mask = 0;
		int stride = elements.Stride;
		byte* p = (byte*)elements.GetUnsafeReadOnlyPtr();
		for (int i = 0; i < 64; i++)
			if (p[i * stride] != 0) mask |= 1UL << i;
		return new BitField64 { Value = mask };
	}
}

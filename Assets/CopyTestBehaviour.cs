using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Burst.CompilerServices;
using Unity.Burst.Intrinsics;
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
	[SerializeField] private int segmentSpacing = 64;
	[SerializeField] private int segmentLength = 8;
	[SerializeField] private bool runAndMeasureBasicCopyJob;
	[SerializeField] private bool runAndMeasureConditionalCopyJob;
	[SerializeField] private bool runAndMeasureFilterJob;

	private NativeArray<byte> src;
	private NativeList<byte> dstData;
	private NativeReference<int> counter;
	private NativeReference<int> tempCounter;
	private NativeArray<int> counts;
	private NativeArray<BitField64> indices;
	private NativeList<int> filterIndices;
	private ComputeBuffer gpuBuffer;
	private JobHandle handle;

	private static readonly ProfilerMarker indexingSumJobMarker =
		new (nameof(ParallelIndexingSumJob<byte, GreaterThanZeroDel>));
	private static readonly ProfilerMarker parallelCopyJobMarker =
		new (nameof(ParallelConditionalCopyJob<byte, DataRW<byte>>));
	private static readonly ProfilerMarker basicCopyJobMarker =
		new (nameof(CopyJob<byte>));
	private static readonly ProfilerMarker conditionalListCopyJobMarker =
		new (nameof(ConditionalCopyJob<byte, GreaterThanZeroDel>));
	private static readonly ProfilerMarker filterJobMarker =
		new (nameof(FilterCopy<byte, GreaterThanZeroDel>));

	private void Start()
	{
		src = new NativeArray<byte>(dataLength, Allocator.Persistent);
		indices = new NativeArray<BitField64>((int)math.ceil(dataLength / 64f), Allocator.Persistent);
		dstData = new NativeList<byte>(dataLength, Allocator.Persistent);
		counts = new NativeArray<int>(indices.Length, Allocator.Persistent);
		counter = new NativeReference<int>(Allocator.Persistent);
		filterIndices = new NativeList<int>(dataLength, Allocator.Persistent);

		for (int i = 0; i < src.Length; i++)
			src[i] = GetData(i);

		gpuBuffer = new ComputeBuffer(dataLength, UnsafeUtility.SizeOf(typeof(byte)), ComputeBufferType.Default,
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
			new ConditionalCopyJob<byte, GreaterThanZeroDel> { src = src, dst = dstData.AsParallelWriter(), Validator = default }.Schedule(src.Length, writeBatchCount).Complete();
			conditionalListCopyJobMarker.End();
		}

		// Reset for next test
		dstData.Resize(0, NativeArrayOptions.UninitializedMemory);
		filterIndices.Resize(0, NativeArrayOptions.UninitializedMemory);

		if (runAndMeasureFilterJob)
		{
			filterJobMarker.Begin();
			FilterCopy<byte, GreaterThanZeroDel>.Schedule(src, dstData, filterIndices, writeBatchCount).Complete();
			filterJobMarker.End();
		}

		// Reset for next test
		dstData.Resize(src.Length, NativeArrayOptions.UninitializedMemory);

		if (runAndMeasureBasicCopyJob)
		{
			basicCopyJobMarker.Begin();
			new CopyJob<byte> { src = src, dst = dstData.AsArray() }.Schedule().Complete();
			basicCopyJobMarker.End();
		}

		// Reset for next test

		if (useScheduleUtility)
		{
			parallelCopyJobMarker.Begin();

			if (useGPUBuffer) // Array
			{
				var dst = gpuBuffer.BeginWrite<byte>(0, dataLength);
				handle = src.IfCopyToParallel<byte, GreaterThanZeroDel>(dst, out tempCounter, indexingBatchCount, writeBatchCount, default,
					useScheduleUtilityPreAllocatedCollections ? indices : default,
					useScheduleUtilityPreAllocatedCollections ? counts : default);
				tempCounter.Dispose(handle);
			}
			else // List
			{
				handle = src.IfCopyToParallel<byte, GreaterThanZeroDel>(dstData, indexingBatchCount, writeBatchCount, default,
					useScheduleUtilityPreAllocatedCollections ? indices : default,
					useScheduleUtilityPreAllocatedCollections ? counts : default);
			}

			if (!completeInLateUpdate)
				handle.Complete();
			parallelCopyJobMarker.End();
		}
		else
		{
			var writer = useGPUBuffer ?
				new DataRW<byte>(src, gpuBuffer.BeginWrite<byte>(0, dataLength)) :
				new DataRW<byte>(src, dstData.AsArray());

			var copyJob = new ParallelConditionalCopyJob<byte, DataRW<byte>>(writer, indices, counts);

			parallelCopyJobMarker.Begin();
			indexingSumJobMarker.Begin();
			handle = ParallelIndexingSumJob<byte, GreaterThanZeroDel>.Schedule(src, indices, counts, counter, indexingBatchCount);
			if (!completeInLateUpdate)
				handle.Complete();
			indexingSumJobMarker.End();

			handle = copyJob.Schedule(indices.Length, writeBatchCount, handle);
			// handle = copyJob.Schedule(indices.Length, handle);

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
			gpuBuffer.EndWrite<byte>(tempCounter.IsCreated ? tempCounter.Value : dataLength);
	}

	protected byte GetData(int i) =>
		dataGenMethod switch
		{
			TestDataType.None => 0,
			TestDataType.All => 1,
			TestDataType.Odd => (byte)(i % 2 == 0 ? 0 : 1),
			TestDataType.Half => (byte)(i > 50 ? 1 : 0),
			TestDataType.Segments => (byte)((i % math.max(segmentSpacing, 1)) < segmentLength ? 1 : 0),
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
		Half,
		Segments
	}

	[BurstCompile]
	public struct GreaterThanZeroDel : IBatchValidator<byte>
	{
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public bool Validate(int index, byte element) => element > 0;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public unsafe BitField64 Validate(in NativeSlice<byte> elements)
		{
			Hint.Assume(elements.Length == 64);
			Hint.Assume(elements.Stride == sizeof(byte));
			byte* p = (byte*)elements.GetUnsafeReadOnlyPtr();

			ulong mask;
			if (X86.Avx2.IsAvx2Supported)
				mask = NonZeroMask64Avx2(p);
			else if (X86.Sse2.IsSse2Supported)
				mask = NonZeroMask64Sse2(p);
			else if (Arm.Neon.IsNeonSupported)
				mask = NonZeroMask64Neon(p);
			else
				mask = NonZeroMask64Scalar(p);

			return new BitField64 { Value = mask };
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		private static unsafe ulong NonZeroMask64Scalar(byte* p)
		{
			ulong mask = 0;
			for (int i = 0; i < 64; i++)
				if (p[i] != 0) mask |= 1UL << i;
			return mask;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		private static unsafe ulong NonZeroMask64Avx2(byte* p)
		{
			// Inner guard is required, not just the dispatcher's: Burst compiles each method
			// for every target, and only DCE's intrinsics inside a same-method IsXxxSupported branch.
			if (X86.Avx2.IsAvx2Supported)
			{
				// Two 32-byte loads, compare each byte against zero, movemask, invert.
				v256 zero = X86.Avx.mm256_setzero_si256();
				v256 a = X86.Avx.mm256_loadu_si256(p);
				v256 b = X86.Avx.mm256_loadu_si256(p + 32);
				// movemask returns 32 set-where-byte==0 bits; cast through uint to avoid sign extension.
				uint zeroLo = (uint)X86.Avx2.mm256_movemask_epi8(X86.Avx2.mm256_cmpeq_epi8(a, zero));
				uint zeroHi = (uint)X86.Avx2.mm256_movemask_epi8(X86.Avx2.mm256_cmpeq_epi8(b, zero));
				return ~(((ulong)zeroHi << 32) | zeroLo);
			}
			return 0;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		private static unsafe ulong NonZeroMask64Sse2(byte* p)
		{
			if (X86.Sse2.IsSse2Supported)
			{
				// Four 16-byte blocks, cmpeq vs zero, movemask, pack & invert.
				v128 zero = X86.Sse2.setzero_si128();
				ulong zeroMask = 0;
				for (int i = 0; i < 4; i++)
				{
					v128 v = X86.Sse2.loadu_si128(p + i * 16);
					uint mm = (ushort)X86.Sse2.movemask_epi8(X86.Sse2.cmpeq_epi8(v, zero));
					zeroMask |= (ulong)mm << (i * 16);
				}
				return ~zeroMask;
			}
			return 0;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		private static unsafe ulong NonZeroMask64Neon(byte* p)
		{
			if (Arm.Neon.IsNeonSupported)
			{
				// vtstq_u8(v,v) yields 0xFF per lane where v != 0. AND with per-lane bit-position
				// constants, then pairwise-add up to u64 to get (bits 0..7, bits 8..15) of each
				// 16-byte block's submask. vpaddlq_* works on ARMv7+AArch64 (vaddv_u8 is AArch64-only).
				v128 bitMask = new v128(
					1, 2, 4, 8, 16, 32, 64, 128,
					1, 2, 4, 8, 16, 32, 64, 128);

				ulong mask = 0;
				for (int b = 0; b < 4; b++)
				{
					v128 v = Arm.Neon.vld1q_u8(p + b * 16);
					v128 cmp = Arm.Neon.vtstq_u8(v, v);
					v128 masked = Arm.Neon.vandq_u8(cmp, bitMask);
					v128 sum16 = Arm.Neon.vpaddlq_u8(masked);
					v128 sum32 = Arm.Neon.vpaddlq_u16(sum16);
					v128 sum64 = Arm.Neon.vpaddlq_u32(sum32);
					ulong lo = Arm.Neon.vgetq_lane_u64(sum64, 0);
					ulong hi = Arm.Neon.vgetq_lane_u64(sum64, 1);
					mask |= (lo | (hi << 8)) << (b * 16);
				}
				return mask;
			}
			return 0;
		}
	}

	[BurstCompile(CompileSynchronously = true)]
	private struct CopyJob<T> : IJob where T : unmanaged
	{
		[ReadOnly] public NativeArray<T> src;
		[WriteOnly] public NativeArray<T> dst;

		public void Execute() => src.CopyTo(dst);
	}
}
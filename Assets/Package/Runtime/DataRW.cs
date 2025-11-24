using System;
using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
#if UNITY_BURST_EXPERIMENTAL_PREFETCH_INTRINSIC
using Unity.Burst.Intrinsics;
#endif

[GenerateTestsForBurstCompatibility, BurstCompile]
public unsafe struct DataRW<T> : IIndexWriter<T>, IIndexReader<T> where T : unmanaged
{
	[ReadOnly, NativeDisableParallelForRestriction]
	private NativeArray<T> src;

	[WriteOnly, NativeDisableParallelForRestriction]
	private NativeArray<T> dst;

	[ReadOnly, NativeDisableUnsafePtrRestriction]
	private readonly T* srcPtr;

	[WriteOnly, NativeDisableUnsafePtrRestriction]
	private readonly T* dstPtr;

	[ReadOnly] private static readonly int Stride;
	static DataRW() => Stride = UnsafeUtility.SizeOf<T>();

	public DataRW(NativeArray<T> src, NativeArray<T> dst) : this(src, dst, (T*)src.GetUnsafeReadOnlyPtr(),
		(T*)dst.GetUnsafeReadOnlyPtr())
	{
	}

	public DataRW(NativeArray<T> src, NativeArray<T> dst, T* srcReadOnlyPtr, T* dstReadOnlyPtr)
	{
		this.dst = dst;
		this.src = src;
		srcPtr = srcReadOnlyPtr;
		dstPtr = dstReadOnlyPtr;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public T Read(int index) => src[index];

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public void Write(int startIndex, in ReadOnlySpan<T> values, int length)
	{
#if UNITY_BURST_EXPERIMENTAL_PREFETCH_INTRINSIC
		PrefetchSrc(startIndex);
#endif
		fixed (T* ptr = values)
			UnsafeUtility.MemCpy(dstPtr + startIndex, ptr, length * Stride);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public void Write(int dstIndex, int srcIndex) => dst[dstIndex] = src[srcIndex];

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public void Write(int dstIndex, int srcIndex, int srcRange)
	{
		for (int i = 0; i < srcRange; i++)
			dstPtr[dstIndex + i] = srcPtr[srcIndex + i];
	}

#if UNITY_BURST_EXPERIMENTAL_PREFETCH_INTRINSIC
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public readonly void Prefetch(int dstIndex, int srcIndex)
	{
		PrefetchDst(dstIndex);
		PrefetchSrc(srcIndex);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public readonly void PrefetchSrc(int index) => 
		Common.Prefetch(srcPtr + index, Common.ReadWrite.Read, Common.Locality.LowTemporalLocality);
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public readonly void PrefetchDst(int index) => 
		Common.Prefetch(dstPtr + index, Common.ReadWrite.Write, Common.Locality.HighTemporalLocality);
#endif
}
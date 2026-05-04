using System;
using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Burst.CompilerServices;
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

	[ReadOnly] private static readonly int stride;
	static DataRW() => stride = UnsafeUtility.SizeOf<T>();

	public int Stride => stride;
	
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
	public T Read([AssumeRange(0, int.MaxValue)] int index)
	{
		Hint.Assume(src.Length > 0);
		return src[index];
	}

	public NativeSlice<T> Read([AssumeRange(0, int.MaxValue)] int startIndex, [AssumeRange(0, 64)] int count) => src.Slice(startIndex, count);

	public void CopyTo([AssumeRange(0, int.MaxValue)] int startIndex, [AssumeRange(0, 64)] int count, Span<T> other)
	{
		fixed(T* ptr = other)
			UnsafeUtility.MemCpy(ptr, srcPtr + startIndex, Stride * count);
	}
	// public Span<T> ReadAsSpan(int startIndex, int count) => src.AsSpan().Slice(startIndex, count);
	
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public void Write([AssumeRange(0, int.MaxValue)] int startIndex, in ReadOnlySpan<T> values,
		[AssumeRange(0, 64)] int length)
	{
		Hint.Assume(src.Length > 0);
		Hint.Assume(dst.Length > 0);
		Hint.Assume(values.Length > 0);

#if UNITY_BURST_EXPERIMENTAL_PREFETCH_INTRINSIC
		PrefetchSrc(startIndex + values.Length);
#endif

		fixed (T* ptr = values)
			UnsafeUtility.MemCpy(dstPtr + startIndex, ptr, length * stride);
	}
	
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public void Write([AssumeRange(0, int.MaxValue)] int startIndex, in ReadOnlySpan<int> indices)
	{
		Hint.Assume(src.Length > 0);
		Hint.Assume(dst.Length > 0);

		for (int i = 0; i < indices.Length; i++)
		{
			dst[startIndex + i] = src[indices[i]];
		}
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public void Write([AssumeRange(0, int.MaxValue)] int dstIndex, [AssumeRange(0, int.MaxValue)] int srcIndex)
	{
		Hint.Assume(src.Length > 0);
		Hint.Assume(dst.Length > 0);

		dst[dstIndex] = src[srcIndex];
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public void Write([AssumeRange(0, int.MaxValue)] int dstIndex, [AssumeRange(0, int.MaxValue)] int srcIndex,
		[AssumeRange(0, int.MaxValue)] int srcRange)
	{
		Hint.Assume(src.Length > 0);
		Hint.Assume(dst.Length > 0);

		
#if UNITY_BURST_EXPERIMENTAL_PREFETCH_INTRINSIC
		PrefetchSrc(srcIndex + srcRange);
		PrefetchDst(dstIndex + srcRange);
#endif
		// for (int i = 0; i < srcRange; i++)
		// 	dstPtr[dstIndex + i] = srcPtr[srcIndex + i];

		UnsafeUtility.MemCpy(dstPtr + dstIndex, srcPtr + srcIndex, srcRange * stride);
	}

#if UNITY_BURST_EXPERIMENTAL_PREFETCH_INTRINSIC
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public readonly void Prefetch([AssumeRange(0, int.MaxValue)] int dstIndex,
		[AssumeRange(0, int.MaxValue)] int srcIndex)
	{
		PrefetchDst(dstIndex);
		PrefetchSrc(srcIndex);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public readonly void PrefetchSrc([AssumeRange(0, int.MaxValue)] int index) => 
		PrefetchRead(srcPtr, index);

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public readonly void PrefetchDst([AssumeRange(0, int.MaxValue)] int index) =>
		PrefetchWrite(dstPtr, index);
	
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static void PrefetchWrite(T* ptr, [AssumeRange(0, int.MaxValue)] int index) =>
		Common.Prefetch(ptr + index, Common.ReadWrite.Write, Common.Locality.HighTemporalLocality);
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public static void PrefetchRead(T* ptr, [AssumeRange(0, int.MaxValue)] int index) =>
		Common.Prefetch(ptr + index, Common.ReadWrite.Read, Common.Locality.LowTemporalLocality);
	
#endif
}
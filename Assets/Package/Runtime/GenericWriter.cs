using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

[GenerateTestsForBurstCompatibility, BurstCompile]
public unsafe struct GenericWriter<T> : IIndexWriter<T> where T : unmanaged
{
	[ReadOnly, NativeDisableParallelForRestriction]
	private NativeArray<T> src;

	[WriteOnly, NativeDisableParallelForRestriction]
	private NativeArray<T> dst;

	[ReadOnly, NativeDisableUnsafePtrRestriction]
	private readonly T* srcPtr;

	[WriteOnly, NativeDisableUnsafePtrRestriction]
	private readonly T* dstPtr;

	public GenericWriter(NativeArray<T> src, NativeArray<T> dst) : this(src, dst, (T*)src.GetUnsafeReadOnlyPtr(), (T*)dst.GetUnsafeReadOnlyPtr())
	{
	}

	public GenericWriter(NativeArray<T> src, NativeArray<T> dst, T* srcReadOnlyPtr, T* dstReadOnlyPtr)
	{
		this.dst = dst;
		this.src = src;
		srcPtr = srcReadOnlyPtr;
		dstPtr = dstReadOnlyPtr;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public void Write(int dstIndex, int srcIndex)
	{
		Prefetch(dstIndex, srcIndex);
		dst[dstIndex] = src[srcIndex];
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public void Write(int dstIndex, int srcIndex, int srcRange)
	{
		for (int i = 0; i < srcRange; i++)
			dstPtr[dstIndex + i] = srcPtr[srcIndex + i];
	}

	public readonly void Prefetch(int dstIndex, int srcIndex)
	{
#if UNITY_BURST_EXPERIMENTAL_PREFETCH_INTRINSIC
		Common.Prefetch(srcPtr + srcIndex, Common.ReadWrite.Read, Common.Locality.LowTemporalLocality);
		Common.Prefetch(dstPtr + dstIndex, Common.ReadWrite.Write, Common.Locality.HighTemporalLocality);
#endif
	}
}
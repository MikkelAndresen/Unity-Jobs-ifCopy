using System;
using Unity.Collections;
using Unity.Mathematics;

public interface IValidator<in T> where T : unmanaged
{
	bool Validate(int index, T element);
}

public interface IValidatorVectorized<in T> where T : unmanaged
{
	bool4 Validate(int4 indices, T element);
}

public interface IIndexWriter
{
	void Write(int dstIndex, int srcIndex);
	void Write(int dstIndex, int srcIndex, int srcRange);
	int Stride { get; }
#if UNITY_BURST_EXPERIMENTAL_PREFETCH_INTRINSIC
	void Prefetch(int dstIndex, int srcIndex);
	void PrefetchSrc(int index);
	void PrefetchDst(int index);
	#endif
}

public interface IIndexWriter<T> : IIndexWriter where T : unmanaged
{
	void Write(int startIndex, in ReadOnlySpan<T> values, int length);
}

public interface IIndexReader<T> : IIndexWriter where T : unmanaged
{
	T Read(int index);
	NativeSlice<T> Read(int startIndex, int count);

	void CopyTo(int startIndex, int count, Span<T> other);
	// Span<T> ReadAsSpan(int startIndex, int count);
}
	
public interface IConditionalCopyJob<T, W> where T : unmanaged where W : struct, IIndexWriter<T>
{
}

public interface IConditionalIndexingJob<T, M> where T : unmanaged where M : IValidator<T>
{
}
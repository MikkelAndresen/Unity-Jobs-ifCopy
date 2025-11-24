using System;
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
	public void Write(int dstIndex, int srcIndex);
	public void Write(int dstIndex, int srcIndex, int srcRange);
#if UNITY_BURST_EXPERIMENTAL_PREFETCH_INTRINSIC
	public void Prefetch(int dstIndex, int srcIndex);
	public void PrefetchSrc(int index);
	public void PrefetchDst(int index);
	#endif
}

public interface IIndexWriter<T> : IIndexWriter where T : unmanaged
{
	public void Write(int startIndex, in ReadOnlySpan<T> values, int length);
}

public interface IIndexReader<out T> : IIndexWriter where T : unmanaged
{
	public T Read(int index);
}
	
public interface IConditionalCopyJob<T, W> where T : unmanaged where W : struct, IIndexWriter<T>
{
}

public interface IConditionalIndexingJob<T, M> where T : unmanaged where M : IValidator<T>
{
}
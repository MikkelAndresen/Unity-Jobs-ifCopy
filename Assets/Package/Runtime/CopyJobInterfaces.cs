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
	public void Prefetch(int dstIndex, int srcIndex);
}

public interface IIndexWriter<T> : IIndexWriter where T : unmanaged
{
}

public interface IConditionalCopyJob<T, W> where T : unmanaged where W : struct, IIndexWriter<T>
{
}

public interface IConditionalIndexingJob<T, M> where T : unmanaged where M : IValidator<T>
{
}
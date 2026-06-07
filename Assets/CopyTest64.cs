using System.Runtime.InteropServices;

/// <summary>64-byte stride. One cache line per element — copy bandwidth starts to dominate predicate cost here.</summary>
public sealed class CopyTest64 : CopyTestBehaviour<CopyTest64.Element64, FirstByteNonZeroValidator<CopyTest64.Element64>>
{
	protected override Element64 MakeElement(int i, bool keep) => new() { keepFlag = (byte)(keep ? 1 : 0) };

	[StructLayout(LayoutKind.Sequential, Size = 64)]
	public struct Element64
	{
		public byte keepFlag;
	}
}

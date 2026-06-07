using System.Runtime.InteropServices;

/// <summary>16-byte stride. SIMD-friendly element size, fits one AVX2 lane pair per 2 elements.</summary>
public sealed class CopyTest16 : CopyTestBehaviour<CopyTest16.Element16, FirstByteNonZeroValidator<CopyTest16.Element16>>
{
	protected override Element16 MakeElement(int i, bool keep) => new() { keepFlag = (byte)(keep ? 1 : 0) };

	[StructLayout(LayoutKind.Sequential, Size = 16)]
	public struct Element16
	{
		public byte keepFlag;
	}
}

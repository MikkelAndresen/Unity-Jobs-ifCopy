using System.Runtime.InteropServices;

/// <summary>256-byte stride. DRAM-bandwidth-bound case — every technique converges toward the memcpy floor.</summary>
public sealed class CopyTest256 : CopyTestBehaviour<CopyTest256.Element256, FirstByteNonZeroValidator<CopyTest256.Element256>>
{
	protected override Element256 MakeElement(int i, bool keep) => new() { keepFlag = (byte)(keep ? 1 : 0) };

	[StructLayout(LayoutKind.Sequential, Size = 256)]
	public struct Element256
	{
		public byte keepFlag;
	}
}

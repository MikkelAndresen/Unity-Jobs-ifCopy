using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Burst.CompilerServices;
using Unity.Burst.Intrinsics;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

/// <summary>1-byte element variant. SIMD compresses naturally at stride 1; this is the best case for IJobFilter.</summary>
public sealed class CopyTestByte : CopyTestBehaviour<byte, CopyTestByte.GreaterThanZeroDel>
{
	protected override byte MakeElement(int i, bool keep) => (byte)(keep ? 1 : 0);

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
				v256 zero = X86.Avx.mm256_setzero_si256();
				v256 a = X86.Avx.mm256_loadu_si256(p);
				v256 b = X86.Avx.mm256_loadu_si256(p + 32);
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
}

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using NUnit.Framework;
using Unity.Burst.Intrinsics;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;

namespace Tests
{
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

		public GenericWriter(NativeArray<T> src, NativeArray<T> dst)
		{
			this.dst = dst;
			this.src = src;
			srcPtr = (T*)src.GetUnsafeReadOnlyPtr();
			dstPtr = (T*)dst.GetUnsafeReadOnlyPtr();
		}

		public NativeArray<T> GetSrc() => src;
		public NativeArray<T> GetDst() => dst;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public void Write(int dstIndex, int srcIndex) => dst[dstIndex] = src[srcIndex];

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public void Write(int dstIndex, int srcIndex, int srcRange)
		{
			// Prefetch(dstIndex, srcIndex);
			for (int i = 0; i < srcRange; i++)
				dstPtr[dstIndex + i] = srcPtr[srcIndex + i];
		}

		public void Prefetch(int dstIndex, int srcIndex /*, Common.ReadWrite rw, Common.Locality locality*/)
		{
#if UNITY_BURST_EXPERIMENTAL_PREFETCH_INTRINSIC
			Common.Prefetch(srcPtr + srcIndex, Common.ReadWrite.Read, Common.Locality.LowTemporalLocality);
			Common.Prefetch(dstPtr + dstIndex, Common.ReadWrite.Write, Common.Locality.HighTemporalLocality);
#endif
		}
	}
	
	public class JobTests
	{	
		private struct GreaterThanZeroDel : IValidator<float>
		{
			public bool Validate(float element) => element > 0;
		}

		private struct ValidateTrue : IValidator<float>
		{
			public bool Validate(float element) => true;
		}

		private struct ValidateFalse : IValidator<float>
		{
			public bool Validate(float element) => false;
		}

		[Test]
		public static void GenericWriterTest()
		{
			NativeArray<float> src = new NativeArray<float>(11, Allocator.Persistent);
			NativeArray<float> dst = new NativeArray<float>(src.Length, Allocator.Persistent);
			
			for (int i = 0; i < src.Length; i++)
				src[i] = i;
			GenericWriter<float> writer = new GenericWriter<float>(src, dst);
			writer.Write(2,2,7);
			for (int i = 0; i < 11; i++)
				Assert.AreEqual( i is >= 2 and < 9 ? i : 0, dst[i]);
		}

		[Test]
		public void TestCopyAll1Bits() => TestBothSingleAndParallelCopyJobs<ValidateTrue>((i) => i);
	
		[Test]
		public void TestCopyAll0Bits() => TestBothSingleAndParallelCopyJobs<ValidateFalse>((i) => i);

		[Test]
		public void TestCopyAllOddBits() => TestBothSingleAndParallelCopyJobs<GreaterThanZeroDel>((i) => i % 2 == 0 ? -1f : 1);

		[Test]
		public void TestCopyAllBatchedBits()
		{
			int j = 0;
			TestBothSingleAndParallelCopyJobs<GreaterThanZeroDel>((_) => 
			{
				j++;
				if (j >= 5)
					j = -5;
				return j > 0 ? 1 : -1;
			});
		}

		private static void TestBothSingleAndParallelCopyJobs<T>(Func<float, float> dataGen) where T : unmanaged, IValidator<float>
		{
			TestParallelConditionParallelCopy<T>(dataGen);
		}

		private static void TestParallelConditionParallelCopy<T>(Func<float, float> dataGen) where T : unmanaged, IValidator<float>
		{
			NativeArray<float> src = new NativeArray<float>(100, Allocator.Persistent);
			for (int i = 0; i < src.Length; i++)
				src[i] = dataGen(i);

			NativeArray<BitField64> bits = new NativeArray<BitField64>((int)math.ceil(100f / 64f), Allocator.Persistent);
			NativeArray<int> counts = new NativeArray<int>(bits.Length, Allocator.Persistent);
			NativeReference<int> counter = new NativeReference<int>(0, Allocator.Persistent);
			ParallelIndexingSumJob<float, T>.Schedule(src, bits, counts, counter, out var job).Complete();
			//Debug.Log(Convert.ToString((long)bits[0].Value, toBase: 2));

			NativeArray<float> dstData = new NativeArray<float>(100, Allocator.Persistent);
			GenericWriter<float> writer = new GenericWriter<float>(src, dstData);
			ParallelConditionalCopyJob<float, GenericWriter<float>>.Schedule(job, writer).Complete();

			int count = counter.Value;
			counter.Dispose();
			bits.Dispose();

			// We copy all the data we wish to assert because if an assertion fails
			// we get exceptions due to native collections not being disposed.
			float[] srcCopy = new float[src.Length];
			src.CopyTo(srcCopy);
			float[] dstCopy = new float[dstData.Length];
			dstData.CopyTo(dstCopy);

			src.Dispose();
			dstData.Dispose();
			TestCopiedData<T>(srcCopy, dstCopy, count);
		}

		private static void TestCopiedData<T>(float[] src, float[] dst, int srcCount) where T : IValidator<float>
		{
			(float[] expected, int expectedLength) = GetExpected<T>(src);

			Assert.AreEqual(expectedLength, srcCount, "Incorrect length");
			Assert.AreEqual(dst.Length, src.Length, "Differing length between dst and src");

			for (int i = 0; i < dst.Length; i++)
			{
				// Debug.Log($"Expected/Actual: {expected[i]}/{dst[i]}");
				Assert.AreEqual(expected[i], dst[i], $"Index {i} had the wrong value");
			}
		}

		private static (float[] arr, int expectedLength) GetExpected<T>(IReadOnlyList<float> data) where T : IValidator<float>
		{
			T comparer = default;
			float[] expected = new float[100];
			int j = 0;
			for (int i = 0; i < expected.Length; i++)
			{
				if (comparer != null && !comparer.Validate(data[i])) continue;
				expected[j] = data[i];
				j++;
			}
			return (expected, j);
		}
	}
}
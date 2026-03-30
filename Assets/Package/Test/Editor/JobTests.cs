using System;
using System.Collections.Generic;
using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;

// TODO Add tests for job dependencies
// TODO Add tests for validators which also include job data such as NativeArrays

namespace Tests
{
	public class JobTests
	{	
		private struct GreaterThanZeroDel : IBatchValidator<float>
		{
			public bool Validate(int index, float element) => element > 0;
			public BitField64 Validate(in NativeSlice<float> elements)
			{
				var bits = new BitField64();
				for (int i = 0; i < 64; i++)
					bits.SetBits(i, elements[i] > 0);
				return bits;
			}
		}

		private struct ValidateTrue : IBatchValidator<float>
		{
			public bool Validate(int index, float element) => true;
			public BitField64 Validate(in NativeSlice<float> elements)
			{
				var bits = new BitField64();
				for (int i = 0; i < 64; i++)
					bits.SetBits(i, true);
				return bits;
			}
		}

		private struct ValidateFalse : IBatchValidator<float>
		{
			public bool Validate(int index, float element) => false;
			public BitField64 Validate(in NativeSlice<float> elements)
			{
				var bits = new BitField64();
				for (int i = 0; i < 64; i++)
					bits.SetBits(i, false);
				return bits;
			}
		}

		[Test]
		public static void GenericWriterTest()
		{
			NativeArray<float> src = new NativeArray<float>(11, Allocator.Persistent);
			NativeArray<float> dst = new NativeArray<float>(src.Length, Allocator.Persistent);
			
			for (int i = 0; i < src.Length; i++)
				src[i] = i;
			DataRW<float> writer = new DataRW<float>(src, dst);
			writer.Write(2,2,7);
			for (int i = 0; i < 11; i++)
				Assert.AreEqual( i is >= 2 and < 9 ? i : 0, dst[i]);
			src.Dispose();
			dst.Dispose();
		}

		[Test]
		public void TestCopyAll1Bits() => TestBothSingleAndParallelCopyJobs<ValidateTrue>((i) => i);
	
		[Test]
		public void TestCopyAll0Bits() => TestBothSingleAndParallelCopyJobs<ValidateFalse>((i) => i);

		[Test]
		public void TestCopyAllOddBits() => TestBothSingleAndParallelCopyJobs<GreaterThanZeroDel>((i) => i % 2 == 0 ? -1f : 1);

		[Test]
		public void RunLengthTest()
		{
			// runLength = math.tzcnt(~(n >> tzcnt))
			ulong threeAtOffset0 = 0b111UL;
			ulong threeAtOffset2 = 0b11100UL;
			ulong singleBitAtOffset1 = 0b10UL;
			ulong allBitsSet = ulong.MaxValue;

			Assert.AreEqual(3,  math.tzcnt(~(threeAtOffset0    >> 0)));
			Assert.AreEqual(3,  math.tzcnt(~(threeAtOffset2    >> 2)));
			Assert.AreEqual(1,  math.tzcnt(~(singleBitAtOffset1 >> 1)));
			Assert.AreEqual(64, math.tzcnt(~(allBitsSet         >> 0)));
		}
		
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

		private static void TestBothSingleAndParallelCopyJobs<T>(Func<float, float> dataGen) where T : unmanaged, IBatchValidator<float>
		{
			TestParallelConditionParallelCopy<T>(dataGen);
		}

		private static void TestParallelConditionParallelCopy<T>(Func<float, float> dataGen) where T : unmanaged, IBatchValidator<float>
		{
			NativeArray<float> src = new NativeArray<float>(100, Allocator.Persistent);
			for (int i = 0; i < src.Length; i++)
				src[i] = dataGen(i);

			NativeArray<BitField64> indices = new NativeArray<BitField64>((int)math.ceil(100f / 64f), Allocator.Persistent);
			NativeArray<int> counts = new NativeArray<int>(indices.Length, Allocator.Persistent);
			NativeArray<float> dstArr = new NativeArray<float>(100, Allocator.Persistent);
			NativeList<float> dstList = new NativeList<float>(100, Allocator.Persistent);

			src.IfCopyToParallel<float, T>(dstArr, out var counter, 10, 10, default, indices, counts).Complete();
			src.IfCopyToParallel<float, T>(dstList, 10, 10, default, indices, counts).Complete();
			
			int count = counter.Value;
			int listCount = dstList.Length;
			counter.Dispose();
			indices.Dispose();
			counts.Dispose();
			
			// We copy all the data we wish to assert because if an assertion fails
			// we get exceptions due to native collections not being disposed.
			float[] srcCopy = new float[src.Length];
			src.CopyTo(srcCopy);
			float[] dstCopy = new float[dstArr.Length];
			dstArr.CopyTo(dstCopy);
			float[] dstListCopy = new float[dstList.Length];
			dstList.AsArray().CopyTo(dstListCopy);
			
			src.Dispose();
			dstArr.Dispose();
			dstList.Dispose();
			
			TestCopiedData<T>(srcCopy, dstCopy, count);
			TestCopiedData<T>(srcCopy, dstListCopy, listCount);
		}

		private static void TestCopiedData<T>(float[] src, float[] dst, int srcCount) where T : IValidator<float>
		{
			(float[] expected, int expectedLength) = GetExpected<T>(src);

			Assert.AreEqual(expectedLength, srcCount, "Incorrect length");

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
				if (comparer != null && !comparer.Validate(i, data[i])) continue;
				expected[j] = data[i];
				j++;
			}
			return (expected, j);
		}
	}
}
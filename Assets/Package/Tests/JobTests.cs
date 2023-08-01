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
	public class JobTests
	{	
		private struct GreaterThanZeroDel : IValidator<float>
		{
			public bool Validate(int index, float element) => element > 0;
		}

		private struct ValidateTrue : IValidator<float>
		{
			public bool Validate(int index, float element) => true;
		}

		private struct ValidateFalse : IValidator<float>
		{
			public bool Validate(int index, float element) => false;
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
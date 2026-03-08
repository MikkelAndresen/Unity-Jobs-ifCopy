using NUnit.Framework;
using Unity.Collections;
using Unity.Jobs;

namespace Tests
{
	public class SimpleCopyTests
	{
		[Test]
		public void ConditionalCopy()
		{
			var src = new NativeArray<int>(10, Allocator.Persistent);
			var dst = new NativeList<int>(10, Allocator.Persistent);
			src.AsSpan().Fill(1);

			RunJob();

			Assert.That(dst.Length, Is.EqualTo(0));
			
			src.AsSpan().Fill(2);
			dst.AsArray().AsSpan().Fill(0);
			dst.Length = 0;
			RunJob();

			Assert.That(dst.Length, Is.EqualTo(10));
			for (int i = 0; i < dst.Length; i++)
				Assert.That(dst[i], Is.EqualTo(2));
			
			var s = src.AsSpan();
			for (int i = 0; i < s.Length; i++)
				s[i] = i;
			
			dst.AsArray().AsSpan().Fill(0);
			dst.Length = 0;
			RunJob();
			
			Assert.That(dst.Length, Is.EqualTo(8));
			void RunJob() => new ConditionalCopyJob<int, Validator>()
				{ Validator = new Validator(), dst = dst.AsParallelWriter(), src = src }.Schedule(src.Length, 1).Complete();
		}
		
		[Test]
		public void ConditionalFilterCopy()
		{
			var src = new NativeArray<int>(10, Allocator.Persistent);
			var dst = new NativeList<int>(10, Allocator.Persistent);
			var indices = new NativeList<int>(10, Allocator.Persistent);
			src.AsSpan().Fill(1);

			RunJob();
			Assert.That(dst.Length, Is.EqualTo(0));
			
			Reset();
			src.AsSpan().Fill(2);
			RunJob();

			Assert.That(dst.Length, Is.EqualTo(10));
			for (int i = 0; i < dst.Length; i++)
				Assert.That(dst[i], Is.EqualTo(2));
			
			var s = src.AsSpan();
			for (int i = 0; i < s.Length; i++)
				s[i] = i;
			
			dst.AsArray().AsSpan().Fill(0);
			Reset();
			RunJob();
			
			Assert.That(dst.Length, Is.EqualTo(8));

			void RunJob() => FilterCopy<int, Validator>.Schedule(src, dst, indices, 1).Complete();

			void Reset()
			{
				dst.AsArray().AsSpan().Fill(0);
				dst.Length = 0;
				indices.Length = 0;
			}
		}
		
		private struct Validator : IValidator<int>
		{
			public bool Validate(int index, int element) => element > 1;
		}
	}
}
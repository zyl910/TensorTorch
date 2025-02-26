using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Numerics;
using System.Numerics.Tensors;
using Zyl.TensorTorch;

namespace Zyl.TensorTorch.Tests {
    [TestFixture()]
    public class TTorchTests {
#pragma warning disable SYSLIB5001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.

        private static TextWriter Writer { get; } = Console.Out;

        [TestCase((float)1)]
        [TestCase((double)2)]
        [TestCase((sbyte)3)]
        [TestCase((byte)4)]
        [TestCase((short)5)]
        [TestCase((ushort)6)]
        [TestCase((int)7)]
        [TestCase((uint)8)]
        [TestCase((long)9)]
        [TestCase((ulong)10)]
        public void CloneTest<T>(T src) where T : INumberBase<T> {
            var A = TTorch.Arange(src);
            var B = A.Clone();
            Assert.AreEqual(A, B);
        }

        [TestCase((float)1)]
        [TestCase((double)2)]
        [TestCase((sbyte)3)]
        [TestCase((byte)4)]
        [TestCase((short)5)]
        [TestCase((ushort)6)]
        [TestCase((int)7)]
        [TestCase((uint)8)]
        [TestCase((long)9)]
        [TestCase((ulong)10)]
        public void ArangeTest<T>(T src) where T : INumberBase<T> {
            int len = int.CreateTruncating(src);
            Writer.WriteLine("len: {0}", len);
            Tensor<T> expected = Tensor.Create<T>(Enumerable.Range(0, len).Select(i => T.CreateTruncating(i)), [len]);

            Tensor<T> dst = TTorch.Arange(src);
            Assert.AreEqual(1, dst.Rank);
            Assert.AreEqual((nint)len, dst.Lengths[0]);
            Assert.AreEqual(expected, dst);

            dst = TTorch.Arange(T.Zero, src);
            Assert.AreEqual(1, dst.Rank);
            Assert.AreEqual((nint)len, dst.Lengths[0]);
            Assert.AreEqual(expected, dst);

            dst = TTorch.Arange(T.Zero, src, T.One);
            Assert.AreEqual(1, dst.Rank);
            Assert.AreEqual((nint)len, dst.Lengths[0]);
            Assert.AreEqual(expected, dst);
        }

        [TestCase((float)1)]
        [TestCase((double)2)]
        [TestCase((sbyte)3)]
        [TestCase((byte)4)]
        [TestCase((short)5)]
        [TestCase((ushort)6)]
        [TestCase((int)7)]
        [TestCase((uint)8)]
        [TestCase((long)9)]
        [TestCase((ulong)10)]
        public void SumTorchTest<T>(T src) where T : INumberBase<T> {
            const int rank = 2;
            const int rank1 = 1;
            const int rankAll = 1;
            const nint one = 1;
            const nint m = 5, n = 4;
            const nint numel = m * n;
            T numelT = T.CreateChecked(numel);
            Tensor<T> A = TTorch.Arange(numelT).Reshape(5, 4);
            T sumData = Tensor.Sum(A.AsReadOnlyTensorSpan());
            Tensor<T> D;

            D = A.SumTorch([]);
            Assert.AreEqual(rankAll, D.Rank);
            Assert.AreEqual(one, D.FlattenedLength);
            Assert.AreEqual(sumData, D[0]);

            D = A.SumTorch([], true);
            Assert.AreEqual(rank, D.Rank);
            Assert.AreEqual(one, D.FlattenedLength);
            Assert.AreEqual(sumData, Tensor.Sum(D.AsReadOnlyTensorSpan()));

            D = A.SumTorch([0]);
            Assert.AreEqual(rank1, D.Rank);
            Assert.AreEqual(n, D.FlattenedLength);
            Assert.AreEqual(sumData, Tensor.Sum(D.AsReadOnlyTensorSpan()));

            D = A.SumTorch([0], true);
            Assert.AreEqual(rank, D.Rank);
            Assert.AreEqual(n, D.FlattenedLength);
            Assert.AreEqual(sumData, Tensor.Sum(D.AsReadOnlyTensorSpan()));

            D = A.SumTorch([1]);
            Assert.AreEqual(rank1, D.Rank);
            Assert.AreEqual(m, D.FlattenedLength);
            Assert.AreEqual(sumData, Tensor.Sum(D.AsReadOnlyTensorSpan()));

            D = A.SumTorch([1], true);
            Assert.AreEqual(rank, D.Rank);
            Assert.AreEqual(m, D.FlattenedLength);
            Assert.AreEqual(sumData, Tensor.Sum(D.AsReadOnlyTensorSpan()));

            D = A.SumTorch([0, 1]);
            Assert.AreEqual(rankAll, D.Rank);
            Assert.AreEqual(one, D.FlattenedLength);
            Assert.AreEqual(sumData, Tensor.Sum(D.AsReadOnlyTensorSpan()));

            D = A.SumTorch([0, 1], true);
            Assert.AreEqual(rank, D.Rank);
            Assert.AreEqual(one, D.FlattenedLength);
            Assert.AreEqual(sumData, Tensor.Sum(D.AsReadOnlyTensorSpan()));
        }

#pragma warning restore SYSLIB5001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.
    }
}

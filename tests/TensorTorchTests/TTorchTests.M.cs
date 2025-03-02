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
    partial class TTorchTests {
#pragma warning disable SYSLIB5001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.

        [TestCase((float)1)]
        [TestCase((double)2)]
        [TestCaseSource(typeof(TestDataSource), nameof(TestDataSource.UseExtendFloats))]
        public void MeanTorchTest<T>(T src) where T : IFloatingPoint<T> {
            const int rank = 2;
            const int rank1 = 1;
            const int rankAll = 1;
            const nint one = 1;
            const nint m = 5, n = 4;
            const nint numel = m * n;
            Writer.WriteLine(string.Format("MeanTorchTest<{0}>", src));
            T numelT = T.CreateChecked(numel);
            Tensor<T> A = TTorch.Arange(numelT).Reshape(m, n);
            T sumData = Tensor.Average(A.AsReadOnlyTensorSpan());
            Tensor<T> D;

            D = A.MeanTorch([]);
            Assert.AreEqual(rankAll, D.Rank);
            Assert.AreEqual(one, D.FlattenedLength);
            Assert.AreEqual(sumData, D[0]);

            D = A.MeanTorch([], true);
            Assert.AreEqual(rank, D.Rank);
            Assert.AreEqual(one, D.FlattenedLength);
            Assert.AreEqual(sumData, Tensor.Average(D.AsReadOnlyTensorSpan()));

            D = A.MeanTorch([0]);
            Assert.AreEqual(rank1, D.Rank);
            Assert.AreEqual(n, D.FlattenedLength);
            Assert.AreEqual(sumData, Tensor.Average(D.AsReadOnlyTensorSpan()));

            D = A.MeanTorch([0], true);
            Assert.AreEqual(rank, D.Rank);
            Assert.AreEqual(n, D.FlattenedLength);
            Assert.AreEqual(sumData, Tensor.Average(D.AsReadOnlyTensorSpan()));

            D = A.MeanTorch([1]);
            Assert.AreEqual(rank1, D.Rank);
            Assert.AreEqual(m, D.FlattenedLength);
            Assert.AreEqual(sumData, Tensor.Average(D.AsReadOnlyTensorSpan()));

            D = A.MeanTorch([1], true);
            Assert.AreEqual(rank, D.Rank);
            Assert.AreEqual(m, D.FlattenedLength);
            Assert.AreEqual(sumData, Tensor.Average(D.AsReadOnlyTensorSpan()));

            D = A.MeanTorch([0, 1]);
            Assert.AreEqual(rankAll, D.Rank);
            Assert.AreEqual(one, D.FlattenedLength);
            Assert.AreEqual(sumData, Tensor.Average(D.AsReadOnlyTensorSpan()));

            D = A.MeanTorch([0, 1], true);
            Assert.AreEqual(rank, D.Rank);
            Assert.AreEqual(one, D.FlattenedLength);
            Assert.AreEqual(sumData, Tensor.Average(D.AsReadOnlyTensorSpan()));
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
        [TestCaseSource(typeof(TestDataSource), nameof(TestDataSource.UseExtendFloats))]
        [TestCaseSource(typeof(TestDataSource), nameof(TestDataSource.UseExtendInts))]
        public void MultiplyVectorTest<T>(T src) where T : INumberBase<T> {
            const nint m = 5, n = 4;
            const nint numel = m * n;
            Writer.WriteLine(string.Format("MultiplyVectorTest<{0}>", src));
            T numelT = T.CreateChecked(numel);
            Tensor<T> A = TTorch.Arange(numelT).Reshape(m, n);
            Tensor<T> x = TTorch.Arange(T.CreateChecked(n));
            Tensor<T> expected = A.MultiplyVector(x);
            Tensor<T> dst;
            Assert.AreEqual(1, expected.Rank);
            Assert.AreEqual(m, expected.FlattenedLength);

            // The vec is row Vector.
            dst = A.MultiplyVector(x.AsReadOnlyTensorSpan(), false);
            Assert.AreEqual(1, dst.Rank);
            Assert.AreEqual(m, dst.FlattenedLength);
            Assert.AreEqual(expected, dst);

            // The vec is column Vector.
            dst = A.MultiplyVector(x.AsReadOnlyTensorSpan().Reshape([n, 1]), true);
            Assert.AreEqual(1, dst.Rank);
            Assert.AreEqual(m, dst.FlattenedLength);
            Assert.AreEqual(expected, dst);

            // The output is row Vector.
            dst.Clear();
            TTorch.MultiplyVector(A.AsReadOnlyTensorSpan(), x, dst);
            Assert.AreEqual(expected, dst);

            // The vec is column Vector.
            dst = Tensor.Create<T>([m, 1]);
            TTorch.MultiplyVector(A.AsReadOnlyTensorSpan(), x, dst);
            Assert.AreEqual(expected, dst);

            // Values.
            float[] values = { 14f, 38, 62, 86, 110 };
            for (int i = 0; i < values.Length; ++i) {
                T item = T.CreateChecked(values[i]);
                Assert.AreEqual(item, expected[i], "item[{0}]", i);
            }

        }

#pragma warning restore SYSLIB5001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.
    }
}

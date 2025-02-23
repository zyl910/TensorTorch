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

#pragma warning restore SYSLIB5001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.
    }
}

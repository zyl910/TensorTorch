using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace Zyl.TensorTorch {
#pragma warning disable SYSLIB5001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.

    /// <summary>
    /// Tensor Torch util (张量 Torch 工具).
    /// </summary>
    public static class TTorch {

        // -- torch.arange(start=0, end, step=1, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)

        /// <summary>
        /// This method returns a tensor containing values from a given interval [start, end) with a specified step size. When the step size is not an integer, floating-point rounding errors may occur, so it is recommended to subtract a small epsilon from the end value for consistency (此方法返回一个张量，其中包含具有指定步长的给定区间 [start，end) 的值。当步长不是整数时，可能会出现浮点舍入错误，因此建议从结束值中减去一个较小的 epsilon 以保持一致性). Like `torch.arange`.
        /// </summary>
        /// <typeparam name="T">The element type (元素类型).</typeparam>
        /// <param name="end">The ending value of the range, exclusive. This parameter is required.</param>
        /// <param name="pinned">A Boolean whether the underlying data should be pinned or not (一个布尔值，表示是否应固定基础数据).</param>
        /// <returns>Returns new Tensor.</returns>
        public static Tensor<T> Arange<T>(T end, bool pinned = false) where T : INumberBase<T> {
            return Arange(T.Zero, end, T.One, pinned);
        }

        /// <summary>
        /// This method returns a tensor containing values from a given interval [start, end) with a specified step size. When the step size is not an integer, floating-point rounding errors may occur, so it is recommended to subtract a small epsilon from the end value for consistency (此方法返回一个张量，其中包含具有指定步长的给定区间 [start，end) 的值。当步长不是整数时，可能会出现浮点舍入错误，因此建议从结束值中减去一个较小的 epsilon 以保持一致性). Like `torch.arange`.
        /// </summary>
        /// <typeparam name="T">The element type (元素类型).</typeparam>
        /// <param name="start">The starting value of the range, inclusive. Defaults to 0.</param>
        /// <param name="end">The ending value of the range, exclusive. This parameter is required.</param>
        /// <param name="pinned">A Boolean whether the underlying data should be pinned or not (一个布尔值，表示是否应固定基础数据).</param>
        /// <returns>Returns new Tensor.</returns>
        public static Tensor<T> Arange<T>(T start, T end, bool pinned = false) where T : INumberBase<T> {
            return Arange(start, end, T.One, pinned);
        }

        /// <summary>
        /// This method returns a tensor containing values from a given interval [start, end) with a specified step size. When the step size is not an integer, floating-point rounding errors may occur, so it is recommended to subtract a small epsilon from the end value for consistency (此方法返回一个张量，其中包含具有指定步长的给定区间 [start，end) 的值。当步长不是整数时，可能会出现浮点舍入错误，因此建议从结束值中减去一个较小的 epsilon 以保持一致性). Like `torch.arange`.
        /// </summary>
        /// <typeparam name="T">The element type (元素类型).</typeparam>
        /// <param name="start">The starting value of the range, inclusive. Defaults to 0.</param>
        /// <param name="end">The ending value of the range, exclusive. This parameter is required.</param>
        /// <param name="step">The difference between each consecutive value in the range. The default value is 1.</param>
        /// <param name="pinned">A Boolean whether the underlying data should be pinned or not (一个布尔值，表示是否应固定基础数据).</param>
        /// <returns>Returns new Tensor.</returns>
        public static Tensor<T> Arange<T>(T start, T end, T step, bool pinned = false) where T : INumberBase<T> {
            nint cnt = GetRangeCount(start, end, step);
            Span<nint> lengths = [cnt];
            Tensor<T> rt = Tensor.CreateUninitialized<T>(lengths, pinned);
            T m = start;
            ref T p = ref rt.GetPinnableReference();
            ref T pEnd = ref Unsafe.Add(ref p, cnt);
            for (; Unsafe.IsAddressLessThan(ref p, ref pEnd); p = ref Unsafe.Add(ref p, 1)) {
                p = m;
                m += step;
            }
            return rt;
        }

        /// <summary>
        /// Create and fill with the given value (创建并使用指定值填充).
        /// </summary>
        /// <typeparam name="T">The element type (元素类型).</typeparam>
        /// <param name="value">Given value (指定值).</param>
        /// <param name="lengths">A <see cref="ReadOnlySpan{T}"/> indicating the lengths of each dimension (表示每个维度的长度).</param>
        /// <param name="pinned">A Boolean whether the underlying data should be pinned or not (一个布尔值，表示是否应固定基础数据).</param>
        /// <returns>Returns new Tensor.</returns>
        public static Tensor<T> CreateAndFill<T>(T value, ReadOnlySpan<IntPtr> lengths, bool pinned = false) {
            Tensor<T> rt = Tensor.CreateUninitialized<T>(lengths, pinned);
            rt.Fill(value);
            return rt;
        }

        /// <summary>
        /// Get rangge count.
        /// </summary>
        /// <typeparam name="T">The element type (元素类型).</typeparam>
        /// <param name="start">The starting value of the range, inclusive.</param>
        /// <param name="end">The ending value of the range, exclusive. This parameter is required.</param>
        /// <param name="step">The difference between each consecutive value in the range.</param>
        /// <returns>Returns rangge count.</returns>
        internal static nint GetRangeCount<T>(T start, T end, T step) where T : INumberBase<T> {
            T nRaw = end - start / step;
            nint n = nint.CreateChecked(nRaw);
            return n;
        }

        // -- torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

        /// <summary>
        /// Create tensor based on 1D array (根据1维数组创建张量). Like `torch.tensor`.
        /// </summary>
        /// <typeparam name="T">The element type (元素类型).</typeparam>
        /// <param name="values">Souce values (源值).</param>
        /// <param name="pinned">A Boolean whether the underlying data should be pinned or not (一个布尔值，表示是否应固定基础数据).</param>
        /// <returns>Returns new Tensor.</returns>
        public static Tensor<T> FromNDArray<T>(T[] values, bool pinned = false) {
            var tensorSpan = new TensorSpan<T>(values);
            Tensor<T> rt = Tensor.Create<T>(tensorSpan.Lengths, pinned);
            tensorSpan.CopyTo(rt);
            return rt;
        }

        /// <summary>
        /// Create tensor based on 2D array (根据2维数组创建张量). Like `torch.tensor`.
        /// </summary>
        /// <typeparam name="T">The element type (元素类型).</typeparam>
        /// <param name="values">Souce values (源值).</param>
        /// <param name="pinned">A Boolean whether the underlying data should be pinned or not (一个布尔值，表示是否应固定基础数据).</param>
        /// <returns>Returns new Tensor.</returns>
        public static Tensor<T> FromNDArray<T>(T[,] values, bool pinned = false) {
            var tensorSpan = new TensorSpan<T>(values);
            Tensor<T> rt = Tensor.Create<T>(tensorSpan.Lengths, pinned);
            tensorSpan.CopyTo(rt);
            return rt;
        }

        /// <summary>
        /// Create tensor based on 3D array (根据3维数组创建张量). Like `torch.tensor`.
        /// </summary>
        /// <typeparam name="T">The element type (元素类型).</typeparam>
        /// <param name="values">Souce values (源值).</param>
        /// <param name="pinned">A Boolean whether the underlying data should be pinned or not (一个布尔值，表示是否应固定基础数据).</param>
        /// <returns>Returns new Tensor.</returns>
        public static Tensor<T> FromNDArray<T>(T[,,] values, bool pinned = false) {
            var tensorSpan = new TensorSpan<T>(values);
            Tensor<T> rt = Tensor.Create<T>(tensorSpan.Lengths, pinned);
            tensorSpan.CopyTo(rt);
            return rt;
        }

        /// <summary>
        /// Create tensor based on 4D array (根据4维数组创建张量). Like `torch.tensor`.
        /// </summary>
        /// <typeparam name="T">The element type (元素类型).</typeparam>
        /// <param name="values">Souce values (源值).</param>
        /// <param name="pinned">A Boolean whether the underlying data should be pinned or not (一个布尔值，表示是否应固定基础数据).</param>
        /// <returns>Returns new Tensor.</returns>
        public static Tensor<T> FromNDArray<T>(T[,,,] values, bool pinned = false) {
            var tensorSpan = new TensorSpan<T>(values);
            Tensor<T> rt = Tensor.Create<T>(tensorSpan.Lengths, pinned);
            tensorSpan.CopyTo(rt);
            return rt;
        }

        /// <summary>
        /// Create tensor based on ND array (根据N维数组创建张量). Like `torch.tensor`.
        /// </summary>
        /// <typeparam name="T">The element type (元素类型).</typeparam>
        /// <param name="values">Souce values (源值).</param>
        /// <param name="pinned">A Boolean whether the underlying data should be pinned or not (一个布尔值，表示是否应固定基础数据).</param>
        /// <returns>Returns new Tensor.</returns>
        public static Tensor<T> FromNDArray<T>(Array values, bool pinned = false) {
            var tensorSpan = new TensorSpan<T>(values);
            Tensor<T> rt = Tensor.Create<T>(tensorSpan.Lengths, pinned);
            tensorSpan.CopyTo(rt);
            return rt;
        }

        // -- torch.ones(size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)

        /// <summary>
        /// This method creates a tensor of a specified shape, where each element is initialized to the scalar value 1 (此方法用于创建指定形状的张量，其中每个元素都初始化为标量值1). Like `torch.ones`.
        /// </summary>
        /// <typeparam name="T">The element type (元素类型).</typeparam>
        /// <param name="lengths">A <see cref="ReadOnlySpan{T}"/> indicating the lengths of each dimension (表示每个维度的长度).</param>
        /// <param name="pinned">A Boolean whether the underlying data should be pinned or not (一个布尔值，表示是否应固定基础数据).</param>
        /// <returns>Returns new Tensor.</returns>
        public static Tensor<T> Ones<T>(ReadOnlySpan<IntPtr> lengths, bool pinned = false) where T : INumberBase<T> {
            return CreateAndFill(T.One, lengths, pinned);
        }

        /// <summary>
        /// ReadOnlySpan to <see cref="StringBuilder"/>.
        /// </summary>
        /// <typeparam name="T">The element type (元素类型).</typeparam>
        /// <param name="builder">Output <see cref="StringBuilder"/>.</param>
        /// <param name="source">The source.</param>
        /// <param name="separator">The separator.</param>
        public static void ToString<T>(StringBuilder builder, ReadOnlySpan<T> source, string? separator = null) {
            if (null == separator) separator = ", ";
            builder.Append('[');
            for (int i = 0; i < source.Length; i++) {
                if (i > 0) {
                    builder.Append(separator);
                }
                T p = source[i];
                builder.Append(p);
            }
            builder.Append(']');
        }

        /// <summary>
        /// ReadOnlySpan to String.
        /// </summary>
        /// <typeparam name="T">The element type (元素类型).</typeparam>
        /// <param name="source">The source.</param>
        /// <param name="separator">The separator.</param>
        /// <returns>Returns string.</returns>
        public static string ToString<T>(ReadOnlySpan<T> source, string? separator = null) {
            StringBuilder builder = new StringBuilder();
            ToString(builder, source, separator);
            return builder.ToString();
        }

        // -- torch.zeros(size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)

        /// <summary>
        /// This method returns a tensor filled with zeros that has a specified shape (此方法返回一个填充有具有指定形状的零的张量). Like `torch.zeros`.
        /// </summary>
        /// <typeparam name="T">The element type (元素类型).</typeparam>
        /// <param name="lengths">A <see cref="ReadOnlySpan{T}"/> indicating the lengths of each dimension (表示每个维度的长度).</param>
        /// <param name="pinned">A Boolean whether the underlying data should be pinned or not (一个布尔值，表示是否应固定基础数据).</param>
        /// <returns>Returns new Tensor.</returns>
        public static Tensor<T> Zeros<T>(ReadOnlySpan<IntPtr> lengths, bool pinned = false) {
            return Tensor.Create<T>(lengths, pinned);
        }

    }

#pragma warning restore SYSLIB5001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.
}

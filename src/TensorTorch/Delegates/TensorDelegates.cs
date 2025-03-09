using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;
using System.Text;
using System.Threading.Tasks;

namespace Zyl.TensorTorch.Delegates {
#pragma warning disable SYSLIB5001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.

    /// <summary>
    /// The creator of tensor (张量的创建者).
    /// </summary>
    /// <typeparam name="TTensor">The tensor type (张量类型).</typeparam>
    /// <typeparam name="T">The element type (元素类型).</typeparam>
    /// <param name="source">The source. It is used to provide information when lengths/strides is empty. The source/lengths cannot be empty at the same time (源. 它用于在 lengths/strides 为空时提供信息. source/lengths 不能同时为空)</param>
    /// <param name="lengths">The lengths of each dimension. If this parameter is empty, it will be taken from the source (各个维度的长度. 若本参数为空, 则会从 source 取值).</param>
    /// <param name="strides">The strides of each dimension. If this parameter is empty, it will be taken from the source (各个维度的跨距. 若本参数为空, 则会从 source 取值).</param>
    /// <param name="pinned">A Boolean whether the underlying data should be pinned or not (一个布尔值，表示是否应固定基础数据).</param>
    /// <returns>Returns new Tensor (返回新张量).</returns>
    public delegate TTensor TensorCreator<out TTensor, T>(ReadOnlyTensorSpan<T> source, ReadOnlySpan<nint> lengths, scoped ReadOnlySpan<nint> strides, bool pinned);

    /// <summary>
    /// Delegates of Tensor (张量的委托).
    /// </summary>
    public static class TensorDelegates {

        /// <summary>
        /// Creates a tensor without initialization, and ignores strides. It has a low memory footprint (创建张量且不进行初始化, 且忽略 strides 参数. 它的内存占用少).
        /// </summary>
        /// <typeparam name="TTensor">The tensor type (张量类型).</typeparam>
        /// <typeparam name="T">The element type (元素类型).</typeparam>
        /// <param name="source">The source. It is used to provide information when lengths/strides is empty. The source/lengths cannot be empty at the same time (源. 它用于在 lengths/strides 为空时提供信息. source/lengths 不能同时为空)</param>
        /// <param name="lengths">The lengths of each dimension. If this parameter is empty, it will be taken from the source (各个维度的长度. 若本参数为空, 则会从 source 取值).</param>
        /// <param name="strides">The strides of each dimension. If this parameter is empty, it will be taken from the source (各个维度的跨距. 若本参数为空, 则会从 source 取值).</param>
        /// <param name="pinned">A Boolean whether the underlying data should be pinned or not (一个布尔值，表示是否应固定基础数据).</param>
        /// <returns>Returns new Tensor (返回新张量).</returns>
        public static TTensor CreateUninitialized<TTensor, T>(ReadOnlyTensorSpan<T> source, ReadOnlySpan<nint> lengths, scoped ReadOnlySpan<nint> strides, bool pinned) where TTensor : ITensor<TTensor, T> {
            ReadOnlySpan<nint> lengths1 = (lengths.IsEmpty) ? source.Lengths : lengths;
            return TTensor.CreateUninitialized(lengths1, pinned);
        }

        /// <summary>
        /// Creates a tensor without initialization, the strides argument is used (创建张量且不进行初始化, 会使用 strides 参数).
        /// </summary>
        /// <typeparam name="TTensor">The tensor type (张量类型).</typeparam>
        /// <typeparam name="T">The element type (元素类型).</typeparam>
        /// <param name="source">The source. It is used to provide information when lengths/strides is empty. The source/lengths cannot be empty at the same time (源. 它用于在 lengths/strides 为空时提供信息. source/lengths 不能同时为空)</param>
        /// <param name="lengths">The lengths of each dimension. If this parameter is empty, it will be taken from the source (各个维度的长度. 若本参数为空, 则会从 source 取值).</param>
        /// <param name="strides">The strides of each dimension. If this parameter is empty, it will be taken from the source (各个维度的跨距. 若本参数为空, 则会从 source 取值).</param>
        /// <param name="pinned">A Boolean whether the underlying data should be pinned or not (一个布尔值，表示是否应固定基础数据).</param>
        /// <returns>Returns new Tensor (返回新张量).</returns>
        public static TTensor CreateUninitializedWithStrides<TTensor, T>(ReadOnlyTensorSpan<T> source, ReadOnlySpan<nint> lengths, scoped ReadOnlySpan<nint> strides, bool pinned) where TTensor : ITensor<TTensor, T> {
            ReadOnlySpan<nint> lengths1 = (lengths.IsEmpty) ? source.Lengths : lengths;
            ReadOnlySpan<nint> strides1 = (strides.IsEmpty) ? source.Strides : strides;
            return TTensor.CreateUninitialized(lengths1, strides1, pinned);
        }

    }

#pragma warning restore SYSLIB5001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.
}

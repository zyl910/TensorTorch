using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace Zyl.TensorTorch {
#pragma warning disable SYSLIB5001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.

    partial class TTorch {

        // -- torch.clone(input, *, memory_format=torch.preserve_format)

        /// <summary>
        /// Returns a copy of source data (返回源数据的拷贝).
        /// </summary>
        /// <typeparam name="T">The element type (元素类型).</typeparam>
        /// <param name="source">The source (源).</param>
        /// <returns>Returns new Tensor (返回新张量).</returns>
        public static Tensor<T> Clone<T>(this in ReadOnlyTensorSpan<T> source) {
            Tensor<T> rt = Tensor.CreateUninitialized<T>(source.Lengths);
            source.CopyTo(rt);
            return rt;
        }

        /// <summary>
        /// Returns a copy of source data (返回源数据的拷贝).
        /// </summary>
        /// <typeparam name="T">The element type (元素类型).</typeparam>
        /// <param name="source">The source (源).</param>
        /// <returns>Returns new Tensor (返回新张量).</returns>
        public static Tensor<T> Clone<T>(this in TensorSpan<T> source) {
            return Clone((ReadOnlyTensorSpan<T>)source);
        }

        /// <summary>
        /// Returns a copy of source data (返回源数据的拷贝).
        /// </summary>
        /// <typeparam name="T">The element type (元素类型).</typeparam>
        /// <param name="source">The source (源).</param>
        /// <returns>Returns new Tensor (返回新张量).</returns>
        public static Tensor<T> Clone<T>(this Tensor<T> source) {
            return Clone(source.AsReadOnlyTensorSpan());
        }

        /// <summary>
        /// Fills the contents of this ranges with the given value (用给定值填充此范围的内容).
        /// </summary>
        /// <typeparam name="T">The element type (元素类型).</typeparam>
        /// <param name="source">The source (源).</param>
        /// <param name="value">The new value (新值).</param>
        /// <param name="ranges">The ranges for the slice (切片的范围).</param>
        /// <exception cref="ArgumentOutOfRangeException">ranges.Length &gt; source.Lengths.Length</exception>
        public static void FillRange<T>(this Tensor<T> source, T value, params scoped ReadOnlySpan<NRange> ranges) {
            if (source.IsEmpty) {
                throw new ArgumentOutOfRangeException(nameof(source), "source is empty!");
            }
            int rank = source.Lengths.Length;
            if (ranges.Length > rank) {
                throw new ArgumentOutOfRangeException(nameof(ranges), "Number of dimensions to slice does not equal the number of dimensions in the span");
            }
            Span<NRange> ranges2 = stackalloc NRange[rank];
            ranges.CopyTo(ranges2);
            for (int i = ranges.Length; i < rank; ++i) {
                ranges2[i] = NRange.All;
            }
            Span<nint> indices = stackalloc nint[rank];
            FillRange_Core(source, value, ranges2, source.Lengths, 0, indices);
        }

        /// <summary>
        /// Fills the contents of this ranges with the given value - On level (用给定值填充此范围的内容 - 在某层).
        /// </summary>
        /// <typeparam name="T">The element type (元素类型).</typeparam>
        /// <param name="source">The source (源).</param>
        /// <param name="value">The new value (新值).</param>
        /// <param name="ranges">The ranges for the slice (切片的范围).</param>
        /// <param name="lengths">Tensor lengths (张量长度).</param>
        /// <param name="level">Current level (当前层级).</param>
        /// <param name="indices">Indices during looping(循环时的索引).</param>
        private static void FillRange_Core<T>(this Tensor<T> source, T value, scoped ReadOnlySpan<NRange> ranges, scoped ReadOnlySpan<nint> lengths, int level, scoped Span<nint> indices) {
            int rank = lengths.Length;
            NRange range = ranges[level];
           (nint offset, nint length) = range.GetOffsetAndLength(lengths[level]);
            nint offsetEnd = offset + length;
            int levelNext = level + 1;
            if (levelNext < rank) {
                for (nint i = offset; i != offsetEnd; ++i) {
                    indices[level] = i;
                    FillRange_Core(source, value, ranges, lengths, levelNext, indices);
                }
            } else {
                for (nint i = offset; i != offsetEnd; ++i) {
                    indices[level] = i;
                    source[indices] = value;
                }
            }
        }

        /// <summary>
        /// Fills the contents of this ranges with the given value (用给定值填充此范围的内容).
        /// </summary>
        /// <typeparam name="T">The element type (元素类型).</typeparam>
        /// <param name="source">The source (源).</param>
        /// <param name="value">The new value (新值).</param>
        /// <param name="ranges">The ranges for the slice (切片的范围).</param>
        /// <exception cref="ArgumentOutOfRangeException">ranges.Length &gt; source.Lengths.Length</exception>
        public static void FillRange<T>(this in TensorSpan<T> source, T value, params scoped ReadOnlySpan<NRange> ranges) {
            if (source.IsEmpty) {
                throw new ArgumentOutOfRangeException(nameof(source), "source is empty!");
            }
            int rank = source.Lengths.Length;
            if (ranges.Length > rank) {
                throw new ArgumentOutOfRangeException(nameof(ranges), "Number of dimensions to slice does not equal the number of dimensions in the span");
            }
            Span<NRange> ranges2 = stackalloc NRange[rank];
            ranges.CopyTo(ranges2);
            for (int i = ranges.Length; i < rank; ++i) {
                ranges2[i] = NRange.All;
            }
            Span<nint> indices = stackalloc nint[rank];
            FillRange_Core(source, value, ranges2, source.Lengths, 0, indices);
        }

        /// <summary>
        /// Fills the contents of this ranges with the given value - On level (用给定值填充此范围的内容 - 在某层).
        /// </summary>
        /// <typeparam name="T">The element type (元素类型).</typeparam>
        /// <param name="source">The source (源).</param>
        /// <param name="value">The new value (新值).</param>
        /// <param name="ranges">The ranges for the slice (切片的范围).</param>
        /// <param name="lengths">Tensor lengths (张量长度).</param>
        /// <param name="level">Current level (当前层级).</param>
        /// <param name="indices">Indices during looping(循环时的索引).</param>
        private static void FillRange_Core<T>(this in TensorSpan<T> source, T value, scoped ReadOnlySpan<NRange> ranges, scoped ReadOnlySpan<nint> lengths, int level, scoped Span<nint> indices) {
            int rank = lengths.Length;
            NRange range = ranges[level];
           (nint offset, nint length) = range.GetOffsetAndLength(lengths[level]);
            nint offsetEnd = offset + length;
            int levelNext = level + 1;
            if (levelNext < rank) {
                for (nint i = offset; i != offsetEnd; ++i) {
                    indices[level] = i;
                    FillRange_Core(source, value, ranges, lengths, levelNext, indices);
                }
            } else {
                for (nint i = offset; i != offsetEnd; ++i) {
                    indices[level] = i;
                    source[indices] = value;
                }
            }
        }

        /// <summary>
        /// Forms a slice out of the given tensor. The parameters are similar to PyTorch (从给定的张量中形成一个切片. 参数与 PyTorch 相似).
        /// </summary>
        /// <typeparam name="T">The element type (元素类型).</typeparam>
        /// <param name="source">The source (源).</param>
        /// <param name="start">The ranges for the slice (切片的范围).</param>
        /// <returns><see cref="Tensor{T}"/> as a slice of the provided ranges (所提供范围的切片).</returns>
        /// <exception cref="ArgumentOutOfRangeException">start.Length &gt; source.Lengths.Length</exception>
        /// <seealso cref="Tensor{T}.Slice(ReadOnlySpan{NRange})"/>
        public static Tensor<T> SliceTorch<T>(this Tensor<T> source, params scoped ReadOnlySpan<NRange> start) {
            if (source.IsEmpty) {
                throw new ArgumentOutOfRangeException(nameof(source), "source is empty!");
            }
            int rank = source.Lengths.Length;
            if (start.Length > rank) {
                throw new ArgumentOutOfRangeException(nameof(start), "Number of dimensions to slice does not equal the number of dimensions in the span");
            } else if (start.Length == source.Lengths.Length) {
                return source.Slice(start);
            }
            Span<NRange> start2 = stackalloc NRange[rank];
            start.CopyTo(start2);
            for (int i = start.Length; i < rank; ++i) {
                start2[i] = NRange.All;
            }
            return source.Slice(start2);
        }

        /// <summary>
        /// Forms a slice out of the given tensor. The parameters are similar to PyTorch (从给定的张量中形成一个切片. 参数与 PyTorch 相似).
        /// </summary>
        /// <typeparam name="T">The element type (元素类型).</typeparam>
        /// <param name="source">The source (源).</param>
        /// <param name="start">The start indexes for the slice (切片的开始索引).</param>
        /// <returns><see cref="Tensor{T}"/> as a slice of the provided ranges (所提供范围的切片).</returns>
        /// <exception cref="ArgumentOutOfRangeException">start.Length &gt; source.Lengths.Length</exception>
        /// <seealso cref="Tensor{T}.Slice(ReadOnlySpan{nint})"/>
        public static Tensor<T> SliceTorch<T>(this Tensor<T> source, params scoped ReadOnlySpan<nint> start) {
            if (source.IsEmpty) {
                throw new ArgumentOutOfRangeException(nameof(source), "source is empty!");
            }
            int rank = source.Lengths.Length;
            if (start.Length > rank) {
                throw new ArgumentOutOfRangeException(nameof(start), "Number of dimensions to slice does not equal the number of dimensions in the span");
            } else if (start.Length == source.Lengths.Length) {
                return source.Slice(start);
            }
            Span<nint> start2 = stackalloc nint[rank];
            start.CopyTo(start2);
            for (int i = start.Length; i < rank; ++i) {
                start2[i] = default;
            }
            return source.Slice(start2);
        }

        /// <summary>
        /// Forms a slice out of the given tensor. The parameters are similar to PyTorch (从给定的张量中形成一个切片. 参数与 PyTorch 相似).
        /// </summary>
        /// <typeparam name="T">The element type (元素类型).</typeparam>
        /// <param name="source">The source (源).</param>
        /// <param name="startIndex">The start indexes for the slice (切片的开始索引).</param>
        /// <returns><see cref="Tensor{T}"/> as a slice of the provided ranges (所提供范围的切片).</returns>
        /// <exception cref="ArgumentOutOfRangeException">startIndex.Length &gt; source.Lengths.Length</exception>
        /// <seealso cref="Tensor{T}.Slice(ReadOnlySpan{NIndex})"/>
        public static Tensor<T> SliceTorch<T>(this Tensor<T> source, params scoped ReadOnlySpan<NIndex> startIndex) {
            if (source.IsEmpty) {
                throw new ArgumentOutOfRangeException(nameof(source), "source is empty!");
            }
            int rank = source.Lengths.Length;
            if (startIndex.Length > rank) {
                throw new ArgumentOutOfRangeException(nameof(startIndex), "Number of dimensions to slice does not equal the number of dimensions in the span");
            } else if (startIndex.Length == source.Lengths.Length) {
                return source.Slice(startIndex);
            }
            Span<NIndex> start2 = stackalloc NIndex[rank];
            startIndex.CopyTo(start2);
            for (int i = startIndex.Length; i < rank; ++i) {
                start2[i] = NIndex.Start;
            }
            return source.Slice(start2);
        }

        /// <summary>
        /// Forms a slice out of the given tensor. The parameters are similar to PyTorch (从给定的张量中形成一个切片. 参数与 PyTorch 相似).
        /// </summary>
        /// <typeparam name="T">The element type (元素类型).</typeparam>
        /// <param name="source">The source (源).</param>
        /// <param name="start">The ranges for the slice (切片的范围).</param>
        /// <returns><see cref="Tensor{T}"/> as a slice of the provided ranges (所提供范围的切片).</returns>
        /// <exception cref="ArgumentOutOfRangeException">start.Length &gt; source.Lengths.Length</exception>
        /// <seealso cref="TensorSpan{T}.Slice(ReadOnlySpan{NRange})"/>
        public static TensorSpan<T> SliceTorch<T>(this in TensorSpan<T> source, params scoped ReadOnlySpan<NRange> start) {
            if (source.IsEmpty) {
                throw new ArgumentOutOfRangeException(nameof(source), "source is empty!");
            }
            int rank = source.Lengths.Length;
            if (start.Length > rank) {
                throw new ArgumentOutOfRangeException(nameof(start), "Number of dimensions to slice does not equal the number of dimensions in the span");
            } else if (start.Length == source.Lengths.Length) {
                return source.Slice(start);
            }
            Span<NRange> start2 = stackalloc NRange[rank];
            start.CopyTo(start2);
            for (int i = start.Length; i < rank; ++i) {
                start2[i] = NRange.All;
            }
            return source.Slice(start2);
        }

        /*
        /// <summary>
        /// Forms a slice out of the given tensor. The parameters are similar to PyTorch (从给定的张量中形成一个切片. 参数与 PyTorch 相似).
        /// </summary>
        /// <typeparam name="T">The element type (元素类型).</typeparam>
        /// <param name="source">The source (源).</param>
        /// <param name="start">The start indexes for the slice (切片的开始索引).</param>
        /// <returns><see cref="Tensor{T}"/> as a slice of the provided ranges (所提供范围的切片).</returns>
        /// <exception cref="ArgumentOutOfRangeException">start.Length &gt; source.Lengths.Length</exception>
        /// <seealso cref="TensorSpan{T}.Slice(ReadOnlySpan{nint})"/>
        public static TensorSpan<T> SliceTorch<T>(this in TensorSpan<T> source, params scoped ReadOnlySpan<nint> start) {
            if (source.IsEmpty) {
                throw new ArgumentOutOfRangeException(nameof(source), "source is empty!");
            }
            int rank = source.Lengths.Length;
            if (start.Length > rank) {
                throw new ArgumentOutOfRangeException(nameof(start), "Number of dimensions to slice does not equal the number of dimensions in the span");
            } else if (start.Length == source.Lengths.Length) {
                return source.Slice(start);
            }
            Span<nint> start2 = stackalloc nint[rank];
            start.CopyTo(start2);
            for (int i = start.Length; i < rank; ++i) {
                start2[i] = default;
            }
            return source.Slice(start2);
        }
        */

        /// <summary>
        /// Forms a slice out of the given tensor. The parameters are similar to PyTorch (从给定的张量中形成一个切片. 参数与 PyTorch 相似).
        /// </summary>
        /// <typeparam name="T">The element type (元素类型).</typeparam>
        /// <param name="source">The source (源).</param>
        /// <param name="startIndex">The start indexes for the slice (切片的开始索引).</param>
        /// <returns><see cref="Tensor{T}"/> as a slice of the provided ranges (所提供范围的切片).</returns>
        /// <exception cref="ArgumentOutOfRangeException">startIndex.Length &gt; source.Lengths.Length</exception>
        /// <seealso cref="TensorSpan{T}.Slice(ReadOnlySpan{NIndex})"/>
        public static TensorSpan<T> SliceTorch<T>(this in TensorSpan<T> source, params scoped ReadOnlySpan<NIndex> startIndex) {
            if (source.IsEmpty) {
                throw new ArgumentOutOfRangeException(nameof(source), "source is empty!");
            }
            int rank = source.Lengths.Length;
            if (startIndex.Length > rank) {
                throw new ArgumentOutOfRangeException(nameof(startIndex), "Number of dimensions to slice does not equal the number of dimensions in the span");
            } else if (startIndex.Length == source.Lengths.Length) {
                return source.Slice(startIndex);
            }
            Span<NIndex> start2 = stackalloc NIndex[rank];
            startIndex.CopyTo(start2);
            for (int i = startIndex.Length; i < rank; ++i) {
                start2[i] = NIndex.Start;
            }
            return source.Slice(start2);
        }

        /// <summary>
        /// Forms a slice out of the given tensor. The parameters are similar to PyTorch (从给定的张量中形成一个切片. 参数与 PyTorch 相似).
        /// </summary>
        /// <typeparam name="T">The element type (元素类型).</typeparam>
        /// <param name="source">The source (源).</param>
        /// <param name="start">The ranges for the slice (切片的范围).</param>
        /// <returns><see cref="Tensor{T}"/> as a slice of the provided ranges (所提供范围的切片).</returns>
        /// <exception cref="ArgumentOutOfRangeException">start.Length &gt; source.Lengths.Length</exception>
        /// <seealso cref="ReadOnlyTensorSpan{T}.Slice(ReadOnlySpan{NRange})"/>
        public static ReadOnlyTensorSpan<T> SliceTorch<T>(this in ReadOnlyTensorSpan<T> source, params scoped ReadOnlySpan<NRange> start) {
            if (source.IsEmpty) {
                throw new ArgumentOutOfRangeException(nameof(source), "source is empty!");
            }
            int rank = source.Lengths.Length;
            if (start.Length > rank) {
                throw new ArgumentOutOfRangeException(nameof(start), "Number of dimensions to slice does not equal the number of dimensions in the span");
            } else if (start.Length == source.Lengths.Length) {
                return source.Slice(start);
            }
            Span<NRange> start2 = stackalloc NRange[rank];
            start.CopyTo(start2);
            for (int i = start.Length; i < rank; ++i) {
                start2[i] = NRange.All;
            }
            return source.Slice(start2);
        }

        /*
        /// <summary>
        /// Forms a slice out of the given tensor. The parameters are similar to PyTorch (从给定的张量中形成一个切片. 参数与 PyTorch 相似).
        /// </summary>
        /// <typeparam name="T">The element type (元素类型).</typeparam>
        /// <param name="source">The source (源).</param>
        /// <param name="start">The start indexes for the slice (切片的开始索引).</param>
        /// <returns><see cref="Tensor{T}"/> as a slice of the provided ranges (所提供范围的切片).</returns>
        /// <exception cref="ArgumentOutOfRangeException">start.Length &gt; source.Lengths.Length</exception>
        /// <seealso cref="ReadOnlyTensorSpan{T}.Slice(ReadOnlySpan{nint})"/>
        public static ReadOnlyTensorSpan<T> SliceTorch<T>(this in ReadOnlyTensorSpan<T> source, params scoped ReadOnlySpan<nint> start) {
            if (source.IsEmpty) {
                throw new ArgumentOutOfRangeException(nameof(source), "source is empty!");
            }
            int rank = source.Lengths.Length;
            if (start.Length > rank) {
                throw new ArgumentOutOfRangeException(nameof(start), "Number of dimensions to slice does not equal the number of dimensions in the span");
            } else if (start.Length == source.Lengths.Length) {
                return source.Slice(start);
            }
            Span<nint> start2 = stackalloc nint[rank];
            start.CopyTo(start2);
            for (int i = start.Length; i < rank; ++i) {
                start2[i] = default;
            }
            return source.Slice(start2);
        }
        */

        /// <summary>
        /// Forms a slice out of the given tensor. The parameters are similar to PyTorch (从给定的张量中形成一个切片. 参数与 PyTorch 相似).
        /// </summary>
        /// <typeparam name="T">The element type (元素类型).</typeparam>
        /// <param name="source">The source (源).</param>
        /// <param name="startIndex">The start indexes for the slice (切片的开始索引).</param>
        /// <returns><see cref="Tensor{T}"/> as a slice of the provided ranges (所提供范围的切片).</returns>
        /// <exception cref="ArgumentOutOfRangeException">startIndex.Length &gt; source.Lengths.Length</exception>
        /// <seealso cref="ReadOnlyTensorSpan{T}.Slice(ReadOnlySpan{NIndex})"/>
        public static ReadOnlyTensorSpan<T> SliceTorch<T>(this in ReadOnlyTensorSpan<T> source, params scoped ReadOnlySpan<NIndex> startIndex) {
            if (source.IsEmpty) {
                throw new ArgumentOutOfRangeException(nameof(source), "source is empty!");
            }
            int rank = source.Lengths.Length;
            if (startIndex.Length > rank) {
                throw new ArgumentOutOfRangeException(nameof(startIndex), "Number of dimensions to slice does not equal the number of dimensions in the span");
            } else if (startIndex.Length == source.Lengths.Length) {
                return source.Slice(startIndex);
            }
            Span<NIndex> start2 = stackalloc NIndex[rank];
            startIndex.CopyTo(start2);
            for (int i = startIndex.Length; i < rank; ++i) {
                start2[i] = NIndex.Start;
            }
            return source.Slice(start2);
        }

        // -- torch.sum(input, *, dim=None, keepdim=False, out=None)

        /// <summary>
        /// This function is used to compute the sum of all elements in the input tensor. Support dim parameter (此函数用于对输入张量中所有元素计算求和. 支持 dim 参数. 支持 dim 参数). Like `torch.sum`.
        /// </summary>
        /// <typeparam name="T">The element type (元素类型).</typeparam>
        /// <param name="source">The source tensor (源张量).</param>
        /// <param name="dim">Specifies the dimension(s) along which the sum is computed. If not specified, the sum is computed over all elements (指定计算求和所依据的维度. 如果未指定, 则计算所有元素的总和).</param>
        /// <param name="keepdim">Whether the output tensor has dimension retained or not (输出张量是否保留维度).</param>
        /// <param name="pinned">A Boolean whether the underlying data should be pinned or not (一个布尔值，表示是否应固定基础数据).</param>
        /// <returns>Return a new tensor that stores the sum results (返回新张量, 存放了求和结果).</returns>
        public static Tensor<T> SumTorch<T>(this Tensor<T> source, Span<int> dim, bool keepdim = false, bool pinned = false) where T : IAdditionOperators<T, T, T>, IAdditiveIdentity<T, T> {
            return SumTorch(source.AsReadOnlyTensorSpan(), dim, keepdim, pinned);
        }

        /// <summary>
        /// This function is used to compute the sum of all elements in the input tensor. Support dim parameter (此函数用于对输入张量中所有元素计算求和. 支持 dim 参数. 支持 dim 参数). Like `torch.sum`.
        /// </summary>
        /// <typeparam name="T">The element type (元素类型).</typeparam>
        /// <param name="source">The source tensor (源张量).</param>
        /// <param name="dim">Specifies the dimension(s) along which the sum is computed. If not specified, the sum is computed over all elements (指定计算求和所依据的维度. 如果未指定, 则计算所有元素的总和).</param>
        /// <param name="keepdim">Whether the output tensor has dimension retained or not (输出张量是否保留维度).</param>
        /// <param name="pinned">A Boolean whether the underlying data should be pinned or not (一个布尔值，表示是否应固定基础数据).</param>
        /// <returns>Return a new tensor that stores the sum results (返回新张量, 存放了求和结果).</returns>
        public static Tensor<T> SumTorch<T>(this in ReadOnlyTensorSpan<T> source, Span<int> dim, bool keepdim = false, bool pinned = false) where T : IAdditionOperators<T, T, T>, IAdditiveIdentity<T, T> {
            if (source.IsEmpty) {
                return Tensor<T>.Empty;
            }

            // If not specified, the sum is computed over all elements.
            int rank = source.Rank;
            Span<nint> lengthsDst = stackalloc nint[rank];
            if (dim.IsEmpty) {
                T sumData = Tensor.Sum(source);
                if (!keepdim) {
                    lengthsDst = lengthsDst.Slice(0, 1);
                }
                lengthsDst.Fill(1);
                Tensor<T> rt = Tensor.CreateUninitialized<T>(lengthsDst);
                rt.Fill(sumData);
                return rt;
            }

            // lengthsDst
            ReadOnlySpan<nint> lengths = source.Lengths;
            Span<nint> lengthsDstKeep = stackalloc nint[rank];
            nint numel;
            int sumableDim;
            lengthsDst = SumTorch_MakeLengthsDst(lengths, dim, keepdim, lengthsDstKeep, lengthsDst, out numel, out sumableDim);

            // Fill values.
            Tensor<T> dst = Tensor.Create<T>(lengthsDst, pinned);
            if (numel <= 1) {
                T sumData = Tensor.Sum(source);
                dst.Fill(sumData);
            } else {
                TensorSpan<T> dstSpan = dst.AsTensorSpan().Reshape(lengthsDstKeep);
                Span<nint> indicesSrc = stackalloc nint[rank];
                Span<nint> indicesDst = stackalloc nint[rank];
                Span<NRange> rangesSrc = stackalloc NRange[rank];
                indicesSrc.Clear();
                indicesDst.Clear();
                rangesSrc.Fill(NRange.All);
                SumTorch_Core(source, dim, keepdim, dstSpan, sumableDim, 0, indicesSrc, indicesDst, rangesSrc);
            }
            return dst;
        }

        /// <summary>
        /// SumTorch - Make lengthsDst.
        /// </summary>
        /// <param name="lengths">The source lengths (源长度).</param>
        /// <param name="dim">Specifies the dimension(s) along which the sum is computed. If not specified, the sum is computed over all elements (指定计算求和所依据的维度. 如果未指定, 则计算所有元素的总和).</param>
        /// <param name="keepdim">Whether the output tensor has dimension retained or not (输出张量是否保留维度).</param>
        /// <param name="lengthsDstKeep">The destination lengths buffer with keepdim (使用keepdim时的目标长度缓冲区).</param>
        /// <param name="lengthsDst">The destination lengths buffer (目标长度缓冲区).</param>
        /// <param name="numel">Return the elements total number of destination (返回目标的元素总数).</param>
        /// <param name="sumableDim">Return the dimensions that can be summed up (返回可求和的维度).</param>
        /// <returns>Returns the destination lengths (返回目标长度).</returns>
        /// <exception cref="ArgumentOutOfRangeException"></exception>
        private static Span<nint> SumTorch_MakeLengthsDst(in ReadOnlySpan<nint> lengths, Span<int> dim, bool keepdim, Span<nint> lengthsDstKeep, Span<nint> lengthsDst, out nint numel, out int sumableDim) {
            int rank = lengths.Length;
            lengths.CopyTo(lengthsDstKeep);
            for (int i = 0; i < dim.Length; i++) {
                int k = dim[i];
                if (k < 0 || k >= rank) {
                    throw new ArgumentOutOfRangeException(nameof(dim), string.Format("The dim[{0}] ({1}) out of range ([0, {2}))!", i, k, rank));
                }
                lengthsDstKeep[k] = 1;
            }

            // lengthsDst
            numel = 1;
            int rankDst = 0;
            for (int i = 0; i < lengths.Length; ++i) {
                bool isfound = dim.Contains(i);
                if (isfound) {
                    // continue
                } else {
                    lengthsDst[rankDst] = lengths[i];
                    rankDst++;
                    numel *= lengths[i];
                }
            }
            if (keepdim) {
                lengthsDstKeep.CopyTo(lengthsDst);
            } else {
                if (rankDst <= 0) {
                    lengthsDst[0] = 1; // Sum all dims.
                    rankDst = 1;
                }
                lengthsDst = lengthsDst.Slice(0, rankDst);
            }
            // sumableDim
            sumableDim = rank;
            for (int i = rank - 1; i >= 0; --i) {
                bool isfound = dim.Contains(i);
                if (isfound) {
                    sumableDim = i;
                } else {
                    break;
                }
            }
            return lengthsDst;
        }

        /// <summary>
        /// SumTorch - Core.
        /// </summary>
        /// <typeparam name="T">The element type (元素类型).</typeparam>
        /// <param name="source">The source tensor (源张量).</param>
        /// <param name="dim">Specifies the dimension(s) along which the sum is computed. If not specified, the sum is computed over all elements (指定计算求和所依据的维度. 如果未指定, 则计算所有元素的总和).</param>
        /// <param name="keepdim">Whether the output tensor has dimension retained or not (输出张量是否保留维度).</param>
        /// <param name="destination">The destination tensor (目标张量).</param>
        /// <param name="sumableDim">The dimensions that can be summed up (可求和的维度).</param>
        /// <param name="level">Current level (当前层级).</param>
        /// <param name="indicesSrc">Indices of source tensor (源张量的索引).</param>
        /// <param name="indicesDst">Indices of destination tensor (目标张量的索引).</param>
        /// <param name="rangesSrc">Sum ranges of source tensor (源张量的求和范围).</param>
        private static void SumTorch_Core<T>(in ReadOnlyTensorSpan<T> source, Span<int> dim, bool keepdim, in TensorSpan<T> destination, int sumableDim, int level, Span<nint> indicesSrc, Span<nint> indicesDst, Span<NRange> rangesSrc) where T : IAdditionOperators<T, T, T>, IAdditiveIdentity<T, T> {
            int dimMax = source.Lengths.Length - 1;
            int levelNext = level + 1;
            nint countSrc = source.Lengths[level];
            nint countDst = destination.Lengths[level];
            indicesDst[level] = 0;
            if (level >= sumableDim) {
                // Batch sum.
                ReadOnlyTensorSpan<T> tensorSpan = source.Slice(rangesSrc);
                T sumData = Tensor.Sum(tensorSpan);
                destination[indicesDst] += sumData;
            } else {
                // Base sum.
                for (nint i = 0; i < countSrc; ++i) {
                    indicesSrc[level] = i;
                    if (countDst == countSrc) {
                        indicesDst[level] = i;
                    }
                    if (level < dimMax) {
                        rangesSrc[level] = new NRange(i, i + 1);
                        SumTorch_Core(source, dim, keepdim, destination, sumableDim, levelNext, indicesSrc, indicesDst, rangesSrc);
                    } else {
                        destination[indicesDst] += source[indicesSrc];
                    }
                }
            }
        }

        //public static void SumTorch<T>(this in ReadOnlyTensorSpan<T> source, Span<int> dim, bool keepdim, in TensorSpan<T> destination) where T : IAdditionOperators<T, T, T>, IAdditiveIdentity<T, T> {
        //}


        /// <inheritdoc cref="To1DArray{T}(in ReadOnlyTensorSpan{T})"/>
        public static T[] To1DArray<T>(this Tensor<T> source) {
            return To1DArray(source.AsReadOnlyTensorSpan());
        }

        /// <summary>
        /// Create 1-dimensional array by tensor. It can also convert N-dimensional tensor into 1-dimensional arrays (根据张量创建1维数组. 它还能将多维张量转为1维数组).
        /// </summary>
        /// <typeparam name="T">The element type (元素类型).</typeparam>
        /// <param name="source">The source (源).</param>
        /// <returns>Returns new array (返回新数组).</returns>
        /// <seealso cref="ToNDArray{T}(in ReadOnlyTensorSpan{T})"/>
        public static T[] To1DArray<T>(this in ReadOnlyTensorSpan<T> source) {
            if (source.IsEmpty) {
                return Array.Empty<T>();
            }
            nint flattenedLength = source.FlattenedLength;
            //if (flattenedLength > Array.MaxLength) {
            //    throw new ArgumentOutOfRangeException(string.Format("Flattened length ({0}) is out of array max length!", flattenedLength));
            //}
            ReadOnlySpan<nint> lengths = source.Lengths;
            T[] rt = new T[flattenedLength];
            TensorSpan<T> tensorSpan = new TensorSpan<T>(rt, ReadOnlySpan<int>.Empty, lengths, []);
            source.CopyTo(tensorSpan);
            return rt;
        }

        /// <inheritdoc cref="To2DArray{T}(in ReadOnlyTensorSpan{T})"/>
        public static T[,] To2DArray<T>(this Tensor<T> source) {
            return To2DArray(source.AsReadOnlyTensorSpan());
        }

        /// <summary>
        /// Create 2-dimensional array by tensor. The rank of a tensor must be equal to 2 (根据张量创建2维数组. 张量的秩必须为 2).
        /// </summary>
        /// <typeparam name="T">The element type (元素类型).</typeparam>
        /// <param name="source">The source (源).</param>
        /// <returns>Returns new array (返回新数组).</returns>
        /// <seealso cref="ToNDArray{T}(in ReadOnlyTensorSpan{T})"/>
        public static T[,] To2DArray<T>(this in ReadOnlyTensorSpan<T> source) {
            if (2 != source.Rank) {
                throw new ArgumentException(string.Format("The rank({0}) of a tensor must be equal to 2!", source.Rank));
            }
            ReadOnlySpan<nint> lengths = source.Lengths;
            //long[] lengthsLong = new long[lengths.Length];
            //TensorPrimitives.ConvertChecked(lengths, lengthsLong.AsSpan());
            var rt = new T[lengths[0], lengths[1]];
            TensorSpan<T> tensorSpan = new TensorSpan<T>(rt);
            source.CopyTo(tensorSpan);
            return rt;
        }

        /// <inheritdoc cref="To3DArray{T}(in ReadOnlyTensorSpan{T})"/>
        public static T[,,] To3DArray<T>(this Tensor<T> source) {
            return To3DArray(source.AsReadOnlyTensorSpan());
        }

        /// <summary>
        /// Create 3-dimensional array by tensor. The rank of a tensor must be equal to 3 (根据张量创建3维数组. 张量的秩必须为 3).
        /// </summary>
        /// <typeparam name="T">The element type (元素类型).</typeparam>
        /// <param name="source">The source (源).</param>
        /// <returns>Returns new array (返回新数组).</returns>
        /// <seealso cref="ToNDArray{T}(in ReadOnlyTensorSpan{T})"/>
        public static T[,,] To3DArray<T>(this in ReadOnlyTensorSpan<T> source) {
            if (3 != source.Rank) {
                throw new ArgumentException(string.Format("The rank({0}) of a tensor must be equal to 3!", source.Rank));
            }
            ReadOnlySpan<nint> lengths = source.Lengths;
            //long[] lengthsLong = new long[lengths.Length];
            //TensorPrimitives.ConvertChecked(lengths, lengthsLong.AsSpan());
            var rt = new T[lengths[0], lengths[1], lengths[2]];
            TensorSpan<T> tensorSpan = new TensorSpan<T>(rt);
            source.CopyTo(tensorSpan);
            return rt;
        }

        /// <inheritdoc cref="To4DArray{T}(in ReadOnlyTensorSpan{T})"/>
        public static T[,,,] To4DArray<T>(this Tensor<T> source) {
            return To4DArray(source.AsReadOnlyTensorSpan());
        }

        /// <summary>
        /// Create 4-dimensional array by tensor. The rank of a tensor must be equal to 4 (根据张量创建4维数组. 张量的秩必须为 4).
        /// </summary>
        /// <typeparam name="T">The element type (元素类型).</typeparam>
        /// <param name="source">The source (源).</param>
        /// <returns>Returns new array (返回新数组).</returns>
        /// <seealso cref="ToNDArray{T}(in ReadOnlyTensorSpan{T})"/>
        public static T[,,,] To4DArray<T>(this in ReadOnlyTensorSpan<T> source) {
            if (4 != source.Rank) {
                throw new ArgumentException(string.Format("The rank({0}) of a tensor must be equal to 4!", source.Rank));
            }
            ReadOnlySpan<nint> lengths = source.Lengths;
            //long[] lengthsLong = new long[lengths.Length];
            //TensorPrimitives.ConvertChecked(lengths, lengthsLong.AsSpan());
            var rt = new T[lengths[0], lengths[1], lengths[2], lengths[3]];
            TensorSpan<T> tensorSpan = new TensorSpan<T>(rt);
            source.CopyTo(tensorSpan);
            return rt;
        }

        /// <inheritdoc cref="ToNDArray{T}(in ReadOnlyTensorSpan{T})"/>
        [RequiresDynamicCode("Array.CreateInstance: The code for an array of the specified type might not be available.")]
        public static Array ToNDArray<T>(this Tensor<T> source) {
            return ToNDArray(source.AsReadOnlyTensorSpan());
        }

        /// <summary>
        /// Create N-dimensional array by tensor (根据张量创建N维数组).
        /// </summary>
        /// <typeparam name="T">The element type (元素类型).</typeparam>
        /// <param name="source">The source (源).</param>
        /// <returns>Returns new array (返回新数组).</returns>
        /// <seealso cref="To1DArray{T}(in ReadOnlyTensorSpan{T})"/>
        /// <seealso cref="To2DArray{T}(in ReadOnlyTensorSpan{T})"/>
        /// <seealso cref="To3DArray{T}(in ReadOnlyTensorSpan{T})"/>
        /// <seealso cref="To4DArray{T}(in ReadOnlyTensorSpan{T})"/>
        [RequiresDynamicCode("Array.CreateInstance: The code for an array of the specified type might not be available.")]
        public static Array ToNDArray<T>(this in ReadOnlyTensorSpan<T> source) {
            ReadOnlySpan<nint> lengths = source.Lengths;
            long[] lengthsLong = new long[lengths.Length];
            TensorPrimitives.ConvertChecked(lengths, lengthsLong.AsSpan());
            Array rt = Array.CreateInstance(typeof(T), lengthsLong);
            TensorSpan<T> tensorSpan = new TensorSpan<T>(rt);
            source.CopyTo(tensorSpan);
            return rt;
        }

    }

#pragma warning restore SYSLIB5001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.
}

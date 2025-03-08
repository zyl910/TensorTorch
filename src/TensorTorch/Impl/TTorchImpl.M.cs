using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics.Tensors;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace Zyl.TensorTorch.Impl {
    partial class TTorchImpl {
#pragma warning disable SYSLIB5001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.

        /// <inheritdoc cref="TTorch.MultiplyVector{T}(in ReadOnlyTensorSpan{T}, in ReadOnlyTensorSpan{T}, in TensorSpan{T})"/>
        public static void MultiplyVector<T>(in ReadOnlyTensorSpan<T> input, in ReadOnlyTensorSpan<T> vec, in TensorSpan<T> output)
                where T : IAdditionOperators<T, T, T>, IAdditiveIdentity<T, T>, IMultiplicativeIdentity<T, T>, IMultiplyOperators<T, T, T> {
            if (!TensorCheck.IsMatrix(input.Lengths, out nint rows, out nint cols)) {
                throw new ArgumentException(string.Format("The input parameter ({0}) is not a matrix!", TensorCheck.ToString(input.Lengths)), nameof(input));
            }
            if (!TensorCheck.IsVector(vec.Lengths, out nint numel, out bool isColumn)) {
                throw new ArgumentException(string.Format("The vec parameter ({0}) is not a vector!", TensorCheck.ToString(vec.Lengths)), nameof(vec));
            }
            if (cols != numel) {
                throw new ArgumentException(string.Format("The number of elements in the vec parameter ({0}) does not match the number of columns in the input matrix ({1})!", TensorCheck.ToString(vec.Lengths), TensorCheck.ToString(input.Lengths)), nameof(vec));
            }
            if (!TensorCheck.IsVector(output.Lengths, out nint numelOutput, out bool isColumnOutput)) {
                throw new ArgumentException(string.Format("The output parameter ({0}) is not a vector!", TensorCheck.ToString(output.Lengths)), nameof(output));
            }
            if (numelOutput < rows) {
                throw new ArgumentException(string.Format("The number of elements in the output parameter ({0}) does less then the number of rows in the input matrix ({1})!", TensorCheck.ToString(output.Lengths), TensorCheck.ToString(input.Lengths)), nameof(output));
            }
            if (isColumnOutput) {
                MultiplyVector_Body(input, vec, output.Reshape([numelOutput]), rows, cols);
            } else {
                MultiplyVector_Body(input, vec, output, rows, cols);
            }
            _ = isColumn;
        }

        /// <inheritdoc cref="MultiplyVector{T}(in ReadOnlyTensorSpan{T}, in ReadOnlyTensorSpan{T}, in TensorSpan{T})"/>
        /// <param name="rows">Number of rows (行数).</param>
        /// <param name="cols">Number of columns (列数).</param>
        public static void MultiplyVector_Body<T>(in ReadOnlyTensorSpan<T> input, in ReadOnlyTensorSpan<T> vec, in TensorSpan<T> output, nint rows, nint cols)
                where T : IAdditionOperators<T, T, T>, IAdditiveIdentity<T, T>, IMultiplicativeIdentity<T, T>, IMultiplyOperators<T, T, T> {
            Debug.Assert(2 == input.Rank, "The input parameter is not a matrix!");
            scoped ReadOnlyTensorSpan<T> vec1;
            if (vec.Rank > 1) {
                vec1 = vec.Reshape([vec.FlattenedLength]);
            } else {
                vec1 = vec;
            }
            NRange rangeCols = new NRange(0, cols); // NRange.All;
            for (nint i = 0; i < rows; ++i) {
                NRange rangeRow = new NRange(i, i + 1);
                T item = Tensor.Dot(input.Slice(rangeRow, rangeCols), vec1);
                output[i] = item;
            }
        }

        /// <inheritdoc cref="TTorch.MultiplyVector{T}(in ReadOnlyTensorSpan{T}, in ReadOnlyTensorSpan{T}, bool)"/>
        public static Tensor<T> MultiplyVector<T>(in ReadOnlyTensorSpan<T> input, in ReadOnlyTensorSpan<T> vec, bool pinned = false)
                where T : IAdditionOperators<T, T, T>, IAdditiveIdentity<T, T>, IMultiplicativeIdentity<T, T>, IMultiplyOperators<T, T, T> {
            if (!TensorCheck.IsMatrix(input.Lengths, out nint rows, out nint cols)) {
                throw new ArgumentException(string.Format("The input parameter ({0}) is not a matrix!", TensorCheck.ToString(input.Lengths)), nameof(input));
            }
            if (!TensorCheck.IsVector(vec.Lengths, out nint numel, out bool isColumn)) {
                throw new ArgumentException(string.Format("The vec parameter ({0}) is not a vector!", TensorCheck.ToString(vec.Lengths)), nameof(vec));
            }
            if (cols != numel) {
                throw new ArgumentException(string.Format("The number of elements in the vec parameter ({0}) does not match the number of columns in the input matrix ({1})!", TensorCheck.ToString(vec.Lengths), TensorCheck.ToString(input.Lengths)), nameof(vec));
            }
            Tensor<T> output = Tensor.CreateUninitialized<T>([rows], pinned);
            MultiplyVector_Body(input, vec, output.AsTensorSpan(), rows, cols);
            _ = isColumn;
            return output;
        }

#pragma warning restore SYSLIB5001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.
    }
}

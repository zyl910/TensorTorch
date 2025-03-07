using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Zyl.TensorTorch {
    /// <summary>
    /// Tensor check tool (张量检查工具).
    /// </summary>
    public static class TensorCheck {

        /// <summary>
        /// Is it a matrix (是不是矩阵).
        /// </summary>
        /// <param name="lengths">The shape of tensor (张量的形状).</param>
        /// <param name="rows">Number of rows (行数).</param>
        /// <param name="cols">Number of columns (列数).</param>
        /// <returns>Return true when it is a matrix, otherwise return false (是矩阵时返回 true, 否则返回 false).</returns>
        public static bool IsMatrix(ReadOnlySpan<IntPtr> lengths, out nint rows, out nint cols) {
            if (2 == lengths.Length) {
                rows = lengths[0];
                cols = lengths[1];
                return true;
            }
            rows = 0;
            cols = 0;
            return false;
        }

        /// <summary>
        /// Is it a vector (是不是向量).
        /// </summary>
        /// <param name="lengths">The shape of tensor (张量的形状).</param>
        /// <param name="numel">The number of elements in a vector (向量的元素个数).</param>
        /// <param name="isColumn">Is it a column vector (是不是列向量).</param>
        /// <returns></returns>
        public static bool IsVector(ReadOnlySpan<IntPtr> lengths, out nint numel, out bool isColumn) {
            if (1 == lengths.Length) {
                numel = lengths[0];
                isColumn = false;
                return true;
            } else if (2 == lengths.Length && 1 == lengths[1]) {
                numel = lengths[0];
                isColumn = true;
                return true;
            }
            numel = 0;
            isColumn = false;
            return false;
        }

    }
}

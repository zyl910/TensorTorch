using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Zyl.TensorTorch.Impl {
    partial class TTorchImpl {

        /// <inheritdoc cref="TTorch.ToString{T}(StringBuilder, ReadOnlySpan{T}, string?)"/>
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

        /// <inheritdoc cref="TTorch.ToString{T}(ReadOnlySpan{T}, string?)"/>
        public static string ToString<T>(ReadOnlySpan<T> source, string? separator = null) {
            StringBuilder builder = new StringBuilder();
            ToString(builder, source, separator);
            return builder.ToString();
        }

    }
}

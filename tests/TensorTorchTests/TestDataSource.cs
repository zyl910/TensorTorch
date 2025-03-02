#if NET5_0_OR_GREATER
#define BCL_TYPE_HALF
#endif // NET5_0_OR_GREATER
#if NET7_0_OR_GREATER
#define BCL_TYPE_INT128
#endif // NET7_0_OR_GREATER

using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Zyl.TensorTorch.Tests {

    /// <summary>
    /// Test data source.
    /// </summary>
    public static class TestDataSource {
        public static readonly IEnumerable<TestCaseData> _emptyList = new TestCaseData[0];

        /// <summary>
        /// Use extend float types (使用扩展的浮点数类型). e.g. <see cref="Half"/>.
        /// </summary>
        /// <returns>Returns <see cref="TestCaseData"/> of extend float types.</returns>
        public static IEnumerable<TestCaseData> UseExtendFloats() {
            return UseHalf();
        }

        /// <summary>
        /// Use extend integer types (使用扩展的整数类型). e.g. <see cref="Int128"/>, <see cref="UInt128"/>.
        /// </summary>
        /// <returns>Returns <see cref="TestCaseData"/> of extend integer types.</returns>
        public static IEnumerable<TestCaseData> UseExtendInts() {
            return UseInt128()
                .Concat(UseUInt128())
                ;
        }

        /// <summary>
        /// Use <see cref="Half"/> type.
        /// </summary>
        /// <returns>Returns <see cref="TestCaseData"/> of <see cref="Half"/> type.</returns>
        public static IEnumerable<TestCaseData> UseHalf() {
#if BCL_TYPE_HALF
            yield return new TestCaseData((Half)11);
#else
            return _emptyList;
#endif // BCL_TYPE_HALF
        }

        /// <summary>
        /// Use <see cref="Int128"/> type.
        /// </summary>
        /// <returns>Returns <see cref="TestCaseData"/> of <see cref="Int128"/> type.</returns>
        public static IEnumerable<TestCaseData> UseInt128() {
#if BCL_TYPE_INT128
            yield return new TestCaseData((Int128)13);
#else
            return _emptyList;
#endif // BCL_TYPE_INT128
        }

        /// <summary>
        /// Use <see cref="UInt128"/> type.
        /// </summary>
        /// <returns>Returns <see cref="TestCaseData"/> of <see cref="UInt128"/> type.</returns>
        public static IEnumerable<TestCaseData> UseUInt128() {
#if BCL_TYPE_INT128
            yield return new TestCaseData((UInt128)14);
#else
            return _emptyList;
#endif // BCL_TYPE_INT128
        }

    }

}

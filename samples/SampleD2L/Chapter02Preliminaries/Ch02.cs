using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Zyl.SampleD2L.Chapter02Preliminaries {
    internal class Ch02 {
        public static void Output(TextWriter writer) {
            writer.WriteLine("## Chapter2 Preliminaries");
            Ch0201Ndarray.Output(writer);
            Ch0202Pandas.Output(writer);
            Ch0203LinearAlgebra.Output(writer);
            // done.
            writer.WriteLine();
        }
    }
}

using Zyl.SampleD2L.Chapter02Preliminaries;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Zyl.SampleD2L {
    /// <summary>
    /// Dive into Deep Learning (动手学深度学习): https://courses.d2l.ai/zh-v2
    /// </summary>
    internal class D2LMain {
        public static void Output(TextWriter writer) {
            writer.WriteLine("# Dive into Deep Learning (动手学深度学习): https://courses.d2l.ai/zh-v2");
            Ch02.Output(writer);
            // done.
            //writer.WriteLine();
        }
    }
}

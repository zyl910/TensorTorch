
namespace Zyl.SampleD2L {
    internal class Program {
        static void Main(string[] args) {
            TextWriter writer = Console.Out;
            writer.WriteLine("SampleD2L");
            writer.WriteLine(string.Format("RuntimeInformation.FrameworkDescription:\t{0}", System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription));
            writer.WriteLine(string.Format("RuntimeInformation.OSArchitecture:\t{0}", System.Runtime.InteropServices.RuntimeInformation.OSArchitecture));
            writer.WriteLine(string.Format("RuntimeInformation.OSDescription:\t{0}", System.Runtime.InteropServices.RuntimeInformation.OSDescription)); // Same Environment.OSVersion. It's more accurate.
            writer.WriteLine(string.Format("RuntimeInformation.RuntimeIdentifier:\t{0}", System.Runtime.InteropServices.RuntimeInformation.RuntimeIdentifier)); // e.g. win10-x64
            writer.WriteLine();

            D2LMain.Output(writer);
        }
    }
}

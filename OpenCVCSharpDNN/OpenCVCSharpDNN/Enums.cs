using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OpenCVCSharpDNN
{
    public enum BackendEnums : int
    {
        Default = 0,
        Halide = 1,
        InferenceEngine = 2,
        OpenCV = 3
    }

    public enum TargetEnums : int
    {
        Cpu = 0,
        OpenCL = 1,
        OpenCL_FP16 = 2,
        MyRiad = 3
    }
}

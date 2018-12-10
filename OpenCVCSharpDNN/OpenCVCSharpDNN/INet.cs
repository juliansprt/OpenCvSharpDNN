using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OpenCVCSharpDNN
{
    public interface INet : IDisposable
    {
        void Initialize(string pathModel, string pathConfig, string[] labels, BackendEnums backend = BackendEnums.Default, TargetEnums target = TargetEnums.Cpu);

        void Initialize(string pathModel, string pathConfig, string pathLabels, BackendEnums backend = BackendEnums.Default, TargetEnums target = TargetEnums.Cpu);


        NetResult[] Detect(Bitmap img, float minProbability = 0.3f, string[] labelsFilters = null);
    }
}

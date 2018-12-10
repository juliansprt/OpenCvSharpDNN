using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using OpenCvSharp.Dnn;
using System.Drawing;
using System.Diagnostics;

namespace OpenCVCSharpDNN
{
    public abstract class NetCustom : INet
    {
        protected Net network = null;

        public bool Initialized { get; protected set; }

        public string[] Labels { get; protected set; }

        public NetResult[] Detect(Bitmap img, float minProbability = 0.3F, string[] labelsFilters = null)
        {
            if (!Initialized)
                throw new Exception("The model has not yet initialized or is empty.");

            Stopwatch watch = new Stopwatch();
            watch.Start();
            var result = BeginDetect(img, minProbability, labelsFilters);
            watch.Stop();
            Debug.WriteLine($"The detect of the model {this.GetType().Name} has taken {watch.ElapsedMilliseconds} milliseconds");
            return result;
        }

        protected abstract NetResult[] BeginDetect(Bitmap img, float minProbability = 0.3F, string[] labelsFilters = null);


        protected abstract void InitializeModel(string pathModel, string pathConfig);

        public void Dispose()
        {
            if(network != null)
            {
                network.Dispose();
            }
        }

        public void Initialize(string pathModel, string pathConfig, string[] labels, BackendEnums backend = BackendEnums.Default, TargetEnums target = TargetEnums.Cpu)
        {
            Stopwatch watch = new Stopwatch();
            watch.Start();

            if (!File.Exists(pathModel))
                throw new FileNotFoundException("The file model has not found", pathModel);
            if (!File.Exists(pathConfig))
                throw new FileNotFoundException("The file config has not found", pathConfig);

            InitializeModel(pathModel, pathConfig);
            if (network == null || network.Empty())
                throw new Exception("The model has not yet initialized or is empty.");
            network.SetPreferableBackend((int)backend);
            network.SetPreferableTarget((int)target);

            this.Labels = labels;
            this.Initialized = true;

            watch.Stop();

            Debug.WriteLine($"Load time of the model {this.GetType().Name} has taken {watch.ElapsedMilliseconds} milliseconds");
        }

        public void Initialize(string pathModel, string pathConfig, string pathLabels, BackendEnums backend = BackendEnums.Default, TargetEnums target = TargetEnums.Cpu)
        {
            if (!File.Exists(pathLabels))
                throw new FileNotFoundException("The file of labels not foud", pathLabels);

            Initialize(pathModel, pathConfig, File.ReadAllLines(pathLabels), backend, target);
        }
    }
}

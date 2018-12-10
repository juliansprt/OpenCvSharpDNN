using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OpenCVCSharpDNN.Impl
{

    /// <summary>
    /// Implementation for Caffe Age Gender
    /// Ref: https://talhassner.github.io/home/publication/2015_CVPR
    /// </summary>
    public class NetCaffeAgeGender : NetCustom
    {
        /// <summary>
        /// Scale for blob image
        /// </summary>
        public double Scale { get; set; }


        /// <summary>
        /// Begin detect pass the blob to the dnn network
        /// </summary>
        /// <param name="img">Image to detect</param>
        /// <param name="minProbability">Min probability to get label</param>
        /// <param name="labelsFilters">Label to filters</param>
        /// <returns>Net Result</returns>
        protected override NetResult[] BeginDetect(Bitmap img, float minProbability = 0.3F, string[] labelsFilters = null)
        {
            using (Mat mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(img))
            {
                //Create a blob
                var blob = CvDnn.BlobFromImage(mat, Scale, size: new OpenCvSharp.Size(256, 256), mean: new Scalar(78.4263377603, 87.7689143744, 114.895847746), swapRB: false);

                //Set the input for the layer "data"
                network.SetInput(blob, "data");

                //Get result of the layer prob
                var prop = network.Forward("prob");

                //Get the maxLoc and max for probability and the label index
                prop.MinMaxLoc(out double min, out double max, out OpenCvSharp.Point minLoc, out OpenCvSharp.Point maxLoc);

                return new NetResult[]
                {
                    new NetResult()
                    {
                        Label = Labels[maxLoc.X],
                        Probability = max
                    }
                };
            }
        }

        protected override void InitializeModel(string pathModel, string pathConfig)
        {

            //Initialize caffe model
            network = CvDnn.ReadNetFromCaffe(pathConfig, pathModel);

            this.Scale = 1;
        }
    }
}

using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OpenCVCSharpDNN.Impl
{ 

    /// <summary>
    /// Implementation of the YoloV3
    /// </summary>
    public class NetYoloV3 : NetCustom
    {
        /// <summary>
        /// Scale for the blob
        /// </summary>
        public double Scale { get; set; }

        /// <summary>
        /// The prefix of the out layer result
        /// </summary>
        const int Prefix = 5;

        /// <summary>
        /// Values of the config
        /// </summary>
        private string[] valuesConfig;


        protected override NetResult[] BeginDetect(Bitmap img, float minProbability = 0.3F, string[] labelsFilters = null)
        {

            //Extract width and height from config file
            ExtractValueFromConfig("width", out int widthBlob);
            ExtractValueFromConfig("height", out int heightBlob);

            using (Mat mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(img))
            {

                //Create the blob
                var blob = CvDnn.BlobFromImage(mat, Scale, size: new OpenCvSharp.Size(widthBlob, heightBlob), crop: false);

                //Set blob of a default layer
                network.SetInput(blob);

                //Get all out layers
                string[] outLayers = network.GetUnconnectedOutLayersNames();

                //Initialize all blobs for the all out layers
                Mat[] result = new Mat[outLayers.Length];
                for (int i = 0; i < result.Length; i++)
                {
                    result[i] = new Mat();
                }

                ///Execute all out layers
                network.Forward(result, outLayers);

                List<NetResult> netResults = new List<NetResult>();
                foreach (var item in result)
                {
                    for (int i = 0; i < item.Rows; i++)
                    {
                        //Get the max loc and max of the col range by prefix result
                        Cv2.MinMaxLoc(item.Row[i].ColRange(Prefix, item.Cols), out double min, out double max, out OpenCvSharp.Point minLoc, out OpenCvSharp.Point maxLoc);

                        //Validate the min probability
                        if(max >= minProbability)
                        {
                            //The label is the max Loc
                            string label = Labels[maxLoc.X];
                            if(labelsFilters != null)
                            {
                                if(!labelsFilters.Contains(label))
                                {
                                    continue;
                                }
                            }

                            //The probability is the max value
                            double probability = max;

                            //Center BoundingBox X is the 0 index result
                            int centerX = Convert.ToInt32(item.At<float>(i, 0) * (float)mat.Width);
                            //Center BoundingBox X is the 1 index result
                            int centerY = Convert.ToInt32(item.At<float>(i, 1) * (float)mat.Height);
                            //Width BoundingBox is the 2 index result
                            int width = Convert.ToInt32(item.At<float>(i, 2) * (float)mat.Width);
                            //Height BoundingBox is the 2 index result
                            int height = Convert.ToInt32(item.At<float>(i, 3) * (float)mat.Height);

                            //Build NetResult
                            netResults.Add(NetResult.Build(centerX, centerY, width, height, label, probability));

                        }
                    }
                }

                return netResults.ToArray();
            }
        }

        /// <summary>
        /// Initialize the model
        /// </summary>
        /// <param name="pathModel"></param>
        /// <param name="pathConfig"></param>
        protected override void InitializeModel(string pathModel, string pathConfig)
        {
            valuesConfig = File.ReadAllLines(pathConfig);

            ///Initialize darknet network
            network = Net.ReadNetFromDarknet(pathConfig, pathModel);

            //Set the scale 1 / 255
            this.Scale = 0.00392;
        }

        private void ExtractValueFromConfig<t>(string item, out t value)
        {
            if (valuesConfig == null || valuesConfig.Length == 0)
                throw new NullReferenceException("The file of config has empty");

            string line = valuesConfig.FirstOrDefault(p => p.ToLower().Contains(item.ToLower()));
            if (string.IsNullOrWhiteSpace(line))
                throw new NullReferenceException($" The item {item} not exits in the file.");

            string strValue = line.Split('=')[1];

            value = (t)Convert.ChangeType(strValue, typeof(t));
        }
    }
}

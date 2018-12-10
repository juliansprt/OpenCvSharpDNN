using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OpenCVCSharpDNN
{

    /// <summary>
    /// Net result
    /// </summary>
    public class NetResult
    {
        /// <summary>
        /// Bounding Box
        /// </summary>
        public Rectangle Rectangle { get; set; }
        /// <summary>
        /// Probability of label
        /// </summary>
        public double Probability { get; set; }
        /// <summary>
        /// Label of the bounding Box
        /// </summary>
        public string Label { get; set; }

        /// <summary>
        /// Build from values
        /// </summary>
        /// <param name="centerX">Center of bounding box</param>
        /// <param name="centerY">Center of bounding box</param>
        /// <param name="width">Width of bounding box</param>
        /// <param name="height">Height of bounding box</param>
        /// <param name="label">Label of the bounding box</param>
        /// <param name="probability">Probability</param>
        /// <returns></returns>
        internal static NetResult Build(int centerX, int centerY, int width, int height, string label, double probability)
        {
            int w = width / 2;
            int h = height / 2;

            return new NetResult()
            {
                Label = label,
                Probability = probability,
                Rectangle = new Rectangle(centerX - w, centerY - h, width, height)
            };
        }
    }
}

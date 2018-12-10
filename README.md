# OpenCvSharp DNN

Example of implementation of YoloV3 and Caffe in [OpenCvSharp](https://github.com/shimat/opencvsharp)
## Requirements

 - Net Framework 4.7.2
 - OpenCvSharp - _[opencvsharp3](https://github.com/shimat/opencvsharp)_

## Usage

This is a implementation usage in YoloV3 and Caffe models
 ```csharp
            //Directory contains the models and configuration files
            string dir = System.IO.Path.Combine(Directory.GetCurrentDirectory(), "data");

            //Model of YoloV3
            string model = System.IO.Path.Combine(dir, "yolov3.weights");
            string cfg = System.IO.Path.Combine(dir, "yolov3.cfg");
            string labelsYolo = System.IO.Path.Combine(dir, "coco.names");


            //Model of face
            string modelFace = System.IO.Path.Combine(dir, "yolov3-wider_16000.weights");
            string cfgFace = System.IO.Path.Combine(dir, "yolov3-face.cfg");

            //Model of Gender classifaction
            string modelGenderCaffe = System.IO.Path.Combine(dir, "gender_net.caffemodel");
            string cfgGenderCaffe = System.IO.Path.Combine(dir, "deploy_gender.prototxt");

            //Image Path
            string testImage = System.IO.Path.Combine(dir, "friends.jpg");


            using (NetYoloV3 yoloV3 = new NetYoloV3())
            using (NetYoloV3 yoloV3Faces = new NetYoloV3())
            using (NetCaffeAgeGender caffeGender = new NetCaffeAgeGender())
            using (Bitmap bitmap = new Bitmap(testImage))
            using (Bitmap resultImage = new Bitmap(testImage))
            {

                //Initialize models
                yoloV3.Initialize(model, cfg, labelsYolo);
                yoloV3Faces.Initialize(modelFace, cfgFace, new string[] { "faces" });
                caffeGender.Initialize(modelGenderCaffe, cfgGenderCaffe, new string[] { "Male", "Female" });


                //Get result of YoloV3
                NetResult[] resultPersons = yoloV3.Detect(bitmap, labelsFilters: new string[] { "person" });


                //Get result of YoloV3 faces train
                NetResult[] resultFaces = yoloV3Faces.Detect(bitmap);

                using (Graphics canvas = Graphics.FromImage(resultImage))
                {
                    Font font = new Font(FontFamily.GenericSansSerif, 15);


                    foreach (NetResult item in resultFaces)
                    {
                        //Create a roi by each faces
                        using (Bitmap roi = (Bitmap)bitmap.Clone(item.Rectangle, bitmap.PixelFormat))
                        {
                            NetResult resultGender = caffeGender.Detect(roi).FirstOrDefault();

                            canvas.DrawString($"{resultGender.Label} {resultGender.Probability:0.0%}",
                                font,
                                new SolidBrush(Color.Green),
                                item.Rectangle.X - font.GetHeight(), item.Rectangle.Y - font.GetHeight());

                        }

                        canvas.DrawRectangle(new Pen(Color.Red, 2), item.Rectangle);
                    }

                    canvas.Save();
                }

                resultImage.Save(Path.Combine(dir, "result.jpg"));

            }
```
## Pre-trained models
You can download the pre-trained models in these links:

Pre-trained model | Link 
--- | --- 
YoloV3 | https://pjreddie.com/darknet/yolo/ 
YoloV3 Faces | http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/index.html 
Caffe Age and Gender Classification | https://talhassner.github.io/home/publication/2015_CVPR 


## Results

For the next image:

![Input image](https://raw.githubusercontent.com/juliansprt/OpenCvSharpDNN/master/input_image.jpg)

This is the result:
![Input image](https://raw.githubusercontent.com/juliansprt/OpenCvSharpDNN/master/result_image.jpg)

And this is the diagnostics the time of execution:
The model NetYoloV3 - Faces has taken **1609 milliseconds**
The model NetCaffeGender has taken **45 milliseconds**

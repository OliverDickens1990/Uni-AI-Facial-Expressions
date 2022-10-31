using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using Prediction;
using DlibDotNet;
using DlibDotNet.Extensions;
using Dlib = DlibDotNet.Dlib;


namespace FeatureExtractionLib
{
    // https://docs.microsoft.com/en-us/dotnet/api/opentk.vector2.normalize?view=xamarin-ios-sdk-12
    struct Vector2
    {
        public Vector2(double x, double y)
        {
            X = x;
            Y = y;
        }

        public double X { get; }
        public double Y { get; }

        // Distance function
        public static double Distance(Vector2 a, Vector2 b)
        {
            return (Math.Sqrt(Math.Pow(a.X - b.X, 2) + Math.Pow(a.Y - b.Y, 2)));
        }

        // Magnitude
        public static double Magnitude(Vector2 a)
        {
            return (Math.Sqrt(Math.Pow(a.X, 2) + Math.Pow(a.Y, 2)));
        }

        // Normalise
        public static Vector2 Normalise(Vector2 a)
        {
            return (new Vector2(a.X / Magnitude(a), a.Y / Magnitude(a)));
        }
    }
    // Extraction Headers
    class FeatureExtraction
    {
        public void CreateNewCSVFileToExtractTo(string fileName)
        {
            // Header definition of the CSV file
            string header = "Label,LeftEyebrow,RightEyebrow,LeftLip,RightLip,LipHeight,LipWidth\n";

            // Create the CSV file and fill in the first line with the header
            File.WriteAllText(fileName, header);
        }

        private void ExtractExpresionDirectory(string directory, string extractTo, string expression = "Default")
        {
            Console.WriteLine($"Extracting Features...");
            string[] inputImages = Directory.GetFiles(directory, "*");

            for (int i = 0; i < inputImages.Length; i++)
            {
                Console.WriteLine($"Extracting from image: {inputImages[i]}");
                ExtractImageFeatures(inputImages[i], extractTo, expression);
            }
        }

        public void ExtractData(string directory, string extractTo)
        {
            // Extract the suprise folder
            string surpriseDir = directory + "/Surprise";
            this.ExtractExpresionDirectory(surpriseDir, extractTo, "Surprise");

            // Extract the sadness folder
            string sadnessDir = directory + "/Sadness";
            this.ExtractExpresionDirectory(sadnessDir, extractTo, "Sadness");

            // Extract the fear folder
            string fearDir = directory + "/Fear";
            this.ExtractExpresionDirectory(fearDir, extractTo, "Fear");

            // Extract the anger folder
            string angerDir = directory + "/Anger";
            this.ExtractExpresionDirectory(angerDir, extractTo, "Anger");

            // Extract the disgust folder
            string disgustDir = directory + "/Disgust";
            this.ExtractExpresionDirectory(disgustDir, extractTo, "Disgust");

            // Extract the joy folder
            string joyDir = directory + "/Joy";
            this.ExtractExpresionDirectory(joyDir, extractTo, "Joy");
        }
        // https://medium.com/machinelearningadvantage/detect-facial-landmark-points-with-c-and-dlib-in-only-50-lines-of-code-71ab59f8873f
        public FacialData ExtractImageFeatures(string imageFile, string extractTo = "DontSave", string expression = "Default")
        {
            // File-paths
            string inputFilePath = imageFile;

            // Facial features
            float leftEyebrow = 0f, rightEyebrow = 0f, leftLip = 0f, rightLip = 0f, lipHeight = 0f, lipWidth = 0f;

            // Change what the label is depending on if an expression was passed in or if the type is in the name of the file
            string label = "";
            if (expression == "Default")
            {
                label = GetExpressionFromImageName(imageFile);
            }
            else
            {
                label = expression;
            }

            // set up Dlib facedetectors and shapedetectors
            using (var fd = Dlib.GetFrontalFaceDetector())
            using (var sp = ShapePredictor.Deserialize("shape_predictor_68_face_landmarks.dat"))
            {
                // load input image
                var img = Dlib.LoadImage<RgbPixel>(inputFilePath);

                // find all faces in the image
                var faces = fd.Operator(img);
                // for each face draw over the facial landmarks
                foreach (var face in faces)
                {
                    // find the landmark points for this face
                    var shape = sp.Detect(img, face);

                    // Eyebrows
                    leftEyebrow = CalculateFeature(shape, 39, 21, 18, 21);
                    rightEyebrow = CalculateFeature(shape, 42, 22, 22, 25);

                    // Lips
                    leftLip = CalculateFeature(shape, 33, 51, 48, 50);
                    rightLip = CalculateFeature(shape, 33, 51, 52, 54);

                    // Lip width and height
                    lipWidth = NormalisedDistBetween2Points(shape, 33, 51, 48, 54);
                    lipHeight = NormalisedDistBetween2Points(shape, 33, 51, 51, 57);

                    //Then write a new line with the calculated feature vector values separated by commas. Check that this works:
                    if (extractTo != "DontSave")
                    {
                        using (System.IO.StreamWriter file = new System.IO.StreamWriter(extractTo, true))
                        {
                            file.WriteLine(label + "," + leftEyebrow + "," + rightEyebrow + "," + leftLip + "," + rightLip + "," + lipHeight + "," + lipWidth);
                        }
                    }
                    else
                    {
                        // Draw the landmark points on the image
                        for (var i = 0; i < shape.Parts; i++)
                        {
                            var point = shape.GetPart((uint)i);
                            var rect = new Rectangle(point);

                            Dlib.DrawRectangle(img, rect, color: new RgbPixel(255, 255, 0), thickness: 4);
                        }

                        // Export the modified image
                        Dlib.SaveJpeg(img, "output.jpg");

                        // Output to the console the feature numbers
                        Console.WriteLine($"LeftEyeBrow: {leftEyebrow}, RightEyeBrow: {rightEyebrow}, LeftLip: {leftLip}, RightLip: {rightLip}, LipWidth: {lipWidth}, LipHeight: {lipHeight}");
                    }
                }

                return new FacialData()
                {
                    LeftEyebrow = leftEyebrow,
                    RightEyebrow = rightEyebrow,
                    LeftLip = leftLip,
                    RightLip = rightLip,
                    LipHeight = lipHeight,
                    LipWidth = lipWidth
                };
            }
        }

        private string GetExpressionFromImageName(string imageFile)
        {
            
            string extractedExpression = Path.GetFileName(imageFile);
            extractedExpression = extractedExpression.Substring(4, 2);

            // Returning expressions
            switch (extractedExpression)
            {
                case "an":
                    return "Angry";
                case "di":
                    return "Disgust";
                case "fe":
                    return "Fear";
                case "ha":
                    return "Joy";
                case "ne":
                    return "Neutral";
                case "sa":
                    return "Sadness";
                case "su":
                    return "Surprise";
                default:
                    return "ERROR";
            }
        }

        private float CalculateFeature(FullObjectDetection shape, int innerPoint, int normalisePoint, int leftMostPoint, int rightMostPoint)
        {
            float feature = 0f;

            // Looping through all the points on the feature 
            // adding on the normalised distance between that point and innerPoint
            for (var i = leftMostPoint; i <= rightMostPoint; i++)
            {
                feature += NormalisedDistBetween2Points(shape, innerPoint, normalisePoint, i, innerPoint);
            }

            return feature;
        }

        private float NormalisedDistBetween2Points(FullObjectDetection shape, int innerPoint, int normalisePoint, int firstPoint, int secondPoint)
        {
            // Get the positions of the innerpoint and the normalise point and use them to calculate the distance normaliser
            Vector2 innerPointPos = new Vector2(shape.GetPart((uint)innerPoint).X, shape.GetPart((uint)innerPoint).Y);
            Vector2 normalisePos = new Vector2(shape.GetPart((uint)normalisePoint).X, shape.GetPart((uint)normalisePoint).Y);
            double distNormaliser = Vector2.Distance(innerPointPos, normalisePos);

            // Get the position of both points and calculate the distance between them
            Vector2 firstPointPos = new Vector2(shape.GetPart((uint)firstPoint).X, shape.GetPart((uint)firstPoint).Y);
            Vector2 secondPointPos = new Vector2(shape.GetPart((uint)secondPoint).X, shape.GetPart((uint)secondPoint).Y);
            double distance = Vector2.Distance(firstPointPos, secondPointPos);

            // calculate the normalised distance between the points and return the value
            float normalisedDistance = (float)(distance / distNormaliser);
            return normalisedDistance;
        }
    }
}

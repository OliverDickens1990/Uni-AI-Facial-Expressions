using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Data;
//Getters and Setters
namespace Prediction
{
    // https://docs.microsoft.com/en-us/dotnet/machine-learning/tutorials/iris-clustering

    class FacialData
    {
        [LoadColumn(0)]
        public string Label { get; set; }

        [LoadColumn(1)]
        public float LeftEyebrow { get; set; }

        [LoadColumn(2)]
        public float RightEyebrow { get; set; }

        [LoadColumn(3)]
        public float LeftLip { get; set; }

        [LoadColumn(4)]
        public float RightLip { get; set; }

        [LoadColumn(5)]
        public float LipHeight { get; set; }

        [LoadColumn(6)]
        public float LipWidth { get; set; }

    }
}

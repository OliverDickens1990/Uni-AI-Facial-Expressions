using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Prediction
{
    //Different facial Expressions
    public enum DifferentExpressionType
    {
        ANGRY = 0,
        DISGUST = 1,
        FEAR = 2,
        JOY = 3,
        SAD = 4,
        SURPRISED = 5
    }

    class DiffernetExpressionPrediction
    {
        [ColumnName("PredictedLabel")]
        public string Label { get; set; }

        [ColumnName("Score")]
        public float[] Scores { get; set; }
    }
}

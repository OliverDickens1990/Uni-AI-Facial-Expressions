using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.IO;
using FeatureExtractionLib;


namespace Prediction
{
    class PredictionModel
    {
        private ITransformer model = null;
        private MLContext mlContext = new MLContext();

        // https://docs.microsoft.com/en-us/dotnet/machine-learning/tutorials/image-classification
        public void TrainingModel(string directory)
        {
            // Let the user know that the training process has begun
            Console.WriteLine($"Training Model...");

            // Load data from text file
            IDataView dataView = mlContext.Data.LoadFromTextFile<FacialData>(directory, hasHeader: true, separatorChar: ',');

            // Define the feature vector name and the label column name
            var featureVectorName = "Features";
            var labelColumnName = "Label";

            // The image transforms the images into the model's expected format
            var pipeline = mlContext.Transforms.Conversion
                .MapValueToKey(inputColumnName: "Label", outputColumnName: labelColumnName)
                .Append(mlContext.Transforms.Concatenate(featureVectorName, "LeftEyebrow", "RightEyebrow", "LeftLip", "RightLip", "LipHeight", "LipWidth"))
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName, featureVectorName))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // Train model
            model = pipeline.Fit(dataView);

            // Save model
            using (var fileStream = new FileStream("model.zip", FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mlContext.Model.Save(model, dataView.Schema, fileStream);
            }
        }
        // Prediction Function
        public void Predictor(string imageDirectory)
        {
            // Week 8 Practicals
            // Loading in the model
            LoadModel();

            // Setup the predictor using the model from the training
            var predictor = mlContext.Model.CreatePredictionEngine<FacialData, DiffernetExpressionPrediction>(model);

            // Extract the features from the image and make a prediction
            FeatureExtraction featureExtraction = new FeatureExtraction();
            FacialData faceData = featureExtraction.ExtractImageFeatures(imageDirectory);
            var FacePrediction = predictor.Predict(faceData);

            // Print Values
            Console.WriteLine($"*** Prediction: {FacePrediction.Label } ***");
            Console.WriteLine($"*** Scores: {string.Join(" ", FacePrediction.Scores)} ***");
        }

        // Evaluate Model
        public void EvaluateModel(string testDataPath)
        {
            // Loading in the model
            LoadModel();
            // Week 10 Practicals 
            // example code was used from: https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.automl.multiclassclassificationexperiment?view=ml-dotnet-preview
            // Evaluating the model
            IDataView testDataView = mlContext.Data.LoadFromTextFile<FacialData>(testDataPath, hasHeader: true, separatorChar: ',');
            var testMetrics = mlContext.MulticlassClassification.Evaluate(model.Transform(testDataView));

            // Writing to the console
            Console.WriteLine($"* Metrics for Multi-class Classification model - Test Data");
            Console.WriteLine($"* MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
            Console.WriteLine($"* MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
            Console.WriteLine($"* LogLoss:          {testMetrics.LogLoss:#.###}");
            Console.WriteLine($"* LogLossReduction: {testMetrics.LogLossReduction:#.###}");

            // Log Loss Per Class:
            Console.WriteLine($"* Log Loss Per Class:");
            System.Collections.Generic.IReadOnlyList<double> logList = testMetrics.PerClassLogLoss;
            for (int i = 0; i < logList.Count; i++)
            {
                Console.WriteLine($"*    - { (DifferentExpressionType)i } : {logList[i]:#.###}");
            }

            // Precision Per Class:
            Console.WriteLine($"* ConfusionMatrixPrecision:");
            System.Collections.Generic.IReadOnlyList<double> precisionList = testMetrics.ConfusionMatrix.PerClassPrecision;
            for (int i = 0; i < precisionList.Count; i++)
            {
                Console.WriteLine($"*    - {(DifferentExpressionType)i} : {precisionList[i]:#.###}");
            }
            // Recall per class
            Console.WriteLine($"* ConfusionMatrixRecall:");
            System.Collections.Generic.IReadOnlyList<double> recallList = testMetrics.ConfusionMatrix.PerClassRecall;
            for (int i = 0; i < recallList.Count; i++)
            {
                Console.WriteLine($"*    - {(DifferentExpressionType)i} : {recallList[i]:#.###}");
            }
        }
 
        private void LoadModel()
        {
            DataViewSchema dataViewSchema = null;
            using (var fileStream = new FileStream("model.zip", FileMode.Open, FileAccess.Read))
            {
                model = mlContext.Model.Load(fileStream, out dataViewSchema);
            }
        }
    }
}

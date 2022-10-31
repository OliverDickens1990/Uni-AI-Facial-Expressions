using System;
using DlibDotNet;
using Microsoft.ML;
using Microsoft.ML.Data;
using DlibDotNet.Extensions;
using Dlib = DlibDotNet.Dlib;
using Prediction;
using System.IO;
using FeatureExtractionLib;

// The main program class
class ProgramMenu
{
    private static void PressEnterToContinue()
    {
        Console.WriteLine("* Press enter to continue:");
        Console.ReadLine();
        Console.Clear();
    }

    // The main program entry point
    static void Main(string[] args)
    {
        // Define default values
        PredictionModel predictionModel = new PredictionModel();
        FeatureExtraction featureExtraction = new FeatureExtraction();
        bool quit = false;

        // Loop until the option 'Exit program' has been selected
        do
        {
            // User Menu 
            Console.WriteLine("* Please enter a number for the option you want: ");
            Console.WriteLine("* 1. Extract training data features");
            Console.WriteLine("* 2. Extract test data features");
            Console.WriteLine("* 3. Train model");
            Console.WriteLine("* 4. Evaluate model");
            Console.WriteLine("* 5. Predict single image");
            Console.WriteLine("* 0. Exit program");

            // Read the users input and clear once they have picked an option
            int inputOption = Convert.ToInt32(Console.ReadLine());
            Console.Clear();

            // Depending on the input option execute different code
            switch (inputOption)
            {
                case 0:
                    // This will exit the loop and end the program
                    quit = true;
                    break;
                case 1:
                    // Extract data from the training directory and store it in a csv file
                    featureExtraction.CreateNewCSVFileToExtractTo(@"feature_vectorsTraining.csv");
                    featureExtraction.ExtractData("Images/TrainingImages", @"feature_vectorsTraining.csv");
                    PressEnterToContinue();
                    break;
                case 2:
                    // Extract data from the testing directory and store it in a csv file
                    featureExtraction.CreateNewCSVFileToExtractTo(@"feature_vectorsTesting.csv");
                    featureExtraction.ExtractData("Images/TestingImages", @"feature_vectorsTesting.csv");
                    PressEnterToContinue();
                    break;
                case 3:
                    // Training the prediction model
                    predictionModel.TrainingModel(@"feature_vectorsTraining.csv");
                    PressEnterToContinue();
                    break;
                case 4:
                    // Evaluate the model using the test data
                    predictionModel.EvaluateModel(@"feature_vectorsTesting.csv");
                    PressEnterToContinue();
                    break;
                case 5:
                    // Prompt the user to enter the directory of the image to predict, then print the results
                    Console.WriteLine("Please enter the directory of the image you want to predict:");
                    string directory = Console.ReadLine();
                    predictionModel.Predictor(directory);
                    PressEnterToContinue();
                    break;
                default:
                    Console.WriteLine("* ERROR - Input was invalid! Press enter to try again:");
                    Console.ReadLine();
                    Console.Clear();
                    break;
            }

        } while (!quit);
    }
}
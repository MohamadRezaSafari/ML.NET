using Microsoft.ML;
using Microsoft.ML.Data;
using Sentiment_BinaryClassification;
using static Microsoft.ML.DataOperationsCatalog;

string _dataPath = Path.Combine($"{Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName}/Data/yelp_labelled.txt");


MLContext mLContext = new MLContext();
TrainTestData splitDataView = LoadData(mLContext);

ITransformer model = BuildAndTrainModel(mLContext, splitDataView.TrainSet);
Evaluate(mLContext, model, splitDataView.TrainSet);
UseModelWithSingleItem(mLContext, model);
UseModelWithBatchItems(mLContext, model);


void UseModelWithBatchItems(MLContext mLContext, ITransformer model)
{
    IEnumerable<SentimentData> setiments = new[]
    {
         new SentimentData(){ SentimentText= "This was a horrible meal" },
         new SentimentData(){ SentimentText = "I love this spaghetti." }
    };

    IDataView batchComments = mLContext.Data.LoadFromEnumerable(setiments);
    IDataView predictions = model.Transform(batchComments);
    IEnumerable<SentimentPrediction> predictedResults = mLContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);

    Console.WriteLine();
    Console.WriteLine("=============== Prediction Test of loaded model with multiple samples ===============");
    Console.WriteLine();

    foreach (SentimentPrediction prediction in predictedResults)
    {
        Console.WriteLine($"Sentiment: {prediction.SentimentText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability} ");
    }
    Console.WriteLine("=============== End of predictions ===============");
}

void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
{
    PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

    SentimentData sampleStatement = new SentimentData { SentimentText = "This was a very bad steak" };

    var resultPrediction = predictionFunction.Predict(sampleStatement);

    Console.WriteLine();
    Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

    Console.WriteLine();
    Console.WriteLine($"Sentiment: {resultPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ");

    Console.WriteLine("=============== End of Predictions ===============");
    Console.WriteLine();
}

void Evaluate(MLContext mLContext, ITransformer model, IDataView splitTestSet)
{
    Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
    IDataView predictions = model.Transform(splitTestSet);
    CalibratedBinaryClassificationMetrics metrics = mLContext.BinaryClassification.Evaluate(predictions, "Label");

    Console.WriteLine();
    Console.WriteLine("Model quality metrics evaluation");
    Console.WriteLine("--------------------------------");
    Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
    Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
    Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
    Console.WriteLine("=============== End of model evaluation ===============");
}

ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
{
    var estimator = mLContext.Transforms.Text
        .FeaturizeText(
            outputColumnName: "Features",
            inputColumnName: nameof(SentimentData.SentimentText))
        .Append(mLContext.BinaryClassification.Trainers
            .SdcaLogisticRegression(
                labelColumnName: "Label",
                featureColumnName: "Features"));

    Console.WriteLine("=============== Create and Train the Model ===============");
    var model = estimator.Fit(splitTrainSet);
    Console.WriteLine("=============== End of training ===============");
    Console.WriteLine();

    return model;
}

TrainTestData LoadData(MLContext mlContext)
{
    IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);
    TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

    return splitDataView;
}
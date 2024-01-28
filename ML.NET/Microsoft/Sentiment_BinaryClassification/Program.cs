
using Microsoft.ML;
using Sentiment_BinaryClassification;
using static Microsoft.ML.DataOperationsCatalog;

string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt");


MLContext mLContext = new MLContext();
TrainTestData splitDataView = LoadData(mLContext);

ITransformer model = BuildAndTrainModel(mLContext, splitDataView.TrainSet);


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
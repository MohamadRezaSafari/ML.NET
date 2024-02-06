

using Microsoft.ML;
using Regression_PricePredictor;

string _appPath = Path.Combine($"{Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName}");
string _trainDataPath = Path.Combine(_appPath, "Data", "taxi-fare-train.csv");
string _testDataPath = Path.Combine(_appPath, "Data", "taxi-fare-test.csv");
string _modelPath = Path.Combine(_appPath, "Data", "Model.zip");


MLContext mLContext = new MLContext(seed: 0);
var model = Train(mLContext, _trainDataPath);
Evaluate(mLContext, model);
TestSinglePrediction(mLContext, model);


void TestSinglePrediction(MLContext mLContext, ITransformer transformer)
{
    var predictionFunction = mLContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(model);

    var taxiTripSample = new TaxiTrip()
    {
        VendorId = "VTS",
        RateCode = "1",
        PassengerCount = 1,
        TripTime = 1140,
        TripDistance = 3.75f,
        PaymentType = "CRD",
        FareAmount = 0
    };

    var prediction = predictionFunction.Predict(taxiTripSample);

    Console.WriteLine($"**********************************************************************");
    Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 15.5");
    Console.WriteLine($"**********************************************************************");
}

void Evaluate(MLContext mlContext, ITransformer model)
{
    IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(_testDataPath, hasHeader: true, separatorChar: ',');
    var predictions = model.Transform(dataView);

    var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

    Console.WriteLine();
    Console.WriteLine($"*************************************************");
    Console.WriteLine($"*       Model quality metrics evaluation         ");
    Console.WriteLine($"*------------------------------------------------");
    Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");
    Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");
}

ITransformer Train(MLContext mlContext, string dataPath)
{
    IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(dataPath, hasHeader: true, separatorChar: ',');

    var pipleline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount")
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
        .Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripDistance", "PaymentTypeEncoded"))
        .Append(mlContext.Regression.Trainers.FastTree());

    var model = pipleline.Fit(dataView);
    return model;
}
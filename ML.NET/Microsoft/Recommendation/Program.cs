using Microsoft.ML;
using Microsoft.ML.Trainers;
using Recommendation;


string _appPath = Path.Combine($"{Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName}");
var trainingDataPath = Path.Combine(_appPath, "Data", "recommendation-ratings-train.csv");
var testDataPath = Path.Combine(_appPath, "Data", "recommendation-ratings-test.csv");
var modelPath = Path.Combine(_appPath, "Data", "MovieRecommenderModel.zip");

MLContext mLContext = new MLContext();
(IDataView trainingDataView, IDataView testDataView) = LoadData(mLContext);
ITransformer model = BuildAndTrainModel(mLContext, trainingDataView);
EvaluateModel(mLContext, testDataView, model);
UseModelForSinglePrediction(mLContext, model);
SaveModel(mLContext, trainingDataView.Schema, model);


void SaveModel(MLContext mLContext, DataViewSchema trainingDataViewSchema, ITransformer model)
{
    Console.WriteLine("=============== Saving the model to a file ===============");

    mLContext.Model.Save(model, trainingDataViewSchema, modelPath);
}

void UseModelForSinglePrediction(MLContext mLContext, ITransformer model)
{
    Console.WriteLine("=============== Making a prediction ===============");

    var predictionEngine = mLContext.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(model);

    var testInput = new MovieRating { UserId = 6, MovieId = 25 };
    var movieratingPrediction = predictionEngine.Predict(testInput);

    if (Math.Round(movieratingPrediction.Score, 1) > 3.5)
    {
        Console.WriteLine("Movie " + testInput.MovieId + " is recommended for user " + testInput.UserId);
    }
    else
    {
        Console.WriteLine("Movie " + testInput.MovieId + " is not recommended for user " + testInput.UserId);
    }
}

void EvaluateModel(MLContext mLContext, IDataView testDataView, ITransformer model)
{
    Console.WriteLine("=============== Evaluating the model ===============");

    var prediction = model.Transform(testDataView);
    var metrics = mLContext.Regression.Evaluate(prediction, labelColumnName: "Label", scoreColumnName: "Score");

    Console.WriteLine("Root Mean Squared Error : " + metrics.RootMeanSquaredError.ToString());
    Console.WriteLine("RSquared: " + metrics.RSquared.ToString());
}

ITransformer BuildAndTrainModel(MLContext mlContext, IDataView trainingDataview)
{
    IEstimator<ITransformer> estimator = mlContext.Transforms.Conversion
        .MapValueToKey(outputColumnName: "userIdEncoded", inputColumnName: "UserId")
        .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "movieIdEncoded", inputColumnName: "MovieId"));

    var options = new MatrixFactorizationTrainer.Options
    {
        MatrixColumnIndexColumnName = "userIdEncoded",
        MatrixRowIndexColumnName = "movieIdEncoded",
        LabelColumnName = "Label",
        NumberOfIterations = 20,
        ApproximationRank = 100
    };

    var trainerEstimator = estimator.Append(mlContext.Recommendation().Trainers.MatrixFactorization(options));

    Console.WriteLine("=============== Training the model ===============");

    ITransformer model = trainerEstimator.Fit(trainingDataview);

    return model;
}

(IDataView training, IDataView test) LoadData(MLContext mlContext)
{
    IDataView trainingDataView = mlContext.Data.LoadFromTextFile<MovieRating>(trainingDataPath, hasHeader: true, separatorChar: ',');
    IDataView testDataView = mlContext.Data.LoadFromTextFile<MovieRating>(testDataPath, hasHeader: true, separatorChar: ',');

    return (trainingDataView, testDataView);
}
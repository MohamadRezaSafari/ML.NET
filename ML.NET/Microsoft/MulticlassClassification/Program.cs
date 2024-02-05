using Microsoft.ML;
using MulticlassClassification;


string _appPath = Path.Combine($"{Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName}");
string _trainDataPath = Path.Combine(_appPath, "Data", "issues_train.tsv.txt");
string _testDataPath = Path.Combine(_appPath, "Data", "issues_test.tsv.txt");
string _modelPath = Path.Combine(_appPath, "Models", "model.zip");


MLContext _mlContext;
PredictionEngine<GitHubIssue, IssuePrediction> _predEngine;
ITransformer _trainedModel;
IDataView _trainingDataView;

_mlContext = new MLContext(seed: 0);
Console.WriteLine($"=============== Loading Dataset  ===============");
_trainingDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(_trainDataPath, hasHeader: true);
Console.WriteLine($"=============== Finished Loading Dataset  ===============");
var pipeline = ProcessData();
var trainingPipeline = BuildAndTrainModel(_trainingDataView, pipeline);
Evaluate(_trainingDataView.Schema);
PredictIssue();


void PredictIssue()
{
    ITransformer loadedModel = _mlContext.Model.Load(_modelPath, out var modelInputSchema);
    GitHubIssue singleIssue = new GitHubIssue() { Title = "Entity Framework crashes", Description = "When connecting to the database, EF is crashing" };
    _predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(loadedModel);
    var prediction = _predEngine.Predict(singleIssue);

    Console.WriteLine($"=============== Single Prediction - Result: {prediction.Area} ===============");
}

void SaveModelAsFile(MLContext mLContext, DataViewSchema trainingDataViewSchema, ITransformer model)
{
    mLContext.Model.Save(model, trainingDataViewSchema, _modelPath);
    Console.WriteLine("The model is saved to {0}", _modelPath);
}

void Evaluate(DataViewSchema trainingDataViewSchema)
{
    Console.WriteLine($"=============== Evaluating to get model's accuracy metrics - Starting time: {DateTime.Now.ToString()} ===============");

    var testdataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(_testDataPath, hasHeader: true);
    var testMetrics = _mlContext.MulticlassClassification.Evaluate(_trainedModel.Transform(testdataView));

    Console.WriteLine($"*************************************************************************************************************");
    Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
    Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
    Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
    Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
    Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
    Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
    Console.WriteLine($"*************************************************************************************************************");

    SaveModelAsFile(_mlContext, trainingDataViewSchema, _trainedModel);
}

IEstimator<ITransformer> BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
{
    var trainingPipeline = pipeline
        .Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
        .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

    _trainedModel = trainingPipeline.Fit(trainingDataView);
    _predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(_trainedModel);

    GitHubIssue issue = new GitHubIssue()
    {
        Title = "WebSockets cmmunication is slow in my machine",
        Description = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.."
    };

    var prediction = _predEngine.Predict(issue);
    Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.Area} ===============");

    return trainingPipeline;
}

IEstimator<ITransformer> ProcessData()
{
    var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Area", outputColumnName: "Label")
        .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized"))
        .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Description", outputColumnName: "DescriptionFeaturized"))
        .Append(_mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
        .AppendCacheCheckpoint(_mlContext);

    return pipeline;
}
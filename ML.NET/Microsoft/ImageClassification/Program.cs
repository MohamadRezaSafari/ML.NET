using ImageClassification;
using Microsoft.ML;
using Microsoft.ML.Data;

string _appPath = Path.Combine($"{Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName}");
string _assetsPath = Path.Combine(_appPath, "assets");
string _imagesFolder = Path.Combine(_assetsPath, "images");
string _trainTagsTsv = Path.Combine(_imagesFolder, "tags.tsv");
string _testTagsTsv = Path.Combine(_imagesFolder, "test-tags.tsv");
string _predictSingleImage = Path.Combine(_imagesFolder, "toaster3.jpg");
string _inceptionTensorFlowModel = Path.Combine(_assetsPath, "inception", "tensorflow_inception_graph.pb");


MLContext mlContext = new MLContext();
ITransformer model = GenerateModel(mlContext);
ClassifySingleImage(mlContext, model);


ITransformer GenerateModel(MLContext mLContext)
{
    IEstimator<ITransformer> pipeline = mlContext.Transforms
        .LoadImages(outputColumnName: "input", imageFolder: _imagesFolder, inputColumnName: nameof(ImageData.ImagePath))
        .Append(mlContext.Transforms.ResizeImages(outputColumnName: "input", imageWidth: InceptionSettings.ImageWidth,
                imageHeight: InceptionSettings.ImageHeight, inputColumnName: "input"))
        .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input",
                interleavePixelColors: InceptionSettings.ChannelsLast, offsetImage: InceptionSettings.Mean))
        .Append(mlContext.Model.LoadTensorFlowModel(_inceptionTensorFlowModel)
            .ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2_pre_activation" },
                inputColumnNames: new[] { "input" }, addBatchDimensionInput: true))
        .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelKey", inputColumnName: "Label"))
        .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "LabelKey",
                featureColumnName: "softmax2_pre_activation"))
        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabelValue", "PredictedLabel"))
        .AppendCacheCheckpoint(mlContext);

    IDataView trainingData = mlContext.Data.LoadFromTextFile<ImageData>(path: _trainTagsTsv, hasHeader: false);
    ITransformer model = pipeline.Fit(trainingData);

    IDataView testData = mlContext.Data.LoadFromTextFile<ImageData>(path: _testTagsTsv, hasHeader: false);
    IDataView predictions = model.Transform(testData);

    IEnumerable<ImagePrediction> imagePredictionData = mlContext.Data.CreateEnumerable<ImagePrediction>(predictions, true);
    DisplpayResults(imagePredictionData);

    MulticlassClassificationMetrics metrics = mlContext.MulticlassClassification.Evaluate(predictions,
        labelColumnName: "LabelKey", predictedLabelColumnName: "PredictedLabel");

    Console.WriteLine($"LogLoss is: {metrics.LogLoss}");
    Console.WriteLine($"PerClassLogLoss is: {String.Join(" , ", metrics.PerClassLogLoss.Select(c => c.ToString()))}");

    return model;
}

void ClassifySingleImage(MLContext mLContext, ITransformer model)
{
    var imageData = new ImageData()
    {
        ImagePath = _predictSingleImage
    };

    var predictor = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
    var prediction = predictor.Predict(imageData);

    Console.WriteLine($"Image: {Path.GetFileName(imageData.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score?.Max()} ");
}

void DisplpayResults(IEnumerable<ImagePrediction> imagePredictionData)
{
    foreach (ImagePrediction prediction in imagePredictionData)
    {
        Console.WriteLine($"Image: {Path.GetFileName(prediction.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score?.Max()} ");
    }
}
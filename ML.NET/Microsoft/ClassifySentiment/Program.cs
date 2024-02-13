using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using ClassifySentiment;

string _appPath = Path.Combine($"{Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName}");
string _modelPath = Path.Combine(_appPath, "sentiment_model");


MLContext mLContext = new MLContext();
TensorFlowModel tensorFlowModel = mLContext.Model.LoadTensorFlowModel(_modelPath);
DataViewSchema schema = tensorFlowModel.GetModelSchema();
Console.WriteLine(" =============== TensorFlow Model Schema =============== ");

var featuresType = (VectorDataViewType)schema["Features"].Type;
Console.WriteLine($"Name: Features, Type: {featuresType.ItemType.RawType}, " +
    $"Size: ({featuresType.Dimensions[0]})");

var predictionType = (VectorDataViewType)schema["Prediction/Softmax"].Type;
Console.WriteLine($"Name: Prediction/Softmax, Type: {predictionType.ItemType.RawType}, " +
    $"Size: ({predictionType.Dimensions[0]})");


var lookupMap = mLContext.Data.LoadFromTextFile(path: Path.Combine(_modelPath, "imdb_word_index.csv"),
    columns: new[]
    {
        new TextLoader.Column("Words", DataKind.String, 0),
        new TextLoader.Column("Ids", DataKind.Int32, 1)
    },
    separatorChar: ','
);

Action<VariableLength, FixedLength> ResizeFeaturesAction = (s, f) =>
{
    var features = s.VariableLengthFeatures;
    Array.Resize(ref features, Config.FeatureLength);
    f.Features = features;
};


IEstimator<ITransformer> pipeline = mLContext.Transforms.Text
    .TokenizeIntoWords("TokenizedWords", "ReviewText")
    .Append(mLContext.Transforms.Conversion.MapValue("VariableLengthFeatures", lookupMap,
        lookupMap.Schema["Words"], lookupMap.Schema["Ids"], "TokenizedWords"))
    .Append(mLContext.Transforms.CustomMapping(ResizeFeaturesAction, "Resize"))
    .Append(tensorFlowModel.ScoreTensorFlowModel("Prediction/Softmax", "Features"))
    .Append(mLContext.Transforms.CopyColumns("Prediction", "Prediction/Softmax"));

IDataView dataView = mLContext.Data.LoadFromEnumerable(new List<MovieReview>());
ITransformer model = pipeline.Fit(dataView);
PredictSentiment(mLContext, model);


void PredictSentiment(MLContext mLContext, ITransformer model)
{
    var engine = mLContext.Model.CreatePredictionEngine<MovieReview, MovieReviewSentimentPrediction>(model);

    var review = new MovieReview()
    {
        ReviewText = "this film is really good"
    };

    var sentimentPrediction = engine.Predict(review);

    Console.WriteLine($"Number of classes: {sentimentPrediction.Prediction?.Length}");
    Console.WriteLine($"Is sentiment/review positive? {(sentimentPrediction.Prediction?[1] > 0.5 ? "Yes." : "No.")}");
}

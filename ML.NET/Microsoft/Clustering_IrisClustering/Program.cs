using Clustering_IrisClustering;
using Microsoft.ML;

string _appPath = Path.Combine($"{Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName}");
string _dataPath = Path.Combine(_appPath, "Data", "iris.data");
string _modelPath = Path.Combine(_appPath, "Data", "IrisClusteringModel.zip");


MLContext mLContext = new MLContext(seed: 0);
IDataView dataView = mLContext.Data.LoadFromTextFile<IrisData>(_dataPath, hasHeader: false, separatorChar: ',');

string featuresColumnName = "Features";
var pipeline = mLContext.Transforms
    .Concatenate(featuresColumnName, "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
    .Append(mLContext.Clustering.Trainers.KMeans(featuresColumnName, numberOfClusters: 3));

var model = pipeline.Fit(dataView);

using var fileStream = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write);
mLContext.Model.Save(model, dataView.Schema, fileStream);

var predictor = mLContext.Model.CreatePredictionEngine<IrisData, ClusterPrediction>(model);
var prediction = predictor.Predict(TestIrisData.Setosa);
Console.WriteLine($"Cluster: {prediction.PredictedClusterId}");
Console.WriteLine($"Distances: {string.Join(" ", prediction.Distances ?? Array.Empty<float>())}");

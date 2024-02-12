
using AnomalyDetection;
using Microsoft.ML;

string _appPath = Path.Combine($"{Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName}");
string _dataPath = Path.Combine(_appPath, "Data", "product-sales.csv");
const int _docsize = 36;


MLContext mLContext = new MLContext();

IDataView dataView = mLContext.Data.LoadFromTextFile<ProductSalesData>(path: _dataPath, hasHeader: true, separatorChar: ',');
DetectSpike(mLContext, _docsize, dataView);
DetectChangePoint(mLContext, _docsize, dataView);


void DetectChangePoint(MLContext mlContext, int docSize, IDataView productSales)
{
    var iidChangePointEstimator = mlContext.Transforms
        .DetectIidChangePoint(outputColumnName: nameof(ProductSalesPrediction.Prediction),
            inputColumnName: nameof(ProductSalesData.NumSales), confidence: 95d, changeHistoryLength: docSize / 4);

    var iidChangePointTransform = iidChangePointEstimator.Fit(CreateEmptyDataView(mlContext));
    IDataView transformedData = iidChangePointTransform.Transform(productSales);
    var predictions = mlContext.Data.CreateEnumerable<ProductSalesPrediction>(transformedData, reuseRowObject: false);

    Console.WriteLine("Alert\tScore\tP-Value\tMartingale value");
    foreach (var p in predictions)
    {
        if (p.Prediction is not null)
        {
            var results = $"{p.Prediction[0]}\t{p.Prediction[1]:f2}\t{p.Prediction[2]:F2}\t{p.Prediction[3]:F2}";

            if (p.Prediction[0] == 1)
            {
                results += " <-- alert is on, predicted changepoint";
            }
            Console.WriteLine(results);
        }
    }
    Console.WriteLine("");
}

void DetectSpike(MLContext mlContext, int docSize, IDataView productSales)
{
    var iidSpikeEstimator = mlContext.Transforms
        .DetectIidSpike(outputColumnName: nameof(ProductSalesPrediction.Prediction), inputColumnName: nameof(ProductSalesData.NumSales),
            confidence: 95d, pvalueHistoryLength: docSize / 4);

    ITransformer iidSpikeTransform = iidSpikeEstimator.Fit(CreateEmptyDataView(mlContext));
    IDataView transformedData = iidSpikeTransform.Transform(productSales);
    var predictions = mlContext.Data.CreateEnumerable<ProductSalesPrediction>(transformedData, reuseRowObject: false);

    Console.WriteLine("Alert\tScore\tP-Value");
    foreach (var p in predictions)
    {
        if (p.Prediction is not null)
        {
            var results = $"{p.Prediction[0]}\t{p.Prediction[1]:f2}\t{p.Prediction[2]:F2}";

            if (p.Prediction[0] == 1)
            {
                results += " <-- Spike detected";
            }

            Console.WriteLine(results);
        }
    }
    Console.WriteLine("");

}

IDataView CreateEmptyDataView(MLContext mlContext)
{
    IEnumerable<ProductSalesData> enumerableData = new List<ProductSalesData>();
    return mlContext.Data.LoadFromEnumerable(enumerableData);
}

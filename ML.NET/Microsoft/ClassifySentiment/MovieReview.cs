using Microsoft.ML.Data;

namespace ClassifySentiment;

public class MovieReview
{
    public string? ReviewText { get; set; }
}

public class VariableLength
{
    [VectorType]
    public int[]? VariableLengthFeatures { get; set; }
}

public class FixedLength
{
    [VectorType(Config.FeatureLength)]
    public int[]? Features { get; set; }
}

public class MovieReviewSentimentPrediction
{
    [VectorType(2)]
    public float[]? Prediction { get; set; }
}

static class Config
{
    public const int FeatureLength = 600;
}
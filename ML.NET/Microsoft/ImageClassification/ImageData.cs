using Microsoft.ML.Data;

namespace ImageClassification;

public class ImageData
{
    [LoadColumn(0)]
    public string? ImagePath { get; set; }

    [LoadColumn(1)]
    public string? Label { get; set; }
}

public class ImagePrediction : ImageData
{
    public float[]? Score;
    public string? PredictedLabelValue;
}

struct InceptionSettings
{
    public const int ImageHeight = 224;
    public const int ImageWidth = 224;
    public const float Mean = 117;
    public const float Scale = 1;
    public const bool ChannelsLast = true;
}
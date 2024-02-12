using Microsoft.ML.Data;
using System.Drawing;

namespace DetectObjectsInImages;

public class ImageNetData
{
    [LoadColumn(0)]
    public string ImagePath;

    [LoadColumn(1)]
    public string Label;

    public static IEnumerable<ImageNetData> ReadFromFile(string imageFolder)
    {
        return Directory
            .GetFiles(imageFolder)
            .Where(filePath => Path.GetExtension(filePath) is not ".md")
            .Select(filePath => new ImageNetData { ImagePath = filePath, Label = Path.GetFileName(filePath) });
    }
}

public class ImageNetPrediction
{
    [ColumnName("grid")]
    public float[] PredictionLabels;
}

public class DimensionBase
{
    public float X { get; set; }
    public float Y { get; set; }
    public float Height { get; set; }
    public float Width { get; set; }
}

public class BoundingBoxDimensions : DimensionBase { }

public class YoloBoundingBox
{
    public BoundingBoxDimensions Dimensions { get; set; }
    public string Label { get; set; }
    public float Confidence { get; set; }
    public RectangleF Rect
    {
        get { return new RectangleF(Dimensions.X, Dimensions.Y, Dimensions.Width, Dimensions.Height); }
    }

    public Color BoxColor { get; set; }
}

public class CellDimensions : DimensionBase { }

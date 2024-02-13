using StableDiffusion.ML.OnnxRuntime;

var watch = System.Diagnostics.Stopwatch.StartNew();

//Default args
var prompt = "a fireplace in an old cabin in the woods";
Console.WriteLine(prompt);

var config = new StableDiffusionConfig
{
    NumInferenceSteps = 15,
    GuidanceScale = 7.5,
    ExecutionProviderTarget = StableDiffusionConfig.ExecutionProvider.Cuda,
    DeviceId = 0,
    TextEncoderOnnxPath = @"C:\code\StableDiffusion\StableDiffusion\models\text_encoder\model.onnx",
    UnetOnnxPath = @"C:\code\StableDiffusion\StableDiffusion\models\unet\model.onnx",
    VaeDecoderOnnxPath = @"C:\code\StableDiffusion\StableDiffusion\models\vae_decoder\model.onnx",
    SafetyModelPath = @"C:\code\StableDiffusion\StableDiffusion\models\safety_checker\model.onnx",
};

// Inference Stable Diff
var image = UNet.Inference(prompt, config);

// If image failed or was unsafe it will return null.
if (image == null)
{
    Console.WriteLine("Unable to create image, please try again.");
}
// Stop the timer
watch.Stop();
var elapsedMs = watch.ElapsedMilliseconds;
Console.WriteLine("Time taken: " + elapsedMs + "ms");
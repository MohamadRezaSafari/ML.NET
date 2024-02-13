//using Microsoft.ML.OnnxRuntime;

//var prompt = "a fireplace in an old cabin in the woods";
////...
//var textTokenized = TextProcessing.TokenizeText(prompt);
//var textPromptEmbeddings = TextProcessing.TextEncoder(textTokenized).ToArray();

////...
//var scheduler = new LMSDiscreteScheduler();
////...
//var timesteps = scheduler.SetTimesteps(numInferenceSteps);
////...
//var seed = new Random().Next();
//var latents = GenerateLatentSample(batchSize, height, width, seed, scheduler.InitNoiseSigma);
////...
//var unetSession = new InferenceSession(modelPath, options);
//var input = new List<NamedOnnxValue>();
////...
//for (int t = 0; t < timesteps.Length; t++)
//{
//    //...
//    var latentModelInput = TensorHelper.Duplicate(latents.ToArray(), new[] { 2, 4, height / 8, width / 8 });
//    //...
//    latentModelInput = scheduler.ScaleInput(latentModelInput, timesteps[t]);
//    //...
//    input = CreateUnetModelInput(textEmbeddings, latentModelInput, timesteps[t]);
//    var output = unetSession.Run(input);
//    //...
//    noisePred = performGuidance(noisePred, noisePredText, guidanceScale);
//    //...
//    latents = scheduler.Step(noisePred, timesteps[t], latents);
//}


//var decoderInput = new List<NamedOnnxValue>
//{ NamedOnnxValue.CreateFromTensor("latent_sample", latents) };
//var imageResultTensor = VaeDecoder.Decoder(decoderInput);
//var image = VaeDecoder.ConvertToImage(imageResultTensor);

modelName = 'speechBaseline4_CTCsmoothing3'

args = {}
args['outputDir'] = "/Users/KatherynZhou/Desktop/BCI class/neural_seq_decoder_project/models/" + modelName
args['datasetPath'] = "/Users/KatherynZhou/Desktop/BCI class/neural_seq_decoder_project/ptDecoder_ctc"
# args['outputDir'] = '/oak/stanford/groups/henderj/stfan/logs/speech_logs/' + modelName
# args['datasetPath'] = '/oak/stanford/groups/henderj/fwillett/speech/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64 #128
args['lrStart'] = 0.03
args['lrEnd'] = 0.002
args['nUnits'] = 1024
args['nBatch'] = 15000 #3000
args['nLayers'] = 5 # number of GRU layers.
args['seed'] = 0
args['nClasses'] = 40 # number of output classes, not including the CTC blank token
args['nInputFeatures'] = 256 # number of neural features (spike band power and threshold crossings)
args['dropout'] = 0.4 # dropout percentage used for GRU layers
args['whiteNoiseSD'] = 0.8 # amount of white noise augmentation to add to neural data during training
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0 # onvolves the neural data with a Gaussian kernel with the specified width
args['strideLen'] = 4 # umber of neural time bins the input is shifted forward at each timestep. This controls how often the GRU makes an output
args['kernelLen'] = 32 # number of neural time bins fed to the GRU at each timestep
args['bidirectional'] = False # True
args['l2_decay'] = 1e-5 # amount of L2 regularization that is applied
args['grad_clip'] = 5.0
args['warmupSteps'] = 500
args['nMasks'] = 2 # number of time masks to implement per batch, make 0 to skip time masking
args['maxMaskLength'] = 20 # max number of timesteps to mask per single mask
args['layerNorm'] = True # whether or not to have layernorm layer between GRU and output
args['CTCsmoothing'] = True
print(f"grad_clip = {args['grad_clip']}")
print(f"warmupSteps = {args['warmupSteps']}")
print(f"nMasks = {args['nMasks']}")
print(f'outputDir {args["outputDir"]}')
print(f"layerNorm = {args['layerNorm']}")
print(f"CTCsmoothing = {args['CTCsmoothing']}")

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)
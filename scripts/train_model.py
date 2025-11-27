import numpy as np

modelName = 'katherine_best_beamdecoding'

args = {}
args['outputDir'] = '/home/onuralp/Desktop/c243/neural_seq_decoder_project/logs/speech_logs/' + modelName
args['datasetPath'] = '/home/onuralp/Desktop/c243/neural_seq_decoder/ptDecoder_ctc'
#args['outputDir'] = '/Users/KatherynZhou/Desktop/BCI class/neural_seq_decoder_project/models/' + modelName
#args['datasetPath'] = '/Users/KatherynZhou/Desktop/BCI class/neural_seq_decoder_project/ptDecoder_ctc'
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
args['nMasks'] = 8 # number of time masks to implement per batch, make 0 to skip time masking
args['maxMaskLength'] = 20 # max number of timesteps to mask per single mask
args['layerNorm'] = True # whether or not to have layernorm layer between GRU and output
args['causalGaussian'] = True # whether to use causal Gaussian smoothing on the neural data
args['CTCsmoothing'] = 0.1
args['beamWidth'] = 5 # beam width for beam search decoding

import sys
sys.path.insert(1, '/home/onuralp/Desktop/c243/neural_seq_decoder_project/src')
from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)

# # test different smoothing values for CTC loss
# for smooth_value in [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1]:
#     modelName = f'CTCsmoothing{smooth_value}'
#     args['CTCsmoothing'] = smooth_value
#     args['outputDir'] = "/Users/KatherynZhou/Desktop/BCI class/neural_seq_decoder_project/models/" + modelName
#     trainModel(args)

# args['CTCsmoothing'] = 0.1

# for nUnits in [32, 64, 128, 256, 512, 1024, 2048]:
#     modelName = f'nUnits{nUnits}'
#     args['nUnits'] = nUnits
#     args['outputDir'] = "/Users/KatherynZhou/Desktop/BCI class/neural_seq_decoder_project/models/" + modelName
#     trainModel(args)

# args['nUnits'] = 1024



"""
#mask_num parameter search
for mask_num in [0, 2, 4, 8, 16]:
    modelName = f'katherine_best_mask_num{mask_num}'
    args['nMasks'] = mask_num
    args['outputDir'] = '/home/onuralp/Desktop/c243/neural_seq_decoder_project/logs/speech_logs/' + modelName
    trainModel(args)

args['nMasks'] = 2
#stride_len parameter search
for stride_len in [2, 4, 8, 16, 32]:
    modelName = f'katherine_best_stride_len{stride_len}'
    args['strideLen'] = stride_len
    args['outputDir'] = '/home/onuralp/Desktop/c243/neural_seq_decoder_project/logs/speech_logs/' + modelName
    trainModel(args)
"""


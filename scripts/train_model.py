import numpy as np

modelName = 'katherine_best_beamdecoding'

batch_factor = 1

args = {}
args['outputDir'] = '/home/onuralp/Desktop/c243/neural_seq_decoder_project/logs/speech_logs/' + modelName
args['datasetPath'] = '/home/onuralp/Desktop/c243/neural_seq_decoder/ptDecoder_ctc'
#args['outputDir'] = '/Users/KatherynZhou/Desktop/BCI class/neural_seq_decoder_project/models/' + modelName
#args['datasetPath'] = '/Users/KatherynZhou/Desktop/BCI class/neural_seq_decoder_project/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = int(64*batch_factor)
args['lrStart'] = 0.03
args['lrEnd'] = 0.002
args['nUnits'] = 1024
args['nBatch'] = int(15000//batch_factor)
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
args['warmupSteps'] = int(500//batch_factor)
args['nMasks'] = 8 # number of time masks to implement per batch, make 0 to skip time masking
args['maxMaskLength'] = 40 # max number of timesteps to mask per single mask
args['layerNorm'] = True # whether or not to have layernorm layer between GRU and output
args['causalGaussian'] = True # whether to use causal Gaussian smoothing on the neural data
args['CTCsmoothing'] = 0.8
args['beamWidth'] = 1 # beam width for beam search decoding

import sys
sys.path.insert(1, '/home/onuralp/Desktop/c243/neural_seq_decoder_project/src')
from neural_decoder.neural_decoder_trainer import trainModel

#trainModel(args)

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

#args['beamWidth'] = 3
#stride_len parameter search
#args['CTCsmoothing'] = 0.8
"""
modelName = f'katherine_best_torch_best_beamwidth{1}_smoothing{0.8}'
args['outputDir'] = '/home/onuralp/Desktop/c243/neural_seq_decoder_project/logs/speech_logs/' + modelName
trainModel(args)
"""


for beamWidth in [1, 3, 5, 7]:
    args['beamWidth'] = beamWidth
    #stride_len parameter search
    for smoothing in [0, 0.2, 0.4, 0.6, 0.8]:
        args['CTCsmoothing'] = smoothing
        modelName = f'katherine_best_torch_fixed_beamwidth{beamWidth}_smoothing{smoothing}'
        args['outputDir'] = '/home/onuralp/Desktop/c243/neural_seq_decoder_project/logs/speech_logs/' + modelName
        trainModel(args)

args['batchSize'] = 64
args['nBatch'] = 15000
args['warmupSteps'] = 500
args['maxMaskLength'] = 20 
for beamWidth in [1, 3, 5, 7]:
    args['beamWidth'] = beamWidth
    #stride_len parameter search
    for smoothing in [0, 0.2, 0.4, 0.6, 0.8]:
        args['CTCsmoothing'] = smoothing
        modelName = f'katherine_best_torch_curiosity_beamwidth{beamWidth}_smoothing{smoothing}'
        args['outputDir'] = '/home/onuralp/Desktop/c243/neural_seq_decoder_project/logs/speech_logs/' + modelName
        trainModel(args)

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

# batch size testing for regularization effect
"""
for batch_factor in [1/16,1/8,1/4,1/2, 1, 2, 4]:
    modelName = f'speechBaseline4_batchFactor{batch_factor:.4g}'.replace(".", "_")
    args['batchSize'] = int(64*batch_factor)
    args['nBatch'] = int(10000/batch_factor)
    args['outputDir'] = '/home/onuralp/Desktop/c243/neural_seq_decoder_project/logs/speech_logs/' + modelName
    trainModel(args)
"""
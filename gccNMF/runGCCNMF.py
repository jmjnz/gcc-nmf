'''
The MIT License (MIT)

Copyright (c) 2016 Sean UN Wood

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

@author: Sean UN Wood
'''

from gccNMFFunctions import *
from gccNMFPlotting import *

def runGCCNMF(mixtureFilePrefix, windowSize, hopSize, numTDOAs, microphoneSeparationInMetres, numTargets=None, windowFunction=hanning):
    ############################################
    # 1. Load mixture signal from the wav file #
    ############################################
    maxTDOA = microphoneSeparationInMetres / SPEED_OF_SOUND_IN_METRES_PER_SECOND
    tdoasInSeconds = linspace(-maxTDOA, maxTDOA, numTDOAs).astype(float32)
    mixtureFileName = getMixtureFileName(mixtureFilePrefix)
    stereoSamples, sampleRate = loadMixtureSignal(mixtureFileName)

    # Plot mixture signal

    numChannels, numSamples = stereoSamples.shape
    durationInSeconds = numSamples / float(sampleRate)
    describeMixtureSignal(stereoSamples, sampleRate)
    figure(figsize=(14,6))
    plotMixtureSignal(stereoSamples, sampleRate)

    ##########################################################################
    # 2. Compute complex mixture spectrograms from input signal with an STFT #
    ##########################################################################
    complexMixtureSpectrogram = computeComplexMixtureSpectrogram(stereoSamples, windowSize, hopSize, windowFunction)
    numChannels, numFrequencies, numTime = complexMixtureSpectrogram.shape
    frequenciesInHz = getFrequenciesInHz(sampleRate, numFrequencies)

    # Plot magnitude mixture spectrograms

    frequenciesInkHz = frequenciesInHz/1000.0
    describeMixtureSpectrograms(windowSize, hopSize, windowFunction, complexMixtureSpectrogram)
    figure(figsize=(12, 8))
    plotMixtureSpectrograms(complexMixtureSpectrogram, frequenciesInkHz, durationInSeconds)

    #################################################################################################
    # 3. Compute NMF decomposition, with left and right magnitude spectrograms concatenated in time #
    #################################################################################################
    V = concatenate( abs(complexMixtureSpectrogram), axis=-1 )
    W, H = performKLNMF(V, dictionarySize=128, numIterations=100, sparsityAlpha=0)
    stereoH = array( hsplit(H, numChannels) )

    # Plot NMF decomposition

    describeNMFDecomposition(V, W, H)
    figure(figsize=(12, 12))
    plotNMFDecomposition(V, W, H, frequenciesInkHz, durationInSeconds, numAtomsToPlot=15)

    ###################################
    # 4. Localize target TDOA indexes #
    ###################################
    spectralCoherenceV = complexMixtureSpectrogram[0] * complexMixtureSpectrogram[1].conj() / abs(complexMixtureSpectrogram[0]) / abs(complexMixtureSpectrogram[1])
    angularSpectrogram = getAngularSpectrogram(spectralCoherenceV, frequenciesInHz, microphoneSeparationInMetres, numTDOAs)
    meanAngularSpectrum = mean(angularSpectrogram, axis=-1) 
    targetTDOAIndexes = estimateTargetTDOAIndexesFromAngularSpectrum(meanAngularSpectrum, microphoneSeparationInMetres, numTDOAs, numTargets)

    # Plot target localization

    figure(figsize=(14, 6))
    plotGCCPHATLocalization(spectralCoherenceV, angularSpectrogram, meanAngularSpectrum,targetTDOAIndexes, microphoneSeparationInMetres, numTDOAs,durationInSeconds)

    ####################################################
    # 5. Compute NMF coefficient masks for each target #
    ####################################################
    targetTDOAGCCNMFs = getTargetTDOAGCCNMFs(spectralCoherenceV, microphoneSeparationInMetres, numTDOAs, frequenciesInHz, targetTDOAIndexes, W, stereoH)
    targetCoefficientMasks = getTargetCoefficientMasks(targetTDOAGCCNMFs, numTargets)

    # Plot NMF coefficient masks for each target, and resulting masked coefficients for each channel

    figure(figsize=(12, 12))
    plotCoefficientMasks(targetCoefficientMasks, stereoH, durationInSeconds)

    ###############################################################################################################
    # 6. Reconstruct source spectrogram estimates using masked NMF coefficients for each target, and each channel #
    ###############################################################################################################
    targetSpectrogramEstimates = getTargetSpectrogramEstimates(targetCoefficientMasks, complexMixtureSpectrogram, W, stereoH)

    # Plot reconstructed source estimate spectrograms

    figure(figsize=(12, 12))
    plotTargetSpectrogramEstimates(targetSpectrogramEstimates, durationInSeconds, frequenciesInkHz)

    ####################################################################################################################
    # 7. Combine source estimate spectrograms with the input mixture spectrogram's phase, and perform the inverse STFT #
    ####################################################################################################################
    targetSignalEstimates = getTargetSignalEstimates(targetSpectrogramEstimates, windowSize, hopSize, windowFunction)
    saveTargetSignalEstimates(targetSignalEstimates, sampleRate, mixtureFileNamePrefix)

    # Plot time domain source signal estimates

    for sourceIndex in range(numSources):
        figure(figsize=(14, 2))
        fileName = getSourceEstimateFileName(mixtureFileNamePrefix, sourceIndex)
        plotTargetSignalEstimate(targetSignalEstimates[sourceIndex], sampleRate, 'Source %d' % (sourceIndex + 1))


if __name__ == '__main__':
    # Preprocessing params
    windowSize = 1024
    fftSize = windowSize
    hopSize = 128
    # hopSize = 64

    windowFunction = hanning
    
    # TDOA params
    numTDOAs = 128

    # NMF params
    dictionarySize = 128
    # dictionarySize = 1024

    numIterations = 100
    sparsityAlpha = 0

    mixtureFileNamePrefix = '../data/dev1_female3_liverec_130ms_1m' # add _mix automatically
    microphoneSeparationInMetres = 1
    numSources = 3

    # mixtureFileNamePrefix = '../data/src_cut'
    # microphoneSeparationInMetres = 0.154
    # numSources = 2
    
    runGCCNMF( mixtureFileNamePrefix, windowSize, hopSize, numTDOAs,
               microphoneSeparationInMetres, numSources, windowFunction )
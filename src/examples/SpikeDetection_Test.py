import sys
sys.path.append('../')

import numpy as np

from Experiment import *
from AEC_Badel import *
from Filter_Rect_LogSpaced import *
from Filter_Rect_LinSpaced import *


"""
This file tests the numpy-vectorized method for spike detection.


First, this test file shows that the numpy method produces identical output to 
the original weave and python methods across multiple spiketrains under normal
conditions.*

Second, speed tests show that the numpy method is ~8X faster than the weave
method, and ~750X faster than the original python method.


Because weave is deprecated in python 3 while numpy continues to be 
well-supported, using the numpy method by default will improve the 
maintainability of the code without sacrificing performance.


*Note: It is possible to obtain different results using the new and old methods
if tref is set to be longer than the shortest (true) inter-spike-intervals in 
the recording. Obviously, this should never happen.
"""


############################################################################################################
# STEP 1: LOAD EXPERIMENTAL DATA
############################################################################################################

myExp = Experiment('spkDetectionTest', 0.1)

PATH = '../../data/gif_test/'

# Load AEC data
myExp.setAECTrace(V = PATH + 'Cell3_Ger1Elec_ch2_1007.ibw', V_units = 1.0, 
                  I = PATH + 'Cell3_Ger1Elec_ch3_1007.ibw', I_units = 1.0, 
                  T = 10000.0, FILETYPE='Igor')

# Load training set data
myExp.addTrainingSetTrace(V = PATH + 'Cell3_Ger1Training_ch2_1008.ibw', V_units = 1.0,
                          I = PATH + 'Cell3_Ger1Training_ch3_1008.ibw', I_units = 1.0, 
                          T = 120000.0, FILETYPE='Igor')

# Load test set data
myExp.addTestSetTrace(V = PATH + 'Cell3_Ger1Test_ch2_1009.ibw', V_units = 1.0, 
                      I = PATH + 'Cell3_Ger1Test_ch3_1009.ibw', I_units = 1.0, 
                      T = 20000.0, FILETYPE='Igor')
myExp.addTestSetTrace(V = PATH + 'Cell3_Ger1Test_ch2_1010.ibw', V_units = 1.0, 
                      I = PATH + 'Cell3_Ger1Test_ch3_1010.ibw', I_units = 1.0, 
                      T = 20000.0, FILETYPE='Igor')
myExp.addTestSetTrace(V = PATH + 'Cell3_Ger1Test_ch2_1011.ibw', V_units = 1.0, 
                      I = PATH + 'Cell3_Ger1Test_ch3_1011.ibw', I_units = 1.0, 
                      T = 20000.0, FILETYPE='Igor')
myExp.addTestSetTrace(V = PATH + 'Cell3_Ger1Test_ch2_1012.ibw', V_units = 1.0, 
                      I = PATH + 'Cell3_Ger1Test_ch3_1012.ibw', I_units = 1.0, 
                      T = 20000.0, FILETYPE='Igor')
myExp.addTestSetTrace(V = PATH + 'Cell3_Ger1Test_ch2_1013.ibw', V_units = 1.0, 
                      I = PATH + 'Cell3_Ger1Test_ch3_1013.ibw', I_units = 1.0, 
                      T = 20000.0, FILETYPE='Igor')
myExp.addTestSetTrace(V = PATH + 'Cell3_Ger1Test_ch2_1014.ibw', V_units = 1.0, 
                      I = PATH + 'Cell3_Ger1Test_ch3_1014.ibw', I_units = 1.0, 
                      T = 20000.0, FILETYPE='Igor')
myExp.addTestSetTrace(V = PATH + 'Cell3_Ger1Test_ch2_1015.ibw', V_units = 1.0, 
                      I = PATH + 'Cell3_Ger1Test_ch3_1015.ibw', I_units = 1.0, 
                      T = 20000.0, FILETYPE='Igor')
myExp.addTestSetTrace(V = PATH + 'Cell3_Ger1Test_ch2_1016.ibw', V_units = 1.0, 
                      I = PATH + 'Cell3_Ger1Test_ch3_1016.ibw', I_units = 1.0, 
                      T = 20000.0, FILETYPE='Igor')
myExp.addTestSetTrace(V = PATH + 'Cell3_Ger1Test_ch2_1017.ibw', V_units = 1.0, 
                      I = PATH + 'Cell3_Ger1Test_ch3_1017.ibw', I_units = 1.0, 
                      T = 20000.0, FILETYPE='Igor')


############################################################################################################
# STEP 2: ACTIVE ELECTRODE COMPENSATION
############################################################################################################

# Create new object to perform AEC
myAEC = AEC_Badel(myExp.dt)

# Define metaparametres
myAEC.K_opt.setMetaParameters(length=150.0, binsize_lb=myExp.dt, 
                              binsize_ub=2.0, slope=30.0, clamp_period=1.0)
myAEC.p_expFitRange = [3.0,150.0]  
myAEC.p_nbRep = 15     

# Assign myAEC to myExp and compensate the voltage recordings
myExp.setAEC(myAEC)  
myExp.performAEC()  


############################################################################################################
# STEP 3: DETECT SPIKES
############################################################################################################

# Detect spks using weave
spks_weave = []

myExp.trainingset_traces[0].detectSpikes_weave()
spks_weave.append(myExp.trainingset_traces[0].spks.copy())

for tr in myExp.testset_traces:
    tr.detectSpikes_weave()
    spks_weave.append(tr.spks.copy())

# Detect spks using python
spks_python = []

myExp.trainingset_traces[0].detectSpikes_python()
spks_python.append(myExp.trainingset_traces[0].spks.copy())

for tr in myExp.testset_traces:
    tr.detectSpikes_python()
    spks_python.append(tr.spks.copy())

# Detect spks using new numpy method
spks_quickpy = []

myExp.trainingset_traces[0].detectSpikes()
spks_quickpy.append(myExp.trainingset_traces[0].spks.copy())

for tr in myExp.testset_traces:
    tr.detectSpikes()
    spks_quickpy.append(tr.spks.copy())


############################################################################################################
# STEP 4: COMPARE SPIKETRAINS
############################################################################################################

# Print tests for identical output

weave_vs_base = all(
        [np.array_equal(arr_1, arr_2) for arr_1, arr_2 in zip(spks_weave, spks_python)])
print('\nWeave and base python methods produce identical output'
      ' (positive control): {}'.format(
              weave_vs_base))

weave_vs_quickpy = all(
        [np.array_equal(arr_1, arr_2) for arr_1, arr_2 in zip(spks_weave, spks_quickpy)])
print('Weave and numpy methods produce identical output: {}'.format(
        weave_vs_quickpy))


############################################################################################################
# STEP 4: COMPARE SPEED
############################################################################################################

# Print speed tests
print('\nTiming weave method...')
%timeit myExp.trainingset_traces[0].detectSpikes_weave()    # ~8ms

print('\nTiming base python method...')
%timeit myExp.trainingset_traces[0].detectSpikes_python()   # ~740ms

print('\nTiming numpy method...')
%timeit myExp.trainingset_traces[0].detectSpikes()          # ~1ms
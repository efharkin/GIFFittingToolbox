import sys
sys.path.append('../')

import numpy as np

from Experiment import *


"""
This file tests the numpy-vectorized method for spike detection.


First, this test file shows that the numpy method produces identical output to 
the original weave and python methods.

Second, speed tests show that the numpy method is only marginally (~1ms) slower
than the weave method, and ~100X faster than the original python method.


Because weave is deprecated in python 3 while numpy continues to be 
well-supported, using the numpy method by default will improve the 
maintainability of the code without sacrificing performance.
"""

PATH = '../../data/gif_test/'

myExp = Experiment('spkDetectionTest', 0.1)
myExp.addTrainingSetTrace(V = PATH + 'Cell3_Ger1Training_ch2_1008.ibw', V_units = 1.0,
                          I = PATH + 'Cell3_Ger1Training_ch3_1008.ibw', I_units = 1.0, 
                          T = 120000.0, FILETYPE='Igor')

# Detect spks using weave
myExp.trainingset_traces[0].detectSpikes()
spks_weave = myExp.trainingset_traces[0].spks.copy()

# Detect spks using python
myExp.trainingset_traces[0].detectSpikes_python()
spks_python = myExp.trainingset_traces[0].spks.copy()

# Detect spks using new numpy method
myExp.trainingset_traces[0].detectSpikes_quickpy()
spks_quickpy = myExp.trainingset_traces[0].spks.copy()

# Print tests for identical output
print('\nWeave and base python methods produce identical output'
      ' (positive control): {}'.format(
              np.array_equal(spks_weave, spks_python)))
print('Weave and numpy methods produce identical output: {}'.format(
        np.array_equal(spks_weave, spks_quickpy)))


# Print speed tests
print('\nTiming weave method...')
%timeit myExp.trainingset_traces[0].detectSpikes()          # ~7.7ms

print('\nTiming base python method...')
%timeit myExp.trainingset_traces[0].detectSpikes_python()   # ~740ms

print('\nTiming numpy method...')
%timeit myExp.trainingset_traces[0].detectSpikes_quickpy()  # ~8.4ms
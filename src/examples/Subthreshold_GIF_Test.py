import sys
sys.path.append('../')

from Experiment import *
from AEC_Badel import *
from SubthreshGIF import *
from Filter_Rect_LogSpaced import *
from Filter_Rect_LinSpaced import *

import matplotlib.pyplot as plt


"""
This file tests whether the current implementation of the GIF model can be
fitted to recordings without any spikes.
"""

############################################################################################################
# STEP 1: LOAD EXPERIMENTAL DATA
############################################################################################################
myExp = Experiment('Fully subthreshold test', 0.1)

PATH = '../../data/subthreshold_gif_test/'

# Load AEC data
myExp.setAECTrace(FILETYPE = 'Axon', 
                  fname = PATH + 'subthresh_AEC.abf', V_channel = 0, I_channel = 1)

# Load training set data
myExp.addTrainingSetTrace(FILETYPE = 'Axon',
                          fname = PATH + 'subthresh_train.abf', V_channel = 0, I_channel = 1)

# Load test set data
myExp.addTestSetTrace(FILETYPE = 'Axon',
                      fname = PATH + 'subthresh_test.abf', V_channel = 0, I_channel = 1)

# Plot data
#myExp.plotTrainingSet()
#myExp.plotTestSet()


############################################################################################################
# STEP 2: ACTIVE ELECTRODE COMPENSATION
############################################################################################################

# Create new object to perform AEC
myAEC = AEC_Badel(myExp.dt)

# Define metaparametres
myAEC.K_opt.setMetaParameters(length=150.0, binsize_lb=myExp.dt, binsize_ub=2.0, slope=30.0, clamp_period=1.0)
myAEC.p_expFitRange = [3.0,150.0]  
myAEC.p_nbRep = 15     

# Assign myAEC to myExp and compensate the voltage recordings
myExp.setAEC(myAEC)  
myExp.performAEC()  

# Plot AEC filters (Kopt and Ke)
myAEC.plotKopt()
myAEC.plotKe()

# Plot training and test set
myExp.plotTrainingSet()
myExp.plotTestSet()


############################################################################################################
# STEP 3: TEST SPIKE DETECTION AND ROI SELECTION IN Trace
############################################################################################################

# Test spike detection
print '\nTesting spike detection...'
try:
    myExp.trainingset_traces[0].detectSpikes()
except:
    print('Spike detection test failed.\n')
    raise
print 'Success!\n'

# Test ROI selection
print '\nTesting ROI selection...'
try:
    myExp.trainingset_traces[0].getROI_FarFromSpikes(5., 5.)
except:
    print('getROI_FarFromSpikes test failed.\n')
    raise
print('Success!\n')


############################################################################################################
# STEP 4: FIT GIF MODEL TO DATA
############################################################################################################

# Create a new object GIF 
myGIF = SubthreshGIF(0.1)

# Define parameters
myGIF.Tref = 4.0  

myGIF.eta = Filter_Rect_LogSpaced()
myGIF.eta.setMetaParameters(length=500.0, binsize_lb=2.0, binsize_ub=1000.0, slope=4.5)


myGIF.gamma = Filter_Rect_LogSpaced()
myGIF.gamma.setMetaParameters(length=500.0, binsize_lb=5.0, binsize_ub=1000.0, slope=5.0)

# Define the ROI of the training set to be used for the fit (in this example we will use only the first 100 s)
myExp.trainingset_traces[0].setROI([[2000,58000]])

# To visualize the training set and the ROI call again
myExp.plotTrainingSet()

# Perform the fit
print '\nTesting model fitting...'
try:
    myGIF.fit(myExp, DT_beforeSpike=5.0)
except:
    print 'Model fit test failed.\n'
    raise
print '\nSuccessful model fit!\n'

# Plot the model parameters
myGIF.printParameters()
myGIF.plotParameters()   

## Save the model
#myGIF.save('./myGIF.pck')


############################################################################################################
# STEP 5: EXTRACT POWER SPECTRUM DENSITY AND COMPARE BETWEEN MODEL AND DATA
############################################################################################################

# Test PSD extraction from data Trace
print '\nTesting PSD extraction...'
try:
    myExp.trainingset_traces[0].extractPowerSpectrumDensity(True)
except:
    print 'PSD extraction test failed.\n'
    raise
print '\nSuccessful PSD extraction!\n'


experimental_PSD = myExp.trainingset_traces[0].extractPowerSpectrumDensity()

t, V_sim = myGIF.simulate(myExp.trainingset_traces[0].I,
                          myExp.trainingset_traces[0].V[0])
GIF_PSD = Trace(V_sim, 
                myExp.trainingset_traces[0].I, 
                60000, 0.1).extractPowerSpectrumDensity()

plt.figure(figsize = (10, 4))

ax = plt.subplot(111)
ax.set_yscale('log')

ax.plot(experimental_PSD[0], experimental_PSD[1],
        'k-', linewidth = 0.5, label = 'Data')
ax.plot(GIF_PSD[0], GIF_PSD[1],
        'r-', linewidth = 0.5, alpha = 0.5, label = 'Simulated')

ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('PSD')
ax.legend()

plt.tight_layout()
plt.show()

############################################################################################################
# STEP 4A (OPTIONAL): PLAY A BIT WITH THE FITTED MODEL
############################################################################################################

## Reload the model
#myGIF = GIF.load('./myGIF.pck')
#
## Generate OU process with temporal correlation 3 ms and mean modulated by a sinusoildal function of 1 Hz
#I_OU = Tools.generateOUprocess_sinMean(f=1.0, T=5000.0, tau=3.0, mu=0.3, delta_mu=0.5, sigma=0.1, dt=0.1)
#
## Simulate the model with the I_OU current. Use the reversal potential El as initial condition (i.e., V(t=0)=El)
#(time, V, I_a, V_t, S) = myGIF.simulate(I_OU, myGIF.El)
#
## Plot the results of the simulation
#plt.figure(figsize=(14,5), facecolor='white')
#plt.subplot(2,1,1)
#plt.plot(time, I_OU, 'gray')
#plt.ylabel('I (nA)')
#plt.subplot(2,1,2)
#plt.plot(time, V,'black', label='V')
#plt.plot(time, V_t,'red', label='V threshold')
#plt.ylabel('V (mV)')
#plt.xlabel('Time (ms)')
#plt.legend()
#plt.show()
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np

import weave


class Trace :

    """
    An experiments includes many experimental traces.
    A Trace contains the experimental data acquired during a single current-clamp injection (e.g., the training set injection, or one repetition of the test set injections) 
    """

    def __init__(self, V, I, T, dt):
        
        """
        V : vector with recorded voltage (mV)
        I : vector with injected current (nA)
        T : length of the recording (ms)
        dt : timestep of recording (ms)
        """
        
        # Perform input checks
        if len(V) != len(I):
            raise ValueError('Could not create Trace using V and I with non-'
                             'identical lengths {} and {}.'.format(
                                     len(V), len(I)))
        if len(V) != int(np.round(T / dt)):
            warn(RuntimeWarning('V array is not of length T/dt; expected {}, ' 
                                'got {}.'.format(int(np.round(T/dt)), len(V))))
        
        # Initialize main attributes related to recording
        self.V_rec = np.array(V, dtype = 'double')                         # mV, recorded voltage (before AEC)
        self.V     = self.V_rec                                            # mV, voltage (after AEC)
        self.I     = np.array(I, dtype = 'double')                         # nA, injected current
        self.T     = T                                                     # ms, duration of the recording    
        self.dt    = dt                                                    # ms, timestep
        
        # Initialize flags
        self.AEC_flag    = False                                           # Has the trace been preprocessed with AEC?
        
        self.spks_flag   = False               # Do spikes have been detected?
        self.spks        = 0                   # spike indices stored in indices (and not in ms!)  
                
        self.useTrace    = True                # if false this trace will be neglected while fitting spiking models to data        
        self.ROI   = [ [0, len(self.V_rec)*self.dt] ] # List of intervals to be used for fitting; includes the whole trace by default
    
    
    
    #################################################################################################
    # FUNCTIONS ASSOCIATED WITH ROI
    #################################################################################################

    def enable(self):
        
        """
        If you want to use this trace call this function. By default traces are enabled.
        When enable is called, ROI is set to be the entire trace.
        """
            
        self.useTrace = True
        self.ROI      = [ [0, len(self.V_rec)*self.dt] ]


    def disable(self):
        
        """
        If you dont want to use this trace during the fit, call this function.
        """
        
        self.useTrace = False
        self.ROI      = [ [0, 0] ]
     
     
    def setROI(self, ROI_intervals):

        """
        ROI intervals are defined in ms.
        Use this function the specify which parts of the trace have to be used for fitting.
        """
        self.useTrace = True
        self.ROI = ROI_intervals   
        

    def getROI(self):
        
        """
        Return indices of the trace which are in ROI
        """
        
        ROI_region = np.zeros(int(self.T/self.dt), dtype = np.bool)

        for ROI_interval in self.ROI :
            ROI_region[int(ROI_interval[0]/self.dt) : int(ROI_interval[1]/self.dt)] = True
        
        ROI_ind = np.where(ROI_region)[0]
        
        # Make sure indices are ok
        ROI_ind = ROI_ind[ np.where(ROI_ind<int(self.T/self.dt))[0] ]
        
        return ROI_ind
    
    
    def getROI_FarFromSpikes(self, DT_before, DT_after):

        """
        Return indices of the trace which are in ROI. Exclude all datapoints which are close to a spike.
        DT_before: ms
        DT_after: ms
        These two parameters define the region to cut around each spike.
        """
        
        L = len(self.V)
        
        LR_flag = np.ones(L, dtype = np.bool)    
        
        
        # Select region in ROI
        ROI_ind = self.getROI()
        LR_flag[ROI_ind] = False

        # Remove region around spikes
        DT_before_i = int(DT_before/self.dt)
        DT_after_i  = int(DT_after/self.dt)
        
        
        for s in self.spks :
            
            lb = max(0, s - DT_before_i)
            ub = min(L, s + DT_after_i)
            
            LR_flag[ lb : ub] = True
            
        
        indices = np.where(~LR_flag)[0]  

        return indices

    
    def getROI_cutInitialSegments(self, DT_initialcutoff):

        """
        Return indices of the trace which are in ROI. Exclude all initial segments in each ROI.
        DT_initialcutoff: ms, width of region to cut at the beginning of each ROI section.
        """
        
        DT_initialcutoff_i = int(DT_initialcutoff/self.dt)
        ROI_region = np.zeros(int(self.T/self.dt), dtype = np.bool)

        for ROI_interval in self.ROI :
            
            lb = int(ROI_interval[0]/self.dt) + DT_initialcutoff_i            
            ub = int(ROI_interval[1]/self.dt)
            
            if lb < ub :
                ROI_region[lb:ub] = True
        
        ROI_ind = np.where(ROI_region)[0]
        
        # Make sure indices are ok
        ROI_ind = ROI_ind[ np.where(ROI_ind<int(self.T/self.dt))[0] ]
        
        return ROI_ind


    #################################################################################################
    # FUNCTIONS ASSOCIATED TO SPIKES IN THE TRACE
    #################################################################################################
    
    def detectSpikes(self, threshold=0.0, ref=3.0):
        
        """
        Detect action potentials by threshold crossing (parameter threshold, mV) from below (i.e. with dV/dt>0).
        To avoid multiple detection of same spike due to noise, use an 'absolute refractory period' ref, in ms.
        Fast, vectorized implementation using numpy.
        """ 
        
        # Convert refractory period into index-based units
        ref_ind = int(np.round(ref/self.dt))
        
        # Detect points above or below threshold to get rising edges
        above_thresh = self.V >= threshold
        below_thresh = ~above_thresh
        
        rising_edges = above_thresh[1:] & below_thresh[:-1]
        
        # Convert rising edges to spk inds
        spks = np.where(rising_edges)[0] + 1
        
        # Remove points that reference the same spk
        redundant_pts = np.where(np.diff(spks) <= ref_ind)[0] + 1
        spks = np.delete(spks, redundant_pts)
        
        # Assign output
        self.spks = spks
        self.spks_flag = True
        
    
    def detectSpikes_python(self, threshold=0.0, ref=3.0):
        
        """
        Detect action potentials by threshold crossing (parameter threshold, mV) from below (i.e. with dV/dt>0).
        To avoid multiple detection of same spike due to noise, use an 'absolute refractory period' ref, in ms.
        """ 
        
        self.spks = []
        ref_ind = int(ref/self.dt)
        t=0
        while (t<len(self.V)-1) :
            
            if (self.V[t] >= threshold and self.V[t-1] <= threshold) :
                self.spks.append(t)
                t+=ref_ind
            t+=1
                        
        self.spks = np.array(self.spks)
        self.spks_flag = True


    def detectSpikes_weave(self, threshold=0.0, ref=3.0):
        
        """
        Detect action potentials by threshold crossing (parameter threshold, mV) from below (i.e. with dV/dt>0).
        To avoid multiple detection of same spike due to noise, use an 'absolute refractory period' ref, in ms.
        Code implemented in C.
        """ 
        
        # Define parameters
        p_T_i     = int(np.round(self.T/self.dt))
        p_ref_ind = int(np.round(ref/self.dt))
        p_threshold = threshold

        # Define vectors
        V  =   np.array(self.V, dtype='double')

                
        spike_train = np.zeros(p_T_i)
        spike_train = np.array(spike_train, dtype='double')
        
                
        code =  """
                #include <math.h>
                
                int T_i = int(p_T_i)-1;                
                int ref_ind = int(p_ref_ind);   
                float threshold = p_threshold;
            
                int t = 0;
                                                                
                while (t < T_i) {
                    
                    if (V[t] >= threshold && V[t-1] < threshold) {
                        spike_train[t] = 1.0;
                        t += ref_ind;
                    }
                    
                    t++;
               
                }  
                """
 
        vars = [ 'p_T_i', 'p_ref_ind', 'p_threshold', 'V', 'spike_train' ]
        
        v = weave.inline(code, vars)

        spks_ind = np.where(spike_train==1.0)[0]

        self.spks = np.array(spks_ind)
        self.spks_flag = True



    def computeAverageSpikeShape(self):
        
        """
        Compute the average spike shape using spikes in ROI.
        """
        
        DT_before = 10.0
        DT_after = 20.0
        
        DT_before_i = int(DT_before/self.dt)
        DT_after_i  = int(DT_after/self.dt)
    
        if self.spks_flag == False :
            self.detectSpikes()
         
                
        all_spikes = []
        
        ROI_ind = self.getROI()

        for s in self.spks :
                        
            # Only spikes in ROI are used
            if s in ROI_ind :
                
                # Avoid using spikes close to boundaries to avoid errors             
                if s > DT_before_i  and s < (len(self.V) - DT_after_i) :
                    all_spikes.append( self.V[ s - DT_before_i : s + DT_after_i ] )

    
        spike_avg = np.mean(all_spikes, axis=0)
        
        support   = np.linspace(-DT_before, DT_after, len(spike_avg))
        spike_nb  = len(all_spikes)
        
        return (support, spike_avg, spike_nb)


    def getSpikeTrain(self):
        
        """
        Return spike train defined as a vector of 0s and 1s. Each bin represent self.dt 
        """
                
        spike_train = np.zeros( int(self.T/self.dt) )
        
        if len(self.spks) > 0 :
        
            spike_train[self.spks] = 1

        return spike_train


    def getSpikeTimes(self):
        
        """
        Return spike times in units of ms.
        """
        
        return self.spks*self.dt



    def getSpikeIndices(self):
        
        """
        Return spike indices in units of dt.
        """
        
        return self.spks


    def getSpikeNb(self):
        
        return len(self.spks)



    def getSpikeNbInROI(self):
        
        """
        Return number of spikes in region of interest.
        """
        
        ROI_ind = self.getROI()
        spike_train = self.getSpikeTrain()
        
        return sum(spike_train[ROI_ind])
        
        


    #################################################################################################
    # GET STATISTICS, COMPUTED IN ROI
    #################################################################################################  
    
    def getSpikeNb_inROI(self):
        
        if len(self.spks) == 0 :
            
            return 0
        
        else :
        
            spike_train = self.getSpikeTrain()
            ROI_ind = self.getROI()  
            
            nbSpikes = np.sum(spike_train[ROI_ind])
            
            return nbSpikes


    def getTraceLength_inROI(self):
        
        """
        Return in ms the duration of ROI region.
        """
        ROI_ind = self.getROI()  
        
        return (len(ROI_ind)*self.dt)



    def getFiringRate_inROI(self):
        
        """
        Return the average firing rate (in Hz) in ROI.
        """
        
        return 1000.0 * self.getSpikeNb_inROI()/self.getTraceLength_inROI()
    
    
    #################################################################################################
    # GET TIME
    #################################################################################################    
    
    def getTime(self):
        
        """
        Get time vector (i.e., the temporal support of the arrays: I, V, etc)
        """
        
        return np.arange(int(self.T/self.dt))*self.dt        


    #################################################################################################
    # FUNCTIONS ASSOCIATED WITH PLOTTING
    #################################################################################################
    
    def plot(self):
        
        """
        Plot input current, recorded voltage, voltage after AEC (is applicable) and detected spike times (if applicable)
        """
        
        time = self.getTime()
        
        plt.figure(figsize=(10,4), facecolor='white')
                
        plt.subplot(2,1,1)
        plt.plot(time,self.I, 'gray')
        plt.ylabel('I (nA)')
        
        plt.subplot(2,1,2)
        plt.plot(time,self.V_rec, 'black')
        
        if self.AEC_flag :
            plt.plot(time,self.V, 'red')  

        if self.spks_flag :
            plt.plot(self.getSpikeTimes(),np.zeros(len(self.spks)), '.', color='blue')  
        
        # Plot ROI (region selected for performing operations)   
        ROI_vector = 100.0*np.ones(len(self.V))      
        ROI_vector[self.getROI() ] = -100.0
        plt.fill_between(self.getTime(), ROI_vector, -100.0, color='0.2')
        
        
        plt.ylim([min(self.V)-5.0, max(self.V)+5.0])
        plt.ylabel('V rec (mV)')
        plt.xlabel('Time (ms)')
        plt.show()
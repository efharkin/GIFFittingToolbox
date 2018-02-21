from warnings import warn

import matplotlib.pyplot as plt
import cPickle as pkl
import quantities as pq
from neo import AxonIO

from SpikeTrainComparator import *
from SpikingModel import *
from Trace import *
from AEC_Dummy import *


class Experiment :
    
    """
    Objects of this class contain the experimental data.
    According to the experimental protocol proposed in Pozzorini et al. PLOS Comp. Biol. 2015 (see Fig. 4) an experimental dataset contains:
    - AEC trace (short and small amplitude subthreshold current injection)
    - AEC training set trace (current clamp injections used to estimate model parameters)
    - AEC test set traces (several current clamp injections of frozen noise used to assess the predictive power of a model)
    Objects of this class have an AEC object that can be used to perform Active Electrode Compensation for data preprocessing.
    """
    
    def __init__(self, name, dt):
        
        """
        Name: string, name of the experiment
        dt: experimental time step (in ms). That is, 1/sampling frequency.
        """
        
        print "Create a new Experiment"

        self.name               = name          # Experiment name
        
        self.dt                 = dt            # ms, experimental time step (all traces in same experiment must have the same sampling frequency)  
        
        
        # Voltage traces
        
        self.AEC_trace          = 0             # Trace object containing voltage and input current used for AEC  
        
        self.trainingset_traces = []            # List of traces for training set data
        
        self.testset_traces     = []            # List of traces of test set data (typically multiple experiments conducted with frozen noise)
        
        
        # AEC object
        
        self.AEC                = AEC_Dummy()   # Object that performs AEC on experimental voltage traces


        # Parameters used to define spike times
        
        self.spikeDetection_threshold    = 0.0  # mV, voltage threshold used to detect spikes
        
        self.spikeDetection_ref          = 3.0  # ms, absolute refractory period used for spike detection to avoid double counting of spikes



    ############################################################################################
    # FUNCTIONS TO ADD TRACES TO THE EXPERIMENT
    ############################################################################################   
    
    @staticmethod
    def _readIgor(V, V_units, I, I_units, T, dt):
        
        """
        Internal method used to create Traces from Igor files.
        
        V : file address of recorded current
        V_units : units in which recorded voltage is stored (for mV use 10**-3)
        I : file address of input current
        I_units : units in which input current is stored (for nA use 10**-9)
        T : length of the recording (ms)
        dt : timestep (ms)
        """
        
        V_rec = ReadIBW.read(V)
        V_rec = np.array(V_rec[:int(T/dt)])*V_units/10**-3  # Convert to mV
        
        I = ReadIBW.read(I)
        I = np.array(I[:int(T/dt)])*I_units/10**-9  # Convert to nA
        
        return Trace(V_rec, I, T, dt)
        
    
    @staticmethod
    def _readABF(fname, V_channel, I_channel, dt):
        
        """
        Internal method used to create Traces from Axon Binary Format files.
        
        fname : file address
        V_channel : index of channel that contains recorded voltage
        I_channel : index of channel that contains injected current
        dt : timestep of recording (used to enforce consistency with Experiment)
        
        Units are automatically detected and converted to mV, nA, and ms.
        """
        
        # Read in sweeps
        sweeps = AxonIO(fname).read()[0].segments
        
        # Verify that V and I_channels are actually in V and A
        base_units = lambda chan: chan.units.simplified.dimensionality
        units = [base_units(chan) for chan in sweeps[0].analogsignals]
        
        if units[V_channel] != pq.V.simplified.dimensionality:
            raise RuntimeError('V_channel ({}) unit dimensionality must be V;'
                               ' got {} instead.'.format(
                                       V_channel, units[V_channel]))
        if units[I_channel] != pq.A.simplified.dimensionality:
            raise RuntimeError('I_channel ({}) unit dimensionality must be A;'
                               ' got {} instead.'.format(
                                       I_channel, units[I_channel]))
        
        # Extract sampling rate from first sweep of V_channel
        dt_tmp = 1./sweeps[0].analogsignals[V_channel].sampling_rate.rescale(pq.Hz)
        dt_tmp = 1000. * float(dt_tmp) # Convert to ms
        
        # Verify that dt of recording is same as value passed to method
        if dt_tmp != dt:
            raise RuntimeError('Got unexpected dt = {}ms from file during ABF'
                               ' import! (Expected {}ms.)'.format(dt_tmp, dt))
        
        # Initialize list to hold Trace objects
        tr_list = []
        
        # Iterate over sweeps
        for sweep in sweeps:
            
            # Convert to mV and nA
            V_tmp = sweep.analogsignals[V_channel].rescale(pq.mV)
            I_tmp = sweep.analogsignals[I_channel].rescale(pq.nA)
            
            # Strip units from Trace inputs
            V_tmp = V_tmp.magnitude.flatten()
            I_tmp = I_tmp.magnitude.flatten()
            T_tmp = float(sweep.analogsignals[V_channel].duration.rescale(pq.ms))
            
            # Add Trace to list
            tr_list.append(Trace(V_tmp, I_tmp, T_tmp, dt_tmp))
            
        return tr_list
        
    
    @staticmethod
    def _readArray(V, V_units, I, I_units, T, dt):
        
        """
        Internal method used to create Traces from vectors.
        """
        
        V_rec   = np.array(V[:int(T/dt)])*V_units/10**-3  # Convert to mV
        I       = np.array(I[:int(T/dt)])*I_units/10**-9  # Convert to nA
        
        return Trace(V_rec, I, T, dt)
    
    
    def _createTrace(self, FILETYPE='Axon', **kwargs):
        
        """
        Internal method used to create Traces from files or vectors.
        
        Selects the appropriate _readX staticmethod based on FILETYPE and verifies that correct arguments have been provided before creating a list of traces.
        """
        
        if FILETYPE=='Axon':
            
            # Check for required arguments
            required_kwargs = ['fname', 'V_channel', 'I_channel']
            if not all([kw in kwargs.keys() for kw in required_kwargs]) :
                raise TypeError(', '.join(required_kwargs) + ' must be'
                                ' supplied as kwargs for Axon FILETYPE.')
            
            # Warn user about unused arguments
            unused_kwargs = [
                    kw for kw in kwargs.keys() if kw not in required_kwargs]
            if len(unused_kwargs) > 0 :
                warn(RuntimeWarning(', '.join(unused_kwargs) + ' kwargs are not'
                                    ' required for Axon FILETYPE and will not'
                                    ' be used.'))
            
            # Read in file using protected static method
            tr_list_tmp = self._readABF(
                    kwargs['fname'], 
                    kwargs['V_channel'], 
                    kwargs['I_channel'],
                    self.dt)
            
            return tr_list_tmp
            
        elif FILETYPE=='Igor':
            
            # Check for required arguments
            required_kwargs = ['V', 'V_units', 'I', 'I_units', 'T']
            if not all([kw in kwargs.keys() for kw in required_kwargs]) :
                raise TypeError(', '.join(required_kwargs) + 'must be supplied'
                                ' as kwargs for Igor FILETYPE.')
            
            # Warn user about unused arguments
            unused_kwargs = [
                    kw for kw in kwargs.keys() if kw not in required_kwargs]
            if len(unused_kwargs) > 0 :
                warn(RuntimeWarning(', '.join(unused_kwargs) + ' kwargs are not'
                                    ' required for Igor FILETYPE and will not'
                                    ' be used.'))
                
            trace_tmp = self._readIgor(
                    kwargs['V'],
                    kwargs['V_units'],
                    kwargs['I'],
                    kwargs['I_units'],
                    kwargs['T'],
                    self.dt)
            
            return trace_tmp
        
        elif FILETYPE=='Array':
            
            # Check for required arguments
            required_kwargs = ['V', 'V_units', 'I', 'I_units', 'T', 'dt']
            if not all([kw in kwargs.keys() for kw in required_kwargs]) :
                raise TypeError(', '.join(required_kwargs) + 'must be supplied'
                                'as kwargs for Array FILETYPE.')
                
            # Warn user about unused kwargs
            unused_kwargs = [
                    kw for kw in kwargs.keys() if kw not in required_kwargs]
            if len(unused_kwargs) > 0 :
                warn(RuntimeWarning(', '.join(unused_kwargs) + ' kwargs are not'
                                    ' required for Array FILETYPE and will not'
                                    ' be used.'))
            
            trace_tmp = self._readArray(
                    kwargs['V'],
                    kwargs['V_units'],
                    kwargs['I'],
                    kwargs['I_units'],
                    kwargs['T'],
                    self.dt)
            
            return trace_tmp
        
        else:
            raise ValueError('Expected one of Axon, Igor, or Array for'
                             ' FILETYPE. Got {} instead.'.format(FILETYPE))
        
    
    
    def setAECTrace(self, FILETYPE='Axon', **kwargs):
    
        """
        Set AEC trace to experiment.
        
        V : recorded voltage (either file address or numpy array depending on FILETYPE)
        V_units : units in which recorded voltage is stored (for mV use 10**-3)
        I : input current (either file address or numpy array depending on FILETYPE)
        I_units : units in which the input current is stored (for nA use 10**-9)
        FILETYPE: either "Igor" or "Array"
        
        For Igor Pro files use 'Igor'. In this case V and I must contain path and filename of the file in which the data are stored.
        For numpy Array data use "Array". In this case V and I must be numpy arrays 
        """
    
        print "Set AEC trace..."
        
        trace_tmp = self._createTrace(FILETYPE, **kwargs)
        
        if type(trace_tmp) is list:
            trace_tmp = trace_tmp[0]
        
        self.AEC_trace = trace_tmp
        
        return trace_tmp
    
    
    def addTrainingSetTrace(self, FILETYPE='Axon', **kwargs):
    
        """
        Add training set trace to experiment.
        
        V : recorded voltage (either file address or numpy array depending on FILETYPE)
        V_units : units in which recorded voltage is stored (for mV use 10**-3)
        I : input current (either file address or numpy array depending on FILETYPE)
        I_units : units in which the input current is stored (for nA use 10**-9)
        FILETYPE: either "Igor" or "Array"
        
        For Igor Pro files use 'Igor'. In this case V and I must contain path and filename of the file in which the data are stored.
        For numpy Array data use "Array". In this case V and I must be numpy arrays 
        """
        
        print "Add Training Set trace..."
        trace_tmp = self._createTrace(FILETYPE, **kwargs)
        self.trainingset_traces.extend( trace_tmp )

        return trace_tmp


    def addTestSetTrace(self, FILETYPE='Axon', **kwargs):
        
        """
        Add test set trace to experiment.
        
        V : recorded voltage (either file address or numpy array depending on FILETYPE)
        V_units : units in which recorded voltage is stored (for mV use 10**-3)
        I : input current (either file address or numpy array depending on FILETYPE)
        I_units : units in which the input current is stored (for nA use 10**-9)
        FILETYPE: either "Igor" or "Array"
        
        For Igor Pro files use 'Igor'. In this case V and I must contain path and filename of the file in which the data are stored.
        For numpy Array data use "Array". In this case V and I must be numpy arrays 
        """
   
        print "Add Test Set trace..."
        trace_tmp = self._createTrace(FILETYPE, **kwargs)
        self.testset_traces.extend( trace_tmp )

        return trace_tmp
    
    

    ############################################################################################
    # FUNCTIONS ASSOCIATED WITH ACTIVE ELECTRODE COMPENSATION
    ############################################################################################    
    def setAEC(self, AEC):
        
        self.AEC = AEC


    def getAEC(self):
        
        return self.AEC    
             
             
    def performAEC(self):

        self.AEC.performAEC(self)
    
    
    ############################################################################################
    # FUNCTIONS FOR SAVING AND LOADING AN EXPERIMENT
    ############################################################################################
    def save(self, path):
        
        """
        Save experiment.
        """
        
        filename = path + "/Experiment_" + self.name + '.pkl'
        
        print "Saving: " + filename + "..."        
        f = open(filename,'w')
        pkl.dump(self, f)
        print "Done!"
        
        
    @classmethod
    def load(cls, filename):
        
        """
        Load experiment from file.
        """
        
        print "Load experiment: " + filename + "..."        
      
        f = open(filename,'r')
        expr = pkl.load(f)
    
        print "Done!" 
           
        return expr      
      
      
    ############################################################################################
    # EVALUATE PERFORMANCES OF A MODEL
    ############################################################################################         
    def predictSpikes(self, spiking_model, nb_rep=500):

        """
        Evaluate the predictive power of a spiking model in predicting the spike timing of the test traces.
        Since the spiking_model can be stochastic, the model is simulated several times.
        
        spiking_model : Spiking Model Object used to predict spiking activity
        np_rep: number of times the spiking model is stimulated to predict spikes
  
        Return a SpikeTrainComparator object that can be used to compute different performance metrics.
        """

        # Collect spike times in test set

        all_spks_times_testset = []

        for tr in self.testset_traces:
            
            if tr.useTrace :
                
                spks_times = tr.getSpikeTimes()
                all_spks_times_testset.append(spks_times)
    
    
        # Predict spike times using model
        
        T_test = self.testset_traces[0].T       # duration of the test set input current
        I_test = self.testset_traces[0].I       # test set current used in experimetns
        
        all_spks_times_prediction = []
        
        print "Predict spike times..."
        
        for rep in np.arange(nb_rep) :
            print "Progress: %2.1f %% \r" % (100*(rep+1)/nb_rep),
            spks_times = spiking_model.simulateSpikingResponse(I_test, self.dt)
            all_spks_times_prediction.append(spks_times)
        
        # Create SpikeTrainComparator object containing experimental and predicted spike times 
        
        prediction = SpikeTrainComparator(T_test, all_spks_times_testset, all_spks_times_prediction)
        
        return prediction
        

        
    ############################################################################################
    # AUXILIARY FUNCTIONS
    ############################################################################################            
    def detectSpikes_python(self, threshold=0.0, ref=3.0):

        """
        Extract spike times form all experimental traces.
        Python implementation (to speed up, use the function detectSpikes implemented in C).
        """

        print "Detect spikes!"
                
        self.spikeDetection_threshold = threshold   
        self.spikeDetection_ref = ref         

        if self.AEC_trace != 0 :
            self.AEC_trace.detectSpikes_python(self.spikeDetection_threshold, self.spikeDetection_ref)
        
        for tr in self.trainingset_traces :
            tr.detectSpikes_python(self.spikeDetection_threshold, self.spikeDetection_ref)           
            
        for tr in self.testset_traces :
            tr.detectSpikes_python(self.spikeDetection_threshold, self.spikeDetection_ref)         
        
        print "Done!"
        
        
    def detectSpikes(self, threshold=0.0, ref=3.0):

        """
        Extract spike times form all experimental traces.
        C implementation.
        """

        print "Detect spikes!"
                
        self.spikeDetection_threshold = threshold   
        self.spikeDetection_ref = ref         

        if self.AEC_trace != 0 :
            self.AEC_trace.detectSpikes(self.spikeDetection_threshold, self.spikeDetection_ref)
        
        for tr in self.trainingset_traces :
            tr.detectSpikes(self.spikeDetection_threshold, self.spikeDetection_ref)           
            
        for tr in self.testset_traces :
            tr.detectSpikes(self.spikeDetection_threshold, self.spikeDetection_ref)         
        
        print "Done!"
    
    
    def getTrainingSetNb(self):
        
        """
        Return the number of training set traces.
        According to the experimental protocol proposed in Pozzorini et al. PLOS Comp. Biol. there is only one training set trace,
        but this Toolbox can handle multiple training set traces.
        """
        
        return len(self.trainingset_traces) 


    def getTrainingSetNbOfSpikes(self):
        
        """
        Return the number of spikes in the training set data (only consider ROI)
        """
        
        nbSpksTot  = 0
        
        for tr in self.trainingset_traces :
            
            if tr.useTrace :
            
                nbSpksTot += tr.getSpikeNbInROI()
            
           
            
        return nbSpksTot
        
        
            

      
    ############################################################################################
    # FUNCTIONS FOR PLOTTING
    ############################################################################################
    def plotTrainingSet(self):
        
        plt.figure(figsize=(12,8), facecolor='white')
        
        cnt = 0
        
        for tr in self.trainingset_traces :
            
            # Plot input current
            plt.subplot(2*self.getTrainingSetNb(),1,cnt*2+1)
            plt.plot(tr.getTime(), tr.I, 'gray')

            # Plot ROI
            ROI_vector = -10.0*np.ones(int(tr.T/tr.dt)) 
            if tr.useTrace :
                ROI_vector[tr.getROI()] = 10.0
            
            plt.fill_between(tr.getTime(), ROI_vector, 10.0, color='0.2')
            
            plt.ylim([min(tr.I)-0.5, max(tr.I)+0.5])
            plt.ylabel("I (nA)")
            plt.xticks([])
            
            # Plot membrane potential    
            plt.subplot(2*self.getTrainingSetNb(),1,cnt*2+2)  
            plt.plot(tr.getTime(), tr.V_rec, 'black')    
            
            if tr.AEC_flag :
                plt.plot(tr.getTime(), tr.V, 'blue')    
                
                
            if tr.spks_flag :
                plt.plot(tr.getSpikeTimes(), np.zeros(tr.getSpikeNb()), '.', color='red')
            
            # Plot ROI
            ROI_vector = -100.0*np.ones(int(tr.T/tr.dt)) 
            if tr.useTrace :
                ROI_vector[tr.getROI()] = 100.0
            
            plt.fill_between(tr.getTime(), ROI_vector, 100.0, color='0.2')
            
            plt.ylim([min(tr.V)-5.0, max(tr.V)+5.0])
            plt.ylabel("Voltage (mV)")   
                  
            cnt +=1
        
        plt.xlabel("Time (ms)")
        
        plt.subplot(2*self.getTrainingSetNb(),1,1)
        plt.title('Experiment ' + self.name + " - Training Set (dark region not selected)")
        plt.subplots_adjust(left=0.10, bottom=0.07, right=0.95, top=0.92, wspace=0.25, hspace=0.25)

        plt.show()

        
    def plotTestSet(self):
        
        plt.figure(figsize=(12,6), facecolor='white')
        
        # Plot  test set currents 
        plt.subplot(3,1,1)
       
        for tr in self.testset_traces :         
            plt.plot(tr.getTime(), tr.I, 'gray')
        plt.ylabel("I (nA)")
        plt.title('Experiment ' + self.name + " - Test Set")
        # Plot  test set voltage        
        plt.subplot(3,1,2)
        for tr in self.testset_traces :          
            plt.plot(tr.getTime(), tr.V, 'black')
        plt.ylabel("Voltage (mV)")

        # Plot test set raster
        plt.subplot(3,1,3)
        
        cnt = 0
        for tr in self.testset_traces :
            cnt += 1      
            if tr.spks_flag :
                plt.plot(tr.getSpikeTimes(), cnt*np.ones(tr.getSpikeNb()), '|', color='black', ms=5, mew=2)
        
        plt.yticks([])
        plt.ylim([0, cnt+1])
        plt.xlabel("Time (ms)")
        
        plt.subplots_adjust(left=0.10, bottom=0.07, right=0.95, top=0.92, wspace=0.25, hspace=0.25)

        plt.show()
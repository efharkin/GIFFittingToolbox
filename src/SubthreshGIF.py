import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

import weave
from numpy.linalg import inv

from GIF import *
from Filter_Rect_LogSpaced import *

from Tools import reprint
from numpy import nan, NaN

import math


class SubthreshGIF(GIF) :

    """
    Generalized Integrate and Fire model defined in Pozzorini et al. PLOS Comp. Biol. 2015
    
    Spike are produced stochastically with firing intensity:
    
    lambda(t) = lambda0 * exp( (V(t)-V_T(t))/DV ),
    
    where the membrane potential dynamics is given by:
    
    C dV/dt = -gl(V-El) + I - sum_j eta(t-\hat t_j)
    
    and the firing threshold V_T is given by:
    
    V_T = Vt_star + sum_j gamma(t-\hat t_j)
    
    and \hat t_j denote the spike times.
    """

    def __init__(self, dt=0.1):
                   
        self.dt = dt                    # dt used in simulations (eta and gamma are interpolated according to this value)
  
        # Define model parameters
        
        self.gl      = 1.0/100.0        # nS, leak conductance
        self.C       = 20.0*self.gl     # nF, capacitance
        self.El      = -65.0            # mV, reversal potential
        
        
    
    

    
    ########################################################################################################
    # IMPLEMENT ABSTRACT METHODS OF Spiking model
    ########################################################################################################
    
    
    def simulateSpikingResponse(self, I, dt):
        
        """
        Subthreshold model does not spike.
        """
        
        raise RuntimeError('Subthreshold model does not spike.')


    ########################################################################################################
    # IMPLEMENT ABSTRACT METHODS OF Threshold Model
    ########################################################################################################
    
    
    def simulateVoltageResponse(self, I, dt) :

        self.setDt(dt)
    
        (time, V, eta_sum, V_T, spks_times) = self.simulate(I, self.El)
        
        return (spks_times, V, V_T)


    ########################################################################################################
    # METHODS FOR NUMERICAL SIMULATIONS
    ########################################################################################################  
      
    def simulate(self, I, V0):
 
        """
        Simulate the spiking response of the GIF model to an input current I (nA) with time step dt.
        V0 indicate the initial condition V(0)=V0.
        The function returns:
        - time     : ms, support for V, eta_sum, V_T, spks
        - V        : mV, membrane potential
        """
 
        # Input parameters
        p_T         = len(I)
        p_dt        = self.dt
        
        # Model parameters
        p_gl        = self.gl
        p_C         = self.C 
        p_El        = self.El
      
        # Define arrays
        V = np.array(np.zeros(p_T), dtype="double")
        I = np.array(I, dtype="double")
 
        # Set initial condition
        V[0] = V0
         
        code =  """
                #include <math.h>
                
                int   T_ind      = int(p_T);                
                float dt         = float(p_dt); 
                
                float gl         = float(p_gl);
                float C          = float(p_C);
                float El         = float(p_El);
                
                                                
                for (int t=0; t<T_ind-1; t++) {
    
    
                    // INTEGRATE VOLTAGE
                    V[t+1] = V[t] + dt/C*( -gl*(V[t] - El) + I[t] );
               
               
                }
                
                """
 
        vars = [ 'p_T','p_dt','p_gl','p_C','p_El','V','I' ]
        
        v = weave.inline(code, vars)

        time = np.arange(p_T)*self.dt
        
        return (time, V)

        
    def simulateDeterministic_forceSpikes(self, *args):
        
        """
        Subthreshold model does not spike.
        """
 
        raise RuntimeError('Subthreshold model does not spike.')
        

           
    ########################################################################################################
    # METHODS FOR MODEL FITTING
    ########################################################################################################  
      
         
    def fit(self, experiment, DT_beforeSpike = 5.0):
        
        """
        Fit the GIF model on experimental data.
        The experimental data are stored in the object experiment provided as an input.
        The parameter DT_beforeSpike (in ms) defines the region that is cut before each spike when fitting the subthreshold dynamics of the membrane potential.
        Only training set traces in experiment are used to perform the fit.
        """
        
        # Three step procedure used for parameters extraction 
        
        print "\n################################"
        print "# Fit GIF"
        print "################################\n"
        
        
        self.fitSubthresholdDynamics(experiment)



    ########################################################################################################
    # FIT VOLTAGE RESET GIVEN ABSOLUTE REFRACOTORY PERIOD (step 1)
    ########################################################################################################


    def fitVoltageReset(self, experiment, Tref, do_plot=False):
        
        """
        Subthreshold model does not spike.
        """
        
        raise RuntimeError('Subthreshold model does not spike.')
        
        


    ########################################################################################################
    # FUNCTIONS RELATED TO FIT OF SUBTHRESHOLD DYNAMICS (step 2)
    ########################################################################################################


    def fitSubthresholdDynamics(self, experiment):
          
        """
        Implement Step 2 of the fitting procedure introduced in Pozzorini et al. PLOS Comb. Biol. 2015
        The voltage reset is estimated by computing the spike-triggered average of the voltage.
        experiment: Experiment object on which the model is fitted.
        DT_beforeSpike: in ms, data right before spikes are excluded from the fit. This parameter can be used to define that time interval.
        """  
                  
        print "\nGIF MODEL - Fit subthreshold dynamics..." 
            
        # Expand eta in basis functions
        self.dt = experiment.dt
        
        
        # Build X matrix and Y vector to perform linear regression (use all traces in training set)    
        # For each training set an X matrix and a Y vector is built.   
        ####################################################################################################
        X = []
        Y = []
    
        cnt = 0
        
        for tr in experiment.trainingset_traces :
        
            if tr.useTrace :
        
                cnt += 1
                reprint( "Compute X matrix for repetition %d" % (cnt) )          
                
                # Compute the the X matrix and Y=\dot_V_data vector used to perform the multilinear linear regression (see Eq. 17.18 in Pozzorini et al. PLOS Comp. Biol. 2015)
                (X_tmp, Y_tmp) = self.fitSubthresholdDynamics_Build_Xmatrix_Yvector(tr)
     
                X.append(X_tmp)
                Y.append(Y_tmp)
    
    
        # Concatenate matrixes associated with different traces to perform a single multilinear regression
        ####################################################################################################
        if cnt == 1:
            X = X[0]
            Y = Y[0]
            
        elif cnt > 1:
            X = np.concatenate(X, axis=0)
            Y = np.concatenate(Y, axis=0)
        
        else :
            print "\nError, at least one training set trace should be selected to perform fit."
        
        
        # Perform linear Regression defined in Eq. 17 of Pozzorini et al. PLOS Comp. Biol. 2015
        ####################################################################################################
        
        print "\nPerform linear regression..."
        XTX     = np.dot(np.transpose(X), X)
        XTX_inv = inv(XTX)
        XTY     = np.dot(np.transpose(X), Y)
        b       = np.dot(XTX_inv, XTY)
        b       = b.flatten()
   
   
        # Extract explicit model parameters from regression result b
        ####################################################################################################

        self.C  = 1./b[1]
        self.gl = -b[0]*self.C
        self.El = b[2]*self.C/self.gl
    
    
        self.printParameters()   
        
        
        # Compute percentage of variance explained on dV/dt
        ####################################################################################################

        var_explained_dV = 1.0 - np.mean((Y - np.dot(X,b))**2)/np.var(Y)
        print "Percentage of variance explained (on dV/dt): %0.2f" % (var_explained_dV*100.0)

        
        # Compute percentage of variance explained on V (see Eq. 26 in Pozzorini et al. PLOS Comp. Biol. 2105)
        ####################################################################################################

        SSE = 0     # sum of squared errors
        VAR = 0     # variance of data
        
        for tr in experiment.trainingset_traces :
        
            if tr.useTrace :

                # Simulate subthreshold dynamics 
                (time, V_est) = self.simulate(tr.I, tr.V[0])
                
                indices_tmp = tr.getROI()
                
                SSE += sum((V_est[indices_tmp] - tr.V[indices_tmp])**2)
                VAR += len(indices_tmp)*np.var(tr.V[indices_tmp])
                
        var_explained_V = 1.0 - SSE / VAR
        
        print "Percentage of variance explained (on V): %0.2f" % (var_explained_V*100.0)
                
                    
    def fitSubthresholdDynamics_Build_Xmatrix_Yvector(self, trace):
           
        """
        Compute the X matrix and the Y vector (i.e. \dot_V_data) used to perfomr the linear regression 
        defined in Eq. 17-18 of Pozzorini et al. 2015 for an individual experimental trace provided as parameter.
        The input parameter trace is an ojbect of class Trace.
        """
                
        # Length of the voltage trace       
        Tref_ind = int(self.Tref/trace.dt)
        
        
        # Select region where to perform linear regression (specified in the ROI of individual taces)
        ####################################################################################################
        selection = trace.getROI()
        selection_l = len(selection)
        
        
        # Build X matrix for linear regression (see Eq. 18 in Pozzorini et al. PLOS Comp. Biol. 2015)
        ####################################################################################################
        X = np.zeros( (selection_l, 3) )
        
        # Fill first two columns of X matrix        
        X[:,0] = trace.V[selection]
        X[:,1] = trace.I[selection]
        X[:,2] = np.ones(selection_l) 
        

        # Build Y vector (voltage derivative \dot_V_data)    
        ####################################################################################################
        Y = np.array( np.concatenate( (np.diff(trace.V)/trace.dt, [0]) ) )[selection]      

        return (X, Y)
        
        
        
    ########################################################################################################
    # FUNCTIONS RELATED TO FIT FIRING THRESHOLD PARAMETERS (step 3)
    ########################################################################################################        
 
         
    def fitStaticThreshold(self, *args):
        
        """
        Subthreshold models do not spike.
        """
        
        raise RuntimeError('Subthreshold models do not spike.')

   
    def fitThresholdDynamics(self, *args):
                  
        """
        Subthreshold models do not spike.
        """
        
        raise RuntimeError('Subthreshold models do not spike.')
          
      
    def maximizeLikelihood(self, *args) :
    
        """
        Subthreshold models do not spike.
        """
        
        raise RuntimeError('Subthreshold models do not spike.')
     
        
    def computeLikelihoodGradientHessian(self, *args) : 
        
        """
        Subthreshold models do not spike.
        """
        
        raise RuntimeError('Subthreshold models do not spike.')


    def buildXmatrix_staticThreshold(self, *args) :

        """
        Subthreshold models do not spike.
        """
        
        raise RuntimeError('Subthreshold models do not spike.')
        
            
    def buildXmatrix_dynamicThreshold(self, *args) :

        """
        Subthreshold models do not spike.
        """
        
        raise RuntimeError('Subthreshold models do not spike.')
 
 
        
    ########################################################################################################
    # PLOT AND PRINT FUNCTIONS
    ########################################################################################################     
        
        
    def plotParameters(self) :
        
        """
        Generate figure with model filters.
        """
        
        plt.figure(facecolor='white', figsize=(14,4))
            
        # Plot kappa
        plt.subplot(1,1,1)
        
        K_support = np.linspace(0,150.0, 300)             
        K = 1./self.C*np.exp(-K_support/(self.C/self.gl)) 
            
        plt.plot(K_support, K, color='red', lw=2)
        plt.plot([K_support[0], K_support[-1]], [0,0], ls=':', color='black', lw=2)
            
        plt.xlim([K_support[0], K_support[-1]])    
        plt.xlabel("Time (ms)")
        plt.ylabel("Membrane filter (MOhm/ms)")        
        
        plt.subplots_adjust(left=0.05, bottom=0.15, right=0.95, top=0.92, wspace=0.35, hspace=0.25)

        plt.show()
      
      
    def printParameters(self):

        """
        Print model parameters on terminal.
        """

        print "\n-------------------------"        
        print "GIF model parameters:"
        print "-------------------------"
        print "tau_m (ms):\t%0.3f"  % (self.C/self.gl)
        print "R (MOhm):\t%0.3f"    % (1.0/self.gl)
        print "C (nF):\t\t%0.3f"    % (self.C)
        print "gl (nS):\t%0.6f"     % (self.gl)
        print "El (mV):\t%0.3f"     % (self.El)         
        print "-------------------------\n"
                  

    @classmethod
    def compareModels(cls, GIFs, labels=None):

        """
        Given a list of GIF models, GIFs, the function produce a plot in which the model parameters are compared.
        """

        # PRINT PARAMETERS        

        print "\n#####################################"
        print "GIF model comparison"
        print "#####################################\n"
        
        cnt = 0
        for GIF in GIFs :
            
            #print "Model: " + labels[cnt]          
            GIF.printParameters()
            cnt+=1

        print "#####################################\n"                
                
        # PLOT PARAMETERS
        plt.figure(facecolor='white', figsize=(9,8)) 
               
        colors = plt.cm.jet( np.linspace(0.7, 1.0, len(GIFs) ) )   
        
        # Membrane filter
        plt.subplot(2,2,1)
            
        cnt = 0
        for GIF in GIFs :
            
            K_support = np.linspace(0,150.0, 1500)             
            K = 1./GIF.C*np.exp(-K_support/(GIF.C/GIF.gl))     
            plt.plot(K_support, K, color=colors[cnt], lw=2)
            cnt += 1
            
        plt.plot([K_support[0], K_support[-1]], [0,0], ls=':', color='black', lw=2, zorder=-1)   
                
        plt.xlim([K_support[0], K_support[-1]])
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane filter (MOhm/ms)')  


        # Spike triggered current
        plt.subplot(2,2,2)
            
        cnt = 0
        for GIF in GIFs :
            
            if labels == None :
                label_tmp =""
            else :
                label_tmp = labels[cnt]
            
            (eta_support, eta) = GIF.eta.getInterpolatedFilter(0.1)         
            plt.plot(eta_support, eta, color=colors[cnt], lw=2, label=label_tmp)
            cnt += 1
            
        plt.plot([eta_support[0], eta_support[-1]], [0,0], ls=':', color='black', lw=2, zorder=-1)   
        
        if labels != None :
            plt.legend()       
            
        
        plt.xlim([eta_support[0], eta_support[-1]])
        plt.xlabel('Time (ms)')
        plt.ylabel('Eta (nA)')        
        

        # Escape rate
        plt.subplot(2,2,3)
            
        cnt = 0
        for GIF in GIFs :
            
            V_support = np.linspace(GIF.Vt_star-5*GIF.DV,GIF.Vt_star+10*GIF.DV, 1000) 
            escape_rate = GIF.lambda0*np.exp((V_support-GIF.Vt_star)/GIF.DV)                
            plt.plot(V_support, escape_rate, color=colors[cnt], lw=2)
            cnt += 1
          
        plt.ylim([0, 100])    
        plt.plot([V_support[0], V_support[-1]], [0,0], ls=':', color='black', lw=2, zorder=-1)   
    
        plt.xlim([V_support[0], V_support[-1]])
        plt.xlabel('Membrane potential (mV)')
        plt.ylabel('Escape rate (Hz)')  


        # Spike triggered threshold movememnt
        plt.subplot(2,2,4)
            
        cnt = 0
        for GIF in GIFs :
            
            (gamma_support, gamma) = GIF.gamma.getInterpolatedFilter(0.1)         
            plt.plot(gamma_support, gamma, color=colors[cnt], lw=2)
            cnt += 1
            
        plt.plot([gamma_support[0], gamma_support[-1]], [0,0], ls=':', color='black', lw=2, zorder=-1)   
      
        plt.xlim([gamma_support[0]+0.1, gamma_support[-1]])
        plt.ylim([-100,100])
        plt.xlabel('Time (ms)')
        plt.ylabel('Gamma (mV)')   

        plt.subplots_adjust(left=0.08, bottom=0.10, right=0.95, top=0.93, wspace=0.25, hspace=0.25)
        
        plt.show()
    
 
    @classmethod
    def plotAverageModel(cls, GIFs):

        """
        Average model parameters and plot summary data.
        """
                   
        #######################################################################################################
        # PLOT PARAMETERS
        #######################################################################################################        
        
        fig = plt.figure(facecolor='white', figsize=(16,7))  
        fig.subplots_adjust(left=0.07, bottom=0.08, right=0.95, top=0.90, wspace=0.35, hspace=0.5)   
        rcParams['xtick.direction'] = 'out'
        rcParams['ytick.direction'] = 'out'
       
       
        # MEMBRANE FILTER
        #######################################################################################################
        
        plt.subplot(2,4,1)
                    
        K_all = []
        
        for GIF in GIFs :
                      
            K_support = np.linspace(0,150.0, 300)             
            K = 1./GIF.C*np.exp(-K_support/(GIF.C/GIF.gl))     
            plt.plot(K_support, K, color='0.3', lw=1, zorder=5)
            
            K_all.append(K)

        K_mean = np.mean(K_all, axis=0)
        K_std  = np.std(K_all, axis=0)
        
        plt.fill_between(K_support, K_mean+K_std,y2=K_mean-K_std, color='gray', zorder=0)
        plt.plot(K_support, np.mean(K_all, axis=0), color='red', lw=2, zorder=10)  
        plt.plot([K_support[0], K_support[-1]], [0,0], ls=':', color='black', lw=2, zorder=-1)   
                
        Tools.removeAxis(plt.gca(), ['top', 'right'])
        plt.xlim([K_support[0], K_support[-1]])
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane filter (MOhm/ms)')  

       
        # SPIKE-TRIGGERED CURRENT
        #######################################################################################################
    
        plt.subplot(2,4,2)
                    
        K_all = []
        
        for GIF in GIFs :
                
            (K_support, K) = GIF.eta.getInterpolatedFilter(0.1)      
   
            plt.plot(K_support, K, color='0.3', lw=1, zorder=5)
            
            K_all.append(K)

        K_mean = np.mean(K_all, axis=0)
        K_std  = np.std(K_all, axis=0)
        
        plt.fill_between(K_support, K_mean+K_std,y2=K_mean-K_std, color='gray', zorder=0)
        plt.plot(K_support, np.mean(K_all, axis=0), color='red', lw=2, zorder=10)  
        plt.plot([K_support[0], K_support[-1]], [0,0], ls=':', color='black', lw=2, zorder=-1)   
                
        Tools.removeAxis(plt.gca(), ['top', 'right'])
        plt.xlim([K_support[0], K_support[-1]/10.0])
        plt.xlabel('Time (ms)')
        plt.ylabel('Spike-triggered current (nA)')  
 
 
        # SPIKE-TRIGGERED MOVEMENT OF THE FIRING THRESHOLD
        #######################################################################################################
    
        plt.subplot(2,4,3)
                    
        K_all = []
        
        for GIF in GIFs :
                
            (K_support, K) = GIF.gamma.getInterpolatedFilter(0.1)      
   
            plt.plot(K_support, K, color='0.3', lw=1, zorder=5)
            
            K_all.append(K)

        K_mean = np.mean(K_all, axis=0)
        K_std  = np.std(K_all, axis=0)
        
        plt.fill_between(K_support, K_mean+K_std,y2=K_mean-K_std, color='gray', zorder=0)
        plt.plot(K_support, np.mean(K_all, axis=0), color='red', lw=2, zorder=10)   
        plt.plot([K_support[0], K_support[-1]], [0,0], ls=':', color='black', lw=2, zorder=-1)   
                
        plt.xlim([K_support[0], K_support[-1]])
        Tools.removeAxis(plt.gca(), ['top', 'right'])
        plt.xlabel('Time (ms)')
        plt.ylabel('Spike-triggered threshold (mV)')  
 
      
        # R
        #######################################################################################################
    
        plt.subplot(4,6,12+1)
 
        p_all = []
        for GIF in GIFs :
                
            p = 1./GIF.gl
            p_all.append(p)
            
        plt.hist(p_all, histtype='bar', color='red', ec='white', lw=2)
        plt.xlabel('R (MOhm)')
        Tools.removeAxis(plt.gca(), ['top', 'left', 'right'])
        plt.yticks([])        
        
        
        # tau_m
        #######################################################################################################
    
        plt.subplot(4,6,18+1)
 
        p_all = []
        for GIF in GIFs :
                
            p = GIF.C/GIF.gl
            p_all.append(p)
            
        plt.hist(p_all, histtype='bar', color='red', ec='white', lw=2)
        plt.xlabel('tau_m (ms)')
        Tools.removeAxis(plt.gca(), ['top', 'left', 'right'])
        plt.yticks([])     
       
   
        # El
        #######################################################################################################
    
        plt.subplot(4,6,12+2)
 
        p_all = []
        for GIF in GIFs :
                
            p = GIF.El
            p_all.append(p)
            
        plt.hist(p_all, histtype='bar', color='red', ec='white', lw=2)
        plt.xlabel('El (mV)')
        Tools.removeAxis(plt.gca(), ['top', 'left', 'right'])
        plt.yticks([])     
       
          
        # V reset
        #######################################################################################################
    
        plt.subplot(4,6,18+2)
 
        p_all = []
        for GIF in GIFs :
                
            p = GIF.Vr
            p_all.append(p)
        
        print "Mean Vr (mV): %0.1f" % (np.mean(p_all))  
        
        plt.hist(p_all, histtype='bar', color='red', ec='white', lw=2)
        plt.xlabel('Vr (mV)')
        Tools.removeAxis(plt.gca(), ['top', 'left', 'right'])
        plt.yticks([])     
        
        
        # Vt*
        #######################################################################################################
    
        plt.subplot(4,6,12+3)
 
        p_all = []
        for GIF in GIFs :
                
            p = GIF.Vt_star
            p_all.append(p)
            
        plt.hist(p_all, histtype='bar', color='red', ec='white', lw=2)
        plt.xlabel('Vt_star (mV)')
        Tools.removeAxis(plt.gca(), ['top', 'left', 'right'])
        plt.yticks([])    
        
        # Vt*
        #######################################################################################################
    
        plt.subplot(4,6,18+3)
 
        p_all = []
        for GIF in GIFs :
                
            p = GIF.DV
            p_all.append(p)
            
        plt.hist(p_all, histtype='bar', color='red', ec='white', lw=2)
        plt.xlabel('DV (mV)')
        Tools.removeAxis(plt.gca(), ['top', 'left', 'right'])
        plt.yticks([])    

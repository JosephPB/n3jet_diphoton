import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import Locator, MaxNLocator

class RivetPlotter:
    """
    Extract and plot data from Rivet analysis .dat file
    """
    
    def __init__(self, dat_file):
        """
        Parameters
        ----------
        dat_file: path to data file
        """
        
        self.dat_file = dat_file
        
    def extract_data(self, **kwargs):
        """
        Extract histogram data from a data file
        
        Assumptions
        -----------
        Two histograms per data file
        
        Returns
        -------
        ScaledBy value for first histogram
        ScaledBy value for second histogram
        data for first histogram as a 2D list
        data for second histogram as a 2D list
        """
        
        dat_file = kwargs.get("dat_file", self.dat_file)
        
        scales = []
        njet_data = []
        nn_data = []
        
        with open(dat_file, 'r') as f:
            line = f.readline()
            extracting_data = False
            data_extracting = 0
            while line:
                line = f.readline()
                if line.startswith('ScaledBy'):
                    scales.append(float(line.split('=')[-1]))
                if line.startswith('# xlow'):
                    extracting_data = True
                if line.startswith('# END HIST'):
                    extracting_data = False
                    data_extracting +=1
                if extracting_data:
                    if data_extracting == 0:
                        njet_data.append(line.split('\t'))
                    elif data_extracting == 1:
                        nn_data.append(line.split('\t'))    
                    else:
                        raise ValueError(
                            'data_extracting is now at {} but should only be 0 or 1'.format(
                                data_extracting
                            )
                        )
            
        return scales[0], scales[1], njet_data, nn_data
    
    def parse_data_step(self, data):
        """
        Parse data extracted from a data file
        
        Parameters
        ----------
        data: 2D list of histogram data with first element being ignored (titles)
              columns:
                  0: bin lower bounds
                  1: bin upper bounds
                  2: bin values
                  3: +/- bin errors (assume symmetric)
                  
        Returns
        -------
        bins: array of bin limits in np.histogram format (i.e. including lower and upper bound)
        vals: bin values with 0'th element duplicated
        errs: bin +/- errors with 0'th element duplicated
        """
        
        data = np.array(data)
        xlow = data[:,0][1:].astype('float')
        xhigh = data[:,1][1:].astype('float')
        vals = data[:,2][1:].astype('float')
        errs = data[:,3][1:].astype('float')
        
        bins = np.append(xlow,xhigh[-1])
        vals = np.append(vals[0], vals)
        errs = np.append(errs[0], errs)/vals
        return bins, vals, errs

    def plot_distirbution(
        self,
        njet_bins,
        nn_bins,
        njet_vals,
        nn_vals,
        njet_errs,
        nn_errs,
        ylabel,
        xlabel
    ):
        """
        Plot two histograms with ratio plot below
        
        Returns
        -------
        matplotlib figure environment
        """
        
        fig = plt.figure(1)
        
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        
        ax1 = fig.add_axes((.1,.4,.8,.8))
        
        ax1.step(njet_bins, njet_vals, color='red')
        ax1.step(nn_bins, nn_vals, color='blue')
        ax1.set_yscale('log')
        ax1.set_xticklabels([])
        
        ax1.tick_params(axis='x', labelsize=15, direction='in', top=True)
        ax1.tick_params(axis='y', labelsize=15, direction='in', right=True, which='both')
        
        ax1.set_xlim((0,njet_bins[-1]))
        
        ax1.set_ylabel(r'{}'.format(ylabel), fontsize=17, labelpad=10)
        
        ax2=fig.add_axes((.1,.1,.8,.3))
        
        ax2.step(njet_bins, np.ones(len(njet_vals)), color='red')
        ax2.step(njet_bins, nn_vals/njet_vals, color='blue')
        ax2.fill_between(
            njet_bins[:-1], 
            (np.ones(len(njet_vals))-njet_errs)[1:], 
            (np.ones(len(njet_vals))+njet_errs)[1:], 
            step='post', color='red', alpha=0.5
        )
        ax2.fill_between(
            nn_bins[:-1], 
            ((nn_vals/njet_vals)-nn_errs)[1:], 
            ((nn_vals/njet_vals)+nn_errs)[1:], 
            step='post', color='blue', alpha=0.5
        )
        
        ax2.set_xlim((0,njet_bins[-1]))
        ax2.set_ylim((0.4,1.6))
        
        ax2.set_ylabel(r'Ratio', fontsize=17, labelpad=10)
        ax2.set_xlabel(r'{}'.format(xlabel), fontsize=17, labelpad=10)
        ax2.tick_params(axis='x', labelsize=15, direction = 'in', top = True)
        ax2.tick_params(axis='y', labelsize=15, direction = 'in', right = True, which='both')
        
        return fig
    
    def plot(self, xlabel, ylabel):
        
        scales[0], scales[1], njet_data, nn_data = self.extract_data()
        njet_bins, njet_vals, njet_errs = self.parse_data_step(njet_data)
        nn_bins, nn_vals, nn_errs = self.parse_data_step(nn_data)
        fig = self.plot_distirbution(
            njet_bins = njet_bins,
            nn_bins = nn_bins,
            njet_vals = njet_vals,
            nn_vals = nn_vals,
            njet_errs = njet_errs,
            nn_errs = nn_errs,
            ylabel = ylabel,
            xlabel = xlabel
        )
        
        return fig
        
        
        
        
        
        
        
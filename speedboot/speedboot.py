import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm

class speedboot:
    """
    Speed boostrap class.

    Attributes:
      data (numpy array or pandas dataframe): fitted model for treatment outcome.
      stats_fun (function): function that takes data as input and outputs one or multiple statistics in a numpy array.
    """                               
    
    def __init__(self, data, stats_fun):
        """
        Initializer for Fast boostrap class.
        """
        self.data = pd.DataFrame(data)
        self.stats_fun = stats_fun

    def fit(self, R=999, bar = True, par = True, seed=0):
        """Bootstrap multiple statistics at once given data.
        
        Attributes:
            R (positive int): fitted model for treatment outcome.
            bar (bool): print progress bar.
            par (bool): parallelization on all cores.
            seed (positive int): random seed for reproducibility.
        """
        
        random.seed(seed)
        num_cores = multiprocessing.cpu_count()
        resamples = [[random.randint(0,len(self.data)-1) for _ in range(len(self.data))] for _ in range(R)]
        boot_dfs = [self.data.iloc[resamples[i]] for i in range(R)]
        if bar and par:
            boot_estimates = Parallel(n_jobs=num_cores)(delayed(self.stats_fun)(i) for i in tqdm(boot_dfs))
        elif bar and not par:
            boot_estimates = [self.stats_fun(boot_df) for boot_df in tqdm(boot_dfs)]
        elif not bar and par:
            boot_estimates = Parallel(n_jobs=num_cores)(delayed(self.stats_fun)(i) for i in boot_dfs)
        elif not bar and not par:
            boot_estimates = [self.stats_fun(boot_df) for boot_df in boot_dfs]  
        self.ests_boot = np.vstack([ests_i.T for ests_i in boot_estimates])
        self.ests = np.array(self.stats_fun(self.data))
    
    def emp_ci(self, risk_a=.05):
        """from an array of estimates and a R x len(slef.ests) matrix of bootstrap estimates
        outputs a len(slef.ests) x 2 matrix of empirical bootstrap CI.
        
        Attributes:
            risk_a (probability float): alpha risk that determines confidence interval width i.e., risk_a=.05 for 95% confidence intervals.
        """
        quantiles = np.array([np.nanquantile(self.ests_boot[:,est_id],[1-risk_a/2, risk_a/2]) for est_id, _ in enumerate(self.ests)])
        if len(self.ests) == 1:
            return 2*self.ests-quantiles
        else:
            return np.multiply(2,self.ests).reshape(len(self.ests),1)-quantiles

    def per_ci(self, risk_a=.05):
        """from an array of estimates and a R x len(slef.ests) matrix of bootstrap estimates
        outputs a len(slef.ests) x 2 matrix of empirical bootstrap CI.
                
        Attributes:
            risk_a (probability float): alpha risk that determines confidence interval width i.e., risk_a=.05 for 95% confidence intervals.
        """
        return np.array([np.nanquantile(self.ests_boot[:,est_id],[risk_a/2, 1-risk_a/2]) for est_id, _ in enumerate(self.ests)])
    
    def plot(self, prec=.05, size=4):
        """plots histograms of the bootstrap estimates
        
        Attributes:
            prec (probability float): determines the binwidth of the histograms. If prec=1 plots as many bins as boostrap estimates.
            size (positive float): size of each plot.
        """
        plt.figure(figsize=(size, size*len(self.ests)))
        n_bins = int(prec * len(self.ests_boot))
        for n in range(len(self.ests)): # adds a new subplot iteratively
            ax = plt.subplot(len(self.ests), 1, n + 1)
            ax.hist(self.ests_boot[:,n], bins=np.linspace(min(self.ests_boot[:,n]), max(self.ests_boot[:,n]), num=n_bins))
            plt.xlim(min(self.ests_boot[:,n]), max(self.ests_boot[:,n]))
            ax.set_title("Estimate " + str(n+1))
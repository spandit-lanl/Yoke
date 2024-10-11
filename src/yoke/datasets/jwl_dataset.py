"""Functions and classes for torch DataSets which sample the Cylinder Test/JWL data."""

####################################
# Packages
####################################
import typing
import numpy as np
from scipy.optimize import root 
import pandas as pd
import torch
from torch.utils.data import Dataset

def compute_CJ(A,B,C,G,R1,R2,v0):
    '''Return CJ state including edet from JWL parameters.
    Args: A,B,C in units of pressure
        G, R1, R2 are unitless
        v0 is in units of specific volume
    Returns: vj/v0 [unitless], pj [pressure], Dj = [velocity=sqrt(energy)], 
             edet [specific energy=pressure x specific volume]'''
    # ideal gas cp/cv constant descibing large expansion behavior
    k = G+1; kp1 = k+1

    # reference isentrope: ps(v/v0)
    ps = lambda vs: A*np.exp(-R1*vs)+B*np.exp(-R2*vs)+C/vs**k
    # adiabatic gamma on reference isentrope: gs(v/v0)
    gs = lambda vs: vs*(A*R1*np.exp(-R1*vs)+B*R2*np.exp(-R2*vs)+C*k/vs**kp1)/ps(vs)
    # at CJ: vj/v0 = gj/(gj+1) assuming p0=0
    fzero = lambda vs: gs(vs)*(vs-1)+vs
    vsj = root(fzero, .5, tol=1e-14).x[0]
    pj = ps(vsj)
    Dj = np.sqrt(pj*v0/(1-vsj))
    # detonation energy available for HE expansion work
    edet = v0*vsj/G*(pj
                    +A*(G/R1/vsj-1)*np.exp(-R1*vsj)
                    +B*(G/R2/vsj-1)*np.exp(-R2*vsj))+pj/2*v0*(vsj-1)
    
    return vsj, pj, Dj, edet

def compute_e_release(A,B,C,G,R1,R2,v0,edet,vs):
    '''Return the detonation energy released at expansion vs=v/v0 
    Args: A,B,C in units of pressure
        G, R1, R2 are unitless
        v0 is in units of specific volume
        edet is the total detonation energy available in units of specific energy=pressure x specific volume 
        vs=v/v0 is the number of initial volumes of expansion
    Returns: detonation energy released in units of specific energy'''
    I = lambda vs: v0*(A/R1*np.exp(-R1*vs)+B/R2*np.exp(-R2*vs)+C/G/vs**G)
    return edet - I(vs)

####################################
# DataSet Classes
####################################

class CYLEX_pdv2jwl_Dataset(Dataset):
    def __init__(self, rng: slice, file: str):
        """The definition of a dataset object for the *CYLEX/JWL* data
        which produces PDV data with corresponding JWL parameters.
        The JWL reference isentrope is given by 
        ps(v/V0) = a e^{-r1 v/V0} + b e^{-r2 v/V0} + c (v/V0)^{-w-1}
        The parameter units are: [a] = [b] = [c] = GPa 
                                 [w] = [r1] = [r2] = -
                                 [V0] = cc/g
        The unit system is {GPa, mm, mus} so that velocity is in km/s and energy is in kJ/g

        Args:
            rng (slice): a slice object (start,stop,step) used to sample the data in *file*.
            file (str): .csv file with recorderd data with header
                         [a, b, c, w, r1, r2, V0, 
                         dcj, pcj, vcj, edet, e1, e2, e3, e4, e5, e6, e7,
                         t0.1, t0.15, t0.25, t0.35, t0.5, t0.75, t1, t1.5, t2, t2.5, t3.5, t4.5]

        """
        # Model Arguments
        self.file = file

        df = pd.read_csv(file, sep=",", header=0, engine="python")
        self.df = df = df.iloc[rng]

        self.Nsamples = N = len(df)
        self.check()

    def check(self):
        for i, row in self.df.iterrows():
            jwls =row['a':'V0']
            chks = row['dcj':'edet']
            es = row['e1':'e7']

            #print('---Sample ' + str(i) + '---')

            vsj, pj, Dj, edet = compute_CJ(*jwls.values)
            #print(pd.concat({'Andrew': pd.Series({'dcj': Dj, 'pcj': pj, 'vcj': v0*vsj, 'edet': edet}),'Chris': chks},axis=1))

            for i in es.index:
                vs = float(i[1:])
                #print(es[i], compute_e_release(*jwls.values,edet,vs))

    def __len__(self):
        """Return number of samples in dataset."""
        return self.Nsamples

    def __getitem__(self, index):
        """Return a tuple of a batch's input and output data for training at a given
        index.

        """
        # Get the PDV input
        ts = self.df.iloc[index]['t0.1':'t4.5']
        # Get the JWL parameter output
        jwls = self.df.iloc[index]['a':'r2']
        
        input = np.array(ts.values)
        output = jwls.values

        return input, output

class CYLEXnorm_pdv2jwl_Dataset(Dataset):
    def __init__(self, rng: slice, file: str):
        """The definition of a dataset object for the *CYLEX/JWL* data
        which produces PDV data with corresponding JWL parameters.
        The JWL reference isentrope is given by 
        ps(v/V0) = a e^{-r1 v/V0} + b e^{-r2 v/V0} + c (v/V0)^{-w-1}
        The parameter units are: [a] = [b] = [c] = GPa 
                                 [w] = [r1] = [r2] = -
                                 [V0] = cc/g
        The unit system is {GPa, mm, mus} so that velocity is in km/s and energy is in kJ/g

        Args:
            rng (slice): a slice object (start,stop,step) used to sample the data in *file*.
            file (str): .csv file with recorderd data with header
                         [a, b, c, w, r1, r2, V0, 
                         dcj, pcj, vcj, edet, e1, e2, e3, e4, e5, e6, e7,
                         t0.1, t0.15, t0.25, t0.35, t0.5, t0.75, t1, t1.5, t2, t2.5, t3.5, t4.5]

        """
        # Model Arguments
        self.file = file

        df = pd.read_csv(file, sep=",", header=0, engine="python")
        self.df = df = df.iloc[rng]
        self.stats = df.describe()

        self.Nsamples = N = len(df)
        self.check()

    def check(self):
        for i, row in self.df.iterrows():
            jwls =row['a':'V0']
            chks = row['dcj':'edet']
            es = row['e1':'e7']

            #print('---Sample ' + str(i) + '---')

            vsj, pj, Dj, edet = compute_CJ(*jwls.values)
            #print(pd.concat({'Andrew': pd.Series({'dcj': Dj, 'pcj': pj, 'vcj': v0*vsj, 'edet': edet}),'Chris': chks},axis=1))

            for i in es.index:
                vs = float(i[1:])
                #print(es[i], compute_e_release(*jwls.values,edet,vs))

    def __len__(self):
        """Return number of samples in dataset."""
        return self.Nsamples

    def __getitem__(self, index):
        """Return a tuple of a batch's input and output data for training at a given
        index.

        """
        # Get the PDV input
        tslice = slice('t0.1','t4.5')
        ts = self.df.iloc[index][tslice]
        tmins = self.stats.loc['min',tslice]
        tmaxs = self.stats.loc['max',tslice]
        input = np.array((ts-tmins)/(tmaxs-tmins))

        # Get the JWL parameter output
        jwlslice = slice('a','r2')
        jwls = self.df.iloc[index][jwlslice]
        jwlmins = self.stats.loc['min',jwlslice]
        jwlmaxs = self.stats.loc['max',jwlslice]
        output = np.array((jwls-jwlmins)/(jwlmaxs-jwlmins))
        
        return input, output
    
if __name__ == '__main__':
    """For testing and debugging.

    """

    # Imports for plotting
    # To view possible matplotlib backends use
    # >>> import matplotlib
    # >>> bklist = matplotlib.rcsetup.interactive_bk
    # >>> print(bklist)
    import matplotlib.pyplot as plt
    import matplotlib
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    import os
    matplotlib.use('MacOSX')
#     matplotlib.use('TkAgg')
#     # Get rid of type 3 fonts in figures
#     matplotlib.rcParams['pdf.fonttype'] = 42
#     matplotlib.rcParams['ps.fonttype'] = 42
#     import matplotlib.pyplot as plt
#     # Ensure LaTeX font
#     font = {'family': 'serif'}
#     plt.rc('font', **font)
#     plt.rcParams['figure.figsize'] = (6, 6)
#     from mpl_toolkits.axes_grid1 import make_axes_locatable

    data_df = pd.read_csv('samples_sand-all.csv', sep=",", header=0, engine="python")
    print(data_df.keys())
    print(data_df.describe())
    f,aax = plt.subplots(1,2)

    cylex = CYLEX_pdv2jwl_Dataset(slice(2500,None),'samples_sand-all.csv')

    tstr = data_df.keys().to_series()['t0.1':'t4.5'].values
    tt = [float(s[1:]) for s in tstr]
    jwlstr = data_df.keys().to_series()['a':'r2'].values 
    for i in range(0,len(cylex)):
         aax[0].plot(tt,cylex[i][0],'o')
         aax[1].semilogy(cylex[i][1],'o')
    ax = aax[0]
    ax.set_xlabel(r'$t$ [$\mu$s]')
    ax.set_ylabel(r'$v$ [km/s]')
    ax = aax[1]
    ax.set_xticks(range(0,len(jwlstr)))
    ax.set_xticklabels(jwlstr)#, rotation='vertical', fontsize=18)
    f.tight_layout()
    plt.show()
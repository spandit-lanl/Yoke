"""Functions and classes for torch DataSets which sample the Cylinder Test/JWL data."""

####################################
# Packages
####################################
import numpy as np
from torch.utils.data import Dataset
from scipy.optimize import root
import pandas as pd


def compute_CJ(
    A: float, B: float, C: float, G: float, R1: float, R2: float, v0: float
) -> (float, float, float):
    """Return CJ state including edet from JWL parameters.

    Args: A,B,C in units of pressure
        G, R1, R2 are unitless
        v0 is in units of specific volume
    Returns: Dj = [velocity=sqrt(energy)], pj [pressure], vj/v0 [unitless],
             edet [specific energy=pressure x specific volume]
    """
    # ideal gas cp/cv constant descibing large expansion behavior
    k = G + 1
    kp1 = k + 1

    # reference isentrope: ps(v/v0)
    def ps(vs: float) -> float:
        return A * np.exp(-R1 * vs) + B * np.exp(-R2 * vs) + C / vs**k
    # adiabatic gamma on reference isentrope: gs(v/v0)
    def gs(vs: float) -> float:
        return (vs * (A * R1 * np.exp(-R1 * vs) +
                      B * R2 * np.exp(-R2 * vs) +
                      C * k / vs**kp1)
         / ps(vs))
    # at CJ: vj/v0 = gj/(gj+1) assuming p0=0
    def fzero(vs: float) -> float: return gs(vs) * (vs - 1) + vs
    vsj = root(fzero, 0.5, tol=1e-14).x[0]
    pj = ps(vsj)
    Dj = np.sqrt(pj * v0 / (1 - vsj))
    # detonation energy available for HE expansion work
    edet = v0 * vsj / G * (
        pj
        + A * (G / R1 / vsj - 1) * np.exp(-R1 * vsj)
        + B * (G / R2 / vsj - 1) * np.exp(-R2 * vsj)
    ) + pj / 2 * v0 * (vsj - 1)

    return np.array((Dj, pj, vsj, edet))


def compute_e_release( A: float, B: float, C: float, G: float,
                      R1: float, R2: float, v0: float, edet: float, vs: float) -> float:
    """Return the detonation energy released at expansion vs=v/v0.

    Args: A,B,C in units of pressure
        G, R1, R2 are unitless
        v0 is in units of specific volume
        edet is the total detonation energy available
             in units of specific energy=pressure x specific volume
        vs=v/v0 is the number of initial volumes of expansion
    Returns: detonation energy released in units of specific energy
    """
    def Int(vs: float) -> float: return v0 * (
        A / R1 * np.exp(-R1 * vs) + B / R2 * np.exp(-R2 * vs) + C / G / vs**G
    )
    return edet - Int(vs)


####################################
# DataSet Classes
####################################


class CYLEX_pdv2jwl_Dataset(Dataset):
    """PDV to JWL Dataset object for the *CYLEX/JWL*."""
    def __init__(self, rng: slice, file: str) -> None:
        """PDV to JWL Dataset object for the *CYLEX/JWL*.

        The JWL reference isentrope is given by
        ps(v/V0) = a e^{-r1 v/V0} + b e^{-r2 v/V0} + c (v/V0)^{-w-1}
        The parameter units are: [a] = [b] = [c] = GPa
                                 [w] = [r1] = [r2] = -
                                 [V0] = cc/g
        The unit system is {GPa, mm, mus} so that velocity is in km/s
        and energy is in kJ/g

        Args:
            rng (slice): a slice object (start,stop,step) used
                         to sample the data in *file*.
            file (str): .csv file with recorderd data with header
                         [a, b, c, w, r1, r2, V0,
                         dcj, pcj, vcj, edet,
                         e1, e2, e3, e4, e5, e6, e7,
                         t0.1, t0.15, t0.25, t0.35, t0.5, t0.75,
                         t1, t1.5, t2, t2.5, t3.5, t4.5]
        """
        # Model Arguments
        self.file = file

        df = pd.read_csv(file, sep=",", header=0, engine="python")
        self.df = df = df.iloc[rng]

        self.Nsamples = len(df)
        #self.resetCJ()

    def resetCJ(self) -> None:
        """Reset CJ and expansion energy for given JWL parameter."""
        for i, row in self.df.iterrows():
            jwls = row["a":"V0"]
            chks = row["dcj":"edet"]
            es = row["e1":"e7"]

            Dj, pj, vsj, edet = compute_CJ(*jwls.values)
            chks["dcj"] = Dj
            chks["pcj"] = pj
            chks["vcj"] = vsj * row["V0"]
            chks["edet"] = edet

            for i in es.index:
                vs = float(i[1:])
                es[i] = compute_e_release(*jwls.values, edet, vs)

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return self.Nsamples

    def __getitem__(self, index: int) -> ([float],[float]):
        """Return a tuple (input,output) data for training at a given index."""
        # Get the PDV input
        vs = self.df.iloc[index]["t0.1":"t4.5"]
        # Get the JWL parameter output
        jwls = self.df.iloc[index]["a":"r2"]

        input = np.array(vs.values)
        output = jwls.values

        return input, output


class CYLEXnorm_pdv2jwl_Dataset(CYLEX_pdv2jwl_Dataset):
    """Normalized PDV to JWL Dataset object for the *CYLEX/JWL*."""
    def __init__(self, rng: slice, file: str) -> None:
        """PDV to JWL Normalized-Dataset object for the *CYLEX/JWL*."""
        # Model Arguments
        super().__init__(rng,file)
        self.stats = self.df.describe()
        self.tslice = slice("t0.1", "t4.5")
        self.pdvmin = self.stats.loc["min", self.tslice].min()
        self.pdvmax = self.stats.loc["max", self.tslice].max()
        self.jwlslice = slice("a", "r2")
        self.jwlmins = self.stats.loc["min", self.jwlslice]
        self.jwlmaxs = self.stats.loc["max", self.jwlslice]

    def __getitem__(self, index: int) -> ([float],[float]):
        """Return a tuple (input,output) data for training at a given index."""
        # Get the PDV input
        pdvs = self.df.iloc[index][self.tslice]
        input = np.array((pdvs - self.pdvmin) / (self.pdvmax - self.pdvmin))

        # Get the JWL parameter output
        jwls = self.df.iloc[index][self.jwlslice]
        output = np.array((jwls - self.jwlmins) / (self.jwlmaxs - self.jwlmins))

        return input, output


if __name__ == "__main__":
    """For testing and debugging.

    """

    # Imports for plotting
    # To view possible matplotlib backends use
    # >>> import matplotlib
    # >>> bklist = matplotlib.rcsetup.interactive_bk
    # >>> print(bklist)
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("MacOSX")
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

    # load data and get initial stats
    data_df = pd.read_csv("samples_sand-all.csv", sep=",", header=0, engine="python")
    print(data_df.keys())
    print(data_df.describe().loc[:, "dcj":"edet"])
    print(data_df.describe().loc[:, "edet":"e7"])

    # create Dataset object and access for plotting
    cylex = CYLEXnorm_pdv2jwl_Dataset(slice(0, None), "samples_sand-all.csv")

    f, aax = plt.subplots(1, 2)
    tstr = data_df.keys().to_series()["t0.1":"t4.5"].values
    tt = np.array([float(s[1:]) for s in tstr])
    tt = tt - tt[0]
    jwlstr = data_df.keys().to_series()["a":"r2"].values
    for i in range(0, len(cylex)):
        aax[0].plot(tt, cylex[i][0], "x")
        aax[1].semilogy(cylex[i][1], "x")
    ax = aax[0]

    # # load Pemberton 2011 Cylinder test data for comparison
    # for sn in range(1, 9):
    #     S_df = pd.read_csv(
    #         "Shot0" + str(sn) + "velocities.txt",
    #         sep=",",
    #         header=7,
    #         engine="python",
    #         skipinitialspace=True,
    #     )
    #     S_df = S_df.T.drop_duplicates().T
    #     i = 1
    #     cols = {}
    #     for k in S_df.keys():
    #         nn = k
    #         pp = k.split(" ")
    #         if pp[0] == "Time":
    #             nn = pp[0] + " " + str(i) + " " + pp[1].split(".")[0]
    #         elif pp[0] == "Sigma_Y":
    #             nn = pp[0] + " " + str(i) + " " + pp[1].split(".")[0]
    #             i = i + 1
    #         cols[k] = nn
    #     S_df = S_df.rename(columns=cols)

    #     for i in range(1, 9):
    #         tt = S_df.loc[:, "Time " + str(i) + " (seconds)"]
    #         vv = S_df.loc[:, "Probe " + str(i) + " (m/s)"]
    #     aax[0].plot((tt - tt[0]) * 1e6, vv.values / 1e3, ".", markersize=2)

    ax.set_xlabel(r"$t$ [$\mu$s]")
    ax.set_ylabel(r"$v$ [km/s]")
    ax.set_xlim([0, 6])
    ax = aax[1]
    ax.set_xticks(range(0, len(jwlstr)))
    ax.set_xticklabels(jwlstr)
    f.tight_layout()
    plt.show()

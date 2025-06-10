import multiprocessing as mp

import ROOT
import numpy as np
import pandas as pd
import math
import itertools
import sys
sys.path.append('/Users/alexanderantonakis/Desktop/Software/CEVNS_RATES/Classes')
import os

# Main workhorse for integration
from scipy.integrate import dblquad


# Custom imports
from snsgen import SNSGen

outname = sys.argv[1]

mdet = 1000 #kg
L = 20. # meters
# POT in 1 year
pot = 2.1*(10**23)
#material = "Ar40"
#material = "He"
material = "Ar40"

gen = SNSGen(mdet, L, pot, material)

print("Set up the SNS generator for our config")

def N(args):
    binx, biny = args
    def f(y, x):
        return 4*math.pi*gen.dN_dEr_dcos_muon(y, x)
    #print(binx, biny)
    Er1 = 1 + (binx-1)
    Er2 = Er1 + 1
    cos1 = 0.01 + (biny-1)*0.01
    cos2 = cos1 + 0.01
    v = dblquad(f, cos1, cos2, Er1/1000, Er2/1000)[0]*gen.pot
    #print(binx, biny, h.GetBinContent(binx, biny))
    return (binx, biny, v)
    


if __name__ == '__main__':
    import locale
    import os

    # Fix locale issue (if needed)
    os.environ["LC_ALL"] = "en_US.UTF-8"
    os.environ["LANG"] = "en_US.UTF-8"
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

    Ermin = 1
    Ermax = 1500

    nbinsx = Ermax-Ermin

    cosmin = 0.01
    cosmax = 1

    coswidth = 0.01
    nbinsy = int((cosmax - cosmin)/coswidth)

    h = ROOT.TH2D("N_2D", "", nbinsx, Ermin, Ermax, nbinsy, cosmin, cosmax)
  

    # Create list of all (binx, biny) pairs
    bin_indices = [
        (binx, biny)
        for binx in range(1, nbinsx + 1)
        for biny in range(1, nbinsy + 1)
    ]

    print("")
    print("starting the Multiprocessing ...")
    print("")
    # Use multiprocessing to compute values in parallel
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(N, bin_indices)


    # Now fill the histogram serially (ROOT is not multiprocessing-safe!)
    for binx, biny, value in results:
        h.SetBinContent(binx, biny, value)



    outfile = ROOT.TFile(outname, "RECREATE")
    h.SetDirectory(0)
    outfile.cd()
    h.Write()
    outfile.Close()
    print("Finished!")
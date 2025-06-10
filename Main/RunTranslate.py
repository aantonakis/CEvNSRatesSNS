import multiprocessing

import ROOT
import numpy as np
import pandas as pd
import math
import itertools
import sys
sys.path.append('/Users/alexanderantonakis/Desktop/Software/CEVNS_RATES/Classes')
import os

from scipy.optimize import minimize
from sklearn.decomposition import PCA


from Translate import*
from TrackFit import*

# SRIM Files of the same Er
files = sys.argv[1]

# configuration passed to this script
sigma_T = float(sys.argv[2])
sigma_L = float(sys.argv[3])
orient = float(sys.argv[4])
eff = float(sys.argv[5])
mult = float(sys.argv[6])
thresh = float(sys.argv[7])
voxel_width = float(sys.argv[8])

# Output file name
out_name = sys.argv[9]

#config = [sigma_T, sigma_L, orient, eff, mult, thresh, voxel_width]

#thetas = np.linspace(0, math.pi/2, 100)
costhetas = np.linspace(0, 1, 5)
thetas = np.arccos(costhetas)
phis = np.linspace(0, math.pi*2, 5)
args = list(itertools.product(thetas, phis))

df = pd.read_csv(files)
df = df.query("Ion != 1")
ions = np.array(list(set(list(df["Ion"].values))))

# Convert to Microns
df["X"] *= (10**-10)
df["X"] *= 10**6
df["Y"] *= (10**-10)
df["Y"] *= 10**6
df["Z"] *= (10**-10)
df["Z"] *= 10**6


print("")
print("Got Ions")
print("")

def Simulate(theta, phi):
    #print("In Simulate")
    pairs = []
    for ion in ions:
        # first get the track info (df) and place in detector coordinates
        #print("Make Detector Coordinates")
        track = rotate_thetar(df, ion, theta)
        rotate_phi(track, phi)

        #print("Diffusion")
        # apply the diffusion of the config
        x, y, z = Detector_Diffusion(track, sigma_T, sigma_L, orientation=orient)

        #print("Pixelate")
        # pixelate according to the voxelization of the config
        h = pixelate(x, y, z, voxel_width, "ion"+str(ion))
        #return 0
        #print("Eff, Gain, Threshold")
        apply_uniform_eff(h, eff)
        apply_gain(h, mult)
        apply_threshold(h, thresh)
        x_reco, y_reco, z_reco, weights_reco = get_data_simple(h)
        p, w = filter_voxels_by_weight(x_reco, y_reco, z_reco, weights_reco, n=5)

        # Fit weighted PCA
        pca_weighted = weighted_pca(p, w, n_components=1)

        # Get the principal component (direction of the line)
        line_direction = pca_weighted.components_[0]
        pairs.append((np.cos(theta), line_direction[2]))

    return pairs

        
if __name__ == '__main__':
    import locale
    import os

    # Fix locale issue (if needed)
    os.environ["LC_ALL"] = "en_US.UTF-8"
    os.environ["LANG"] = "en_US.UTF-8"
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

    print("")
    print("starting the Multiprocessing ...")
    print("")

    #sys.exit()
    with multiprocessing.Pool() as pool:
        results = pool.starmap(Simulate, args)

        print("")
        print("##################################################################")
        print("##################################################################")
        print("##################################################################")
        print("")
        print("DONE : DONE : DONE : DONE : DONE")
        print(results[:5])
        print("")
        print("##################################################################")
        print("##################################################################")
        print("##################################################################")
        print("")

        tree = ROOT.TNtuple("angle_res_tree", "rangle_res_tree", "cos_true:cos_reco")
        for result in results:
            for pair in result:
                tree.Fill(pair[0], pair[1])

        
        config_tree = ROOT.TNtuple("config_tree", "config_tree", "sigma_T:sigma_L:orient:eff:gain:thresh:voxel_size")
        config_tree.Fill(sigma_T, sigma_L, orient, eff, mult, thresh, voxel_width)


        outfile = ROOT.TFile("out.root", "RECREATE")
        tree.SetDirectory(0)
        config_tree.SetDirectory(0)
        outfile.cd()
        config_tree.Write()
        tree.Write()
        outfile.Close()
        print("Finished!")










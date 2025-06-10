import ROOT
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec

from scipy.optimize import minimize
from sklearn.decomposition import PCA


def get_data(h3):
    # Assume h3 is your filled TH3D histogram
    X, weights = [], []

    for ix in range(1, h3.GetNbinsX() + 1):
        for iy in range(1, h3.GetNbinsY() + 1):
            for iz in range(1, h3.GetNbinsZ() + 1):
                content = h3.GetBinContent(ix, iy, iz)
                if content > 0:
                    x = h3.GetXaxis().GetBinCenter(ix)
                    y = h3.GetYaxis().GetBinCenter(iy)
                    z = h3.GetZaxis().GetBinCenter(iz)
                    X.append([x, y, z])
                    #x_vals.append(x)
                    #y_vals.append(y)
                    #z_vals.append(z)
                    weights.append(content)

    #points = np.array([x_vals, y_vals, z_vals]).T
    X = np.array(X)
    weights = np.array(weights)
    return X, weights

def get_data_simple(h3):
    # Assume h3 is your filled TH3D histogram
    x_vals, y_vals, z_vals, weights = [], [], [], []

    for ix in range(1, h3.GetNbinsX() + 1):
        for iy in range(1, h3.GetNbinsY() + 1):
            for iz in range(1, h3.GetNbinsZ() + 1):
                content = h3.GetBinContent(ix, iy, iz)
                if content > 0:
                    x = h3.GetXaxis().GetBinCenter(ix)
                    y = h3.GetYaxis().GetBinCenter(iy)
                    z = h3.GetZaxis().GetBinCenter(iz)
                    x_vals.append(x)
                    y_vals.append(y)
                    z_vals.append(z)
                    weights.append(content)

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    z_vals = np.array(z_vals)
    weights = np.array(weights)
    return x_vals, y_vals, z_vals, weights

def make_line(point_on_line, direction_vector):
    line = ROOT.TPolyLine3D(2)
    length = 50  # adjust as needed

    # Calculate two points along the fit line
    p1 = point_on_line - length * direction_vector
    p2 = point_on_line + length * direction_vector

    line.SetPoint(0, p1[0], p1[1], p1[2])
    line.SetPoint(1, p2[0], p2[1], p2[2])
    line.SetLineColor(ROOT.kRed)
    line.SetLineWidth(3)
    return line

def weighted_pca(X, weights, n_components=1):
  """
  Performs weighted PCA.

  Args:
    X:  Data matrix (n_samples, n_features).
    weights: Sample weights (n_samples,).
    n_components: Number of principal components to keep.

  Returns:
    PCA object fitted to the weighted data.
  """
  # Scale points by the square root of the weights
  X_weighted = X * np.sqrt(weights[:, np.newaxis])
  # Fit standard PCA to the scaled data
  pca = PCA(n_components=n_components)
  pca.fit(X_weighted)
  return pca

def filter_voxels_by_weight(x, y, z, w, n=0):
    # n is the number of voxels to keep for fitting
    xp, yp, zp, wp = x.copy(), y.copy(), z.copy(), w.copy()
    indices = np.argpartition(wp, -1*n)[-1*n:]
    xp = xp[indices]
    yp = yp[indices]
    zp = zp[indices]
    wp = wp[indices]
    X = []
    for i in range(len(xp)):
        X.append([xp[i], yp[i], zp[i]])
    X = np.array(X)

    return X, wp


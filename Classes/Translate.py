import ROOT
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec


# For a SRIM DataFrame of our File Type, return a track DataFrame for the chosen ion
def rotate_thetar(df, ion, theta):
    track_df = df.query("Ion == "+str(ion))
    #weights = track_df.query("Primary < 2")["NormEnergy"].values
    #x = track_df.query("Primary < 2")["X"].values
    #z = track_df.query("Primary < 2")["Z"].values

    thetap = (math.pi/2) - theta

    M = np.array([[np.cos(thetap), -1*np.sin(thetap)],
                  [np.sin(thetap), np.cos(thetap)]])
    
    def get_xp(row):
        v = np.array([row['X'], row['Z']])
        vp = np.dot(M, v)
        return vp[0]
    
    def get_zp(row):
        v = np.array([row['X'], row['Z']])
        vp = np.dot(M, v)
        return vp[1]  
    
    track_df["xp"] = track_df.apply(get_xp, axis=1)
    track_df["zp"] = track_df.apply(get_zp, axis=1)

    return track_df

def rotate_phi(track_df, phi):

    M = np.array([[np.cos(phi), -1*np.sin(phi)],
                  [np.sin(phi), np.cos(phi)]])
    
    def get_yp(row):
        v = np.array([row['xp'], row['Y']])
        vp = np.dot(M, v)
        return vp[1]
    
    def get_xp(row):
        v = np.array([row['xp'], row['Y']])
        vp = np.dot(M, v)
        return vp[0]  
    
    track_df["yp"] = track_df.apply(get_yp, axis=1)
    track_df["xp"] = track_df.apply(get_xp, axis=1)


def Detector_Diffusion(track_df, sigma_T, sigma_L, orientation=0):
    # orientation == 0: drift transverse to the Beam 
    # orientation == 1: drift longitudinal to the Beam

    weights = track_df.query("Primary < 2")["NormEnergy"].values
    weights /= 23.6  # convert to ionization electrons

    # Assume Detector Coordinates for diffusion
    x = track_df.query("Primary < 2")["xp"].values
    y = track_df.query("Primary < 2")["yp"].values
    z = track_df.query("Primary < 2")["zp"].values

    X, Y, Z = [], [], []
    for num in range(len(x)):
        dev_x, dev_y, dev_z = 0, 0, 0
        dev_y = np.random.normal(loc=0, scale=sigma_T, size=int(weights[num]))
        # transverse drift
        if orientation == 0:
            dev_x = np.random.normal(loc=0, scale=sigma_L, size=int(weights[num]))
            dev_z = np.random.normal(loc=0, scale=sigma_T, size=int(weights[num]))
        # Longitudinal drift
        else:
            dev_x = np.random.normal(loc=0, scale=sigma_T, size=int(weights[num]))
            dev_z = np.random.normal(loc=0, scale=sigma_L, size=int(weights[num]))            
            
        dev_x += x[num]
        dev_y += y[num]
        dev_z += z[num]

        X += list(dev_x)
        Y += list(dev_y)
        Z += list(dev_z)
    
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    
    return X, Y, Z

def pixelate(x, y, z, voxel_width, name):

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    zmin, zmax = np.min(z), np.max(z)

    Nxp = int(abs(xmax)/(voxel_width/2)) + 1
    Nyp = int(abs(ymax)/(voxel_width/2)) + 1
    Nzp = int(abs(zmax)/(voxel_width/2)) + 1
    Nxm = int(abs(xmin)/(voxel_width/2)) + 1
    Nym = int(abs(ymin)/(voxel_width/2)) + 1
    Nzm = int(abs(zmin)/(voxel_width/2)) + 1

    x1 = -1*Nxm*(voxel_width/2)
    x2 = Nxp*(voxel_width/2)
    Lx = x2 - x1
    Bx = int(Lx/voxel_width)

    y1 = -1*Nym*(voxel_width/2)
    y2 = Nyp*(voxel_width/2)
    Ly = y2 - y1
    By = int(Ly/voxel_width)

    z1 = -1*Nzm*(voxel_width/2)
    z2 = Nzp*(voxel_width/2)
    Lz = z2 - z1
    Bz = int(Lz/voxel_width)

    h = ROOT.TH3D(name, name, Bx, x1, x2, By, y1, y2, Bz, z1, z2)

    for num in range(len(x)):
        h.Fill(x[num], y[num], z[num])

    return h


def apply_uniform_eff(h, e):
    h.Scale(e)


def apply_gain(h3, mult):

    for ix in range(1, h3.GetNbinsX() + 1):
        for iy in range(1, h3.GetNbinsY() + 1):
            for iz in range(1, h3.GetNbinsZ() + 1):
                content = h3.GetBinContent(ix, iy, iz)
                if content > 0:
                    h3.SetBinContent(ix, iy, iz, mult**content)

def apply_threshold(h3, threshold):

    for i in range(1, h3.GetNbinsX()+1):
        for j in range(1, h3.GetNbinsY()+1):
            for k in range(1, h3.GetNbinsZ()+1):
                C = h3.GetBinContent(i, j, k)
                if C < threshold:
                    h3.SetBinContent(i, j, k, 0)

############# SIMULATE Full Chain ####################

def DetSimV1(df, ion, theta, phi, args):
    # Adhere to the structure below!
    #theta = args[0]
    #phi = args[1]
    sigma_T = args[0]
    sigma_L = args[1]
    orient = args[2]
    eff = args[3]
    mult = args[4]
    thresh = args[5]
    voxel_width = args[6]

    # first get the track info (df) and place in detector coordinates
    track = rotate_thetar(df, ion, theta)
    rotate_phi(track, phi)

    # apply the diffusion of the config
    x, y, z = Detector_Diffusion(track, sigma_T, sigma_L, orientation=orient)

    # pixelate according to the voxelization of the config
    h = pixelate(x, y, z, voxel_width, "ion"+str(ion))

    apply_uniform_eff(h, eff)
    apply_gain(h, mult)
    apply_threshold(h, thresh)
    return h



#!/usr/bin/env python3
"""
Author: Antoine Venturini
Date: 2022
The purpose of this script is the
calculation of derivatives of the chi square function
with respect to local and global alignment parameters,
and the definition of functions to calculate
the proper matrices for solving the minimization problems
for the CDCH alignment, accordingly with the MillePede
strategy. This will be conducted with cosmic rays.
"""
import gc
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import time
import struct
import array

from ROOT import TChain, TFile, gInterpreter, TH1D
# Load trk library
gInterpreter.ProcessLine('#include "cosmics_includes.h"')

import matplotlib.pyplot as plt
from numpy.linalg import inv
import numpy as np
from symengine import lambdify, diff, symbols, sin, cos, Matrix, sqrt # Package for symbolic calc in Python. Based on C and C++
import sympy as sym # Package for symbolic calc in Python


####################### GEOMETRY DEFINITION OF THE CDCH #################################

# Definition of variables for sym
sym.init_printing(use_unicode=True)
# global parameters
x0, y0, z0, s, gamma, theta, phi, L, z_ds, z_us, zi, ti, sigma_i = symbols("x0 y0 z0 s gamma theta phi L z_ds z_us zi ti sigma_i")
# local parameters
mxy, qxy, myz, qyz = symbols("mxy qxy myz qyz")

# Definition of wire geometry
def wire_coord(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, zi):
    """
    Calculate wire coordinates
    """
    # parameterization of the wire from coordinate transformation
    waxis = Matrix([cos(phi)*sin(theta),
                    sin(phi)*sin(theta),
                    cos(theta)])
    wirepos = Matrix([x0, y0, z0])
    vaxis = - (wirepos + zi*waxis)
    vaxis = Matrix([vaxis[0], vaxis[1], 0])
    vaxis = vaxis/(sqrt(vaxis.dot(vaxis)))
    vaxis = vaxis - vaxis.dot(waxis)*waxis
    vaxis = vaxis/(sqrt(vaxis.dot(vaxis)))
    uaxis = vaxis.cross(waxis)
    uaxis = uaxis/sqrt(uaxis.dot(uaxis))
    rot = Matrix([[uaxis[0], vaxis[0], waxis[0]],
                  [uaxis[1], vaxis[1], waxis[1]],
                  [uaxis[2], vaxis[2], waxis[2]]])
    # s point
    sg = sin(gamma)
    cg = cos(gamma)
    sag_v = Matrix([-x0/sqrt(x0*x0 + y0*y0),
                    -y0/sqrt(x0*x0 + y0*y0),
                    -0])
    sag_v = sag_v - (sag_v.dot(waxis))*waxis
    sag_v = sag_v / sqrt(sag_v.dot(sag_v))
    rot_m = Matrix([[waxis[0]**2*(1-cg) + cg,
                     (1-cg)*waxis[0]*waxis[1] - sg*waxis[2],
                     (1-cg)*waxis[0]*waxis[2] + sg*waxis[1]],
                    [(1-cg)*waxis[0]*waxis[1] + sg*waxis[2],
                     waxis[1]**2*(1-cg) + cg,
                     (1-cg)*waxis[1]*waxis[2] - sg*waxis[0]],
                    [(1-cg)*waxis[2]*waxis[0] - sg*waxis[1],
                     (1-cg)*waxis[1]*waxis[2] + sg*waxis[0],
                      waxis[2]**2*(1-cg) + cg]])
    sag_v = rot_m * sag_v
    sag = s * ((zi*2/L)**2 - 1)
    pos0 = wirepos + sag * sag_v
    
    pos = Matrix([0, 0, zi])
    a_wire = pos0 + rot*pos
    return a_wire

def x_wire_coord(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, zi):
    """
    Calculate wire X coordinate
    """
    a_wire = wire_coord(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, zi)
    return a_wire[0]

def y_wire_coord(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, zi):
    """
    Calculate wire Y coordinate
    """
    a_wire = wire_coord(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, zi)
    return a_wire[1]

def z_wire_coord(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, zi):
    """
    Calculate wire Z coordinate
    """
    a_wire = wire_coord(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, zi)
    return a_wire[2]

# Definition of hit residual
def res(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i):
    """
    Calculates the correction to global chi square from a given data point.
    """
    # parameterization of the wire from coordinate transformation
    a_wire = wire_coord(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, zi)
    # parameterization of the track
    # Calculate yt, the intersection point of the line with the drift plane
    line_vector = Matrix([mxy,
                          1,
                          myz])
    line_origin = Matrix([qxy,
                          0,
                          qyz])
    wire_vector = Matrix([cos(phi)*sin(theta),
                    sin(phi)*sin(theta),
                    cos(theta)])
    yt = (a_wire - line_origin).dot(wire_vector) / (line_vector.dot(wire_vector))
    # Track interesection point
    a_track = Matrix([mxy*yt + qxy,
                      yt,
                      myz*yt + qyz])
    # Return the hit residual
    return sqrt((a_track - a_wire).dot(a_track - a_wire)) - ti

# Chi squared function calculation
def chi2(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i):
    """
    Calculates the correction to global chi square from a given data point.
    """
    a_res = res(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i)
    return 0.5 * a_res**2 / sigma_i**2

#In order to make calculation fast, the chi 2 function and its derivatives must be lambdified.
fast_chi2 = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i], [chi2(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i)])
fast_x = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, zi], [x_wire_coord(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, zi)])
fast_y = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, zi], [y_wire_coord(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, zi)])
fast_z = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, zi], [z_wire_coord(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, zi)])
fast_res = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i], [res(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i)])
fast_der_x0 = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i], [diff(res(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i), x0)])
fast_der_y0 = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i], [diff(res(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i), y0)])
fast_der_z0 = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i], [diff(res(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i), z0)])
fast_der_theta = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i], [diff(res(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i), theta)])
fast_der_phi = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i], [diff(res(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i), phi)])
fast_der_gamma = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i], [diff(res(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i), gamma)])
fast_der_s = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i], [diff(res(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i), s)])
fast_der_L = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i], [diff(res(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i), L)])
fast_der_mxy = lambdify([x0, y0, z0,theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i], [diff(res(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i), mxy)])
fast_der_qxy = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i], [diff(res(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i), qxy)])
fast_der_myz = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i], [diff(res(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i), myz)])
fast_der_qyz = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i], [diff(res(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i), qyz)])

fast_wire_coord = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, zi], [wire_coord(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, zi)])

# lists storing derivatives
fast_der_align = [fast_der_x0,
                  fast_der_y0,
             	  fast_der_theta,
                  fast_der_phi,
                  fast_der_s]
                  #fast_der_gamma,
                  #fast_der_s]

fast_der_track = [fast_der_mxy,
             	  fast_der_qxy,
                  fast_der_myz,
                  fast_der_qyz]

########################################################################

####################### GLOBAL VARIABLES ################################

# Definition of dictionary to map wires to a dense vector
good_wires = np.loadtxt("MC_wire_list_iter0.txt", dtype='int', unpack=True) #np.loadtxt("wire_list_CYLDCH35_def.txt", dtype='int', unpack=True)
idx = np.arange(0, len(good_wires), 1)
label = np.arange(0, len(good_wires), 1)
dict_wire_to_label = dict(zip(good_wires, label))
dict_wire_to_idx = dict(zip(good_wires, idx))
dict_idx_to_wire = dict(zip(idx, good_wires))
# Number of global and local parameters
N_WIRES = len(good_wires)
ALIGN_PARS = len(fast_der_align)
N_ALIGN = int(N_WIRES * ALIGN_PARS) # total number of align. parameters
N_FIT = len(fast_der_track) # number of fit parameters to a line in 3D

###########################################################################


####################### CLASSES FOR DATA HANDLING #########################

GEO_ID = 46 # Geometry ID to start millepede

list_wires_parameters = np.loadtxt(f"wire_parameters_CYLDCH{GEO_ID}.txt") 

class Wire():
    """
    Wire class: contains all the informations about the wire
    geometry and alignment parameters.
    The alignment parameters are taken from the wire_parameters_CYLDCH*.txt file,
    ID = 46 for survey, ID = xxx for alignment step.
    the wire number must be declared.
    """
    def __init__(self, ID, wire):
        self.alignpars = list_wires_parameters[wire]

class CRHit() :
    """
    A hit in a cosmic ray event. Infos about the wire alignment parameters
    and the fit parameters are also stored.
    """
    def __init__(self,
                 ID,
                 wire,
                 mx,
                 qx,
                 mz,
                 qz,
                 z_i,
                 di,
                 sigma) :
        this_wire = Wire(ID, wire)
        wire_params = this_wire.alignpars
        self.params = [*wire_params, mx, qx, mz, qz, z_i, di, sigma]
        self.wire = wire
        self.chi2 = fast_chi2(self.params)
        self.res = fast_res(self.params)
        self.x = fast_x([*wire_params, z_i])
        self.y = fast_y([*wire_params, z_i])
        self.z = fast_z([*wire_params, z_i])
        #print("Residual =", self.res)
        self.sigma = sigma
        self.der_align = [der(self.params) for der in fast_der_align]
        #print("Derivatives of alingment parameters", self.der_align)
        if not any(self.der_align):
            print("Found zeros in GLB derivatives!")
        self.der_track = [der(self.params) for der in fast_der_track]
        #print("Derivatives of track params", self.der_track)
        if not any(self.der_track):
            print("Found zeros in local derivatives!")
        # Create chi2 function of one parameter to scan the chi square function
        """
        scanning_parameter = np.linspace(-0.01, 0.01, 100)
        chi2_scan = []
        hit_params_scan = self.params
        for scan in scanning_parameter:
            # replace the desired parameter
            #print(f"hit_params + scan = {self.params[6]:.4f} + {scan:.4f}")
            hit_params_scan[6] = scan
            #if self.chi2 < 3:
            chi2_scan.append(fast_chi2(hit_params_scan))
            #break
        #print(f"Parameters =", self.params)
        self.chi2_scan = np.array(chi2_scan)
        """

class CRTrack() :
    """
    Class allowing to handle easily the 
    data structure of the CRTree in the .root
    file.
    """
    def __init__(self, event, ID):
        """
        creates track and hits
        """
        self.hits = [CRHit(ID,
                           wire,
        		           event.GetLeaf("mxz").GetValue(),
        		           event.GetLeaf("qxz").GetValue(),
        		           event.GetLeaf("myz").GetValue(),
        		           event.GetLeaf("qyz").GetValue(),
        		           z_wire,
        		           doca,
        		           sigma)
                     for wire, z_wire, doca, sigma in zip(event.wire,
                                                          event.w_wire,
                                                          event.doca,
                                                          event.sigma)
                     if abs(z_wire) < 90]
                     #if wire in good_wires] # comment this part to fix the reference of some wires
        if len(self.hits) < 5:
            self.chi2 = 1e30
        else:
            self.chi2 = np.array([hit.chi2 for hit in self.hits]).sum()/(len(self.hits) - 4.) # total chi2/dof
        # Array format to write on binary file for Pede routine
        self.glder = array.array('f')
        self.inder = array.array('i')
        self.glder.append(0.)
        self.inder.append(0)
        if True:
            for hit in self.hits:
                self.glder.append(hit.res)
                self.inder.append(0)
                self.glder.fromlist(hit.der_track)
                self.inder.fromlist([1, 2, 3, 4]) # counts from 1 to N_FIT
                self.glder.append(hit.sigma)
                self.inder.append(0)
                self.glder.fromlist(hit.der_align)
                glb_label = [hit.wire*ALIGN_PARS + i + 1 for i in range(ALIGN_PARS)] #[dict_wire_to_label[hit.wire]*ALIGN_PARS + i + 1 for i in range(ALIGN_PARS)] 
                self.inder.fromlist(glb_label)


    def write_to_binary_file(self, aFile):
        """
        Writes the output binary file to submit to Pede routine
        aFile : object file on which to write. The file should be opened in "ab" mode 
        """
        num_words_to_write = len(self.inder) * 2
        if (len(self.glder) != len(self.inder)):
            print("Array of float and of int not of the same size")
            return
        header = array.array('i')
        header.append(num_words_to_write)
        header.tofile(aFile)
        self.glder.tofile(aFile)
        self.inder.tofile(aFile)


def write_parameter_file(geom):
    """
    write a params.txt file to be used by Pede.
    Contains all initial values of wire parameters from the
    specified geom ID.
    pre-sigma is 0.0 for every parameter except gamma, which is poorly defined
    """
    with open("meg2params_mc.txt", "w") as aFile:
        aFile.write("Parameter\n")
        for w in range(0, 1920):
            wire = Wire(geom, w)
            # Select only relevant params for alignment
            params = [par for i, par in enumerate(wire.alignpars) if i == 0 or i == 1 or (i > 2 and i < 7)]
            for i, par in enumerate(params):
                label = w * ALIGN_PARS + i + 1 #dict_wire_to_label[w] * ALIGN_PARS + i + 1
                pre_sigma = 0.0
                if not w in good_wires:
                    pre_sigma = -1.0 # Fix bad wire parameters 
                if i == 4:
                    pre_sigma = -1.0 # fix gamma
                if ALIGN_PARS == 5:
                    if i < 4:
                        entry = f"{label} {0.0} {pre_sigma}\n"
                        aFile.write(entry)
                    if i > 4:
                        entry = f"{label - 1} {0.0} {pre_sigma}\n"
                        aFile.write(entry)
                elif ALIGN_PARS == 6:
                    entry = f"{label} {0.0} {pre_sigma}\n"
                    aFile.write(entry)

def write_constraint_file():
    """
    Write constraints on parameters.
    For each plane, the global shift in x0 and y0 should be zero (for example).
    """
    with open("meg2const_mc.txt", "w") as aFile:
        # x0 constraints
        for iplane in range(0, 10):
            aFile.write("Constraint 0.0\n")
            for w in range(0, 1920):#good_wires:
                if int(w/192) == iplane:
                    aFile.write(f"{w * ALIGN_PARS + 1} 1.0\n")
        # y0 constraints
        for iplane in range(0, 10):
            aFile.write("Constraint 0.0\n")
            for w in range(0, 1920):#good_wires:
                if int(w/192) == iplane:
                    aFile.write(f"{w * ALIGN_PARS + 2} 1.0\n")

def write_measurement_file():
    """
    Write survey measurements to be used as gaussian constraints
    on parameters. 50 um sigma.
    """
    with open("meg2meas.txt", "w") as aFile:
        id_wire, x_s, y_s, z_s, w_s = np.loadtxt("anode_CYLDCH46.txt", unpack=True)
        for wire, x, y, z, w in zip(id_wire, x_s, y_s, z_s, w_s):
            if int(wire) in good_wires:
                wire_pars = Wire(46, int(wire)).alignpars # ID 46 = nominal, 66 = with 0.1 mm sagittaa
                x0 = wire_pars[0]
                y0 = wire_pars[1]
                z0 = wire_pars[2]
                theta = wire_pars[3]
                phi = wire_pars[4]
                L = wire_pars[7]
                # x measurement
                aFile.write(f"Measurement {x} {0.005}")
                # y measurement


                
##############################################################
#                        BELLURIE                            #
##############################################################

def plot_survey(geom):
    """
    Plot histogram of difference between extremal wires position (w = +/- 95 cm)
    for Nominal Geometry (with values in anode_CYLDCH46.txt) and
    given geometrical ID geom.
    """
    res = []
    # Method 1
    id_wire = np.linspace(0, 1919, 1920, dtype='int')
    z_survey = np.array([-95., 95.])
    for id in id_wire:
        for z in z_survey:
            wire_params = list_wires_parameters[id]
            x_survey = fast_x([*wire_params, z])
            y_survey = fast_y([*wire_params, z])
            z_surv = fast_z([*wire_params, z])
            event = CRHit(geom,
                          id,
                          0.,
                          x_survey + np.random.normal(loc=0., scale=0.003),
                          0.,
                          z_surv + np.random.normal(loc=0., scale=0.003),
                          z,
                          0,
                          0.005)
            res.append(event.res)

    # Method 2
    """
    id_wire, x_s, y_s, z_s, w_s = np.loadtxt("anode_CYLDCH46.txt", unpack=True)
    for wire, x, y, z, w in zip(id_wire, x_s, y_s, z_s, w_s):
        if int(wire) in good_wires:
            event = CRHit(geom,
                          int(wire),
                          0.,
                          x,
                          0.,
                          z,
                          w,
                          0.,
                          0.005)
            res.append(event.res)
    """
    # Plot survey residuals
    plt.figure(1)
    plt.title(f"Difference for GEOMETRY {geom} with respect to Survey")
    plt.xlabel("[cm]")
    plt.hist(np.array(res),
                      bins=100,
                      color="blue")
    plt.grid(alpha=.5)
    plt.legend()
    plt.savefig(f"Survey_residuals_{geom}.pdf")
    plt.show()

def plot_geometry(geom, min_wire=192, max_wire=1920, add_survey=True):
    """
    Plot the Geometry of the Drift Chamber in 3D for all wires between min_wire and max_wire.
    If plot_survey is True, also the US and DS position of the wires as measured in survey are
    drown.
    """
    z_vec = np.linspace(-100., 100., 10)
    wires = [Wire(geom, w) for w in range(min_wire, max_wire)]
    fig = plt.figure()
    plt.title("CDCH Geometry")
    ax = fig.add_subplot(projection="3d")
    # Plot wires
    for w in wires:
        x_w = [fast_wire_coord([*w.alignpars, z])[0][0] for z in z_vec]
        y_w = [fast_wire_coord([*w.alignpars, z])[1][0] for z in z_vec]
        ax.plot(x_w, y_w, z_vec, color='pink', alpha=.2)
    # Add survey points (optional)
    if add_survey:
        id_wire, x_s, y_s, z_s, w_s = np.loadtxt('anode_CYLDCH46.txt', unpack=True) 
        ax.scatter(x_s[w], y_s[w], z_s[w], marker='.', color='red', alpha=.3)
    plt.show()
        

###########################################################################


######################### MILLEPEDE ALGEBRA ###############################
#        Calculating matrices for MillePede.
#        The formulas are taken from the article on CMS alignment
###########################################################################

def calculateGamma(event):
        # matrix Gamma
        matrixGamma = np.zeros(shape=(N_FIT, N_FIT))
        for hit in event:
            if hit.chi2 < 5e1:
                rows = np.array([der_i*hit.der_track/(hit.sigma**2) for der_i in hit.der_track])
                matrixGamma += rows
                del rows
                #gc.collect()
        return matrixGamma

def calculateG(event):
        # matrix G
        matrixG = np.zeros(shape=(N_ALIGN, N_FIT))
        for hit in event:
            if hit.chi2 < 5e1:
                if hit.wire in good_wires:
                    wire = dict_wire_to_idx[hit.wire]
                    rows = np.array([der_i * hit.der_track / (hit.sigma**2) for der_i in hit.der_align])
                    matrixG[wire*ALIGN_PARS : wire*ALIGN_PARS + ALIGN_PARS, : ] += rows
                    del wire, rows
                    #gc.collect()
        return matrixG

def calculateB(event):
    """
    B is the vector containing the misalignment.
    """
    vectorB = np.zeros(N_ALIGN)
    for hit in event:
        if hit.chi2 < 5e1:
            if hit.wire in good_wires:
                wire = dict_wire_to_idx[hit.wire]
                vectorB[wire*ALIGN_PARS : wire*ALIGN_PARS + ALIGN_PARS] += hit.der_align * hit.res / (hit.sigma**2)
                del wire
                #gc.collect()
    return vectorB

def calculateBeta(event):
    """
    Beta is the vector containing the track residuals.
    """
    vectorBeta = np.zeros(N_FIT)
    for hit in event:
        if hit.chi2 < 5e1:
            #if hit.wire in good_wires: #uncomment if no wires is fixed
            vectorBeta += hit.der_track * hit.res / (hit.sigma**2)
    return vectorBeta

def calculateC(event):
    """
    C is the matrix with derivatives with respect
    to the alignment parameters.
    """
    matrixC = np.zeros(shape=(N_ALIGN, N_ALIGN))
    for hit in event:
        if hit.chi2 < 5e1:
            if hit.wire in good_wires:
                wire = dict_wire_to_idx[hit.wire]
                rows = np.array([der_i*hit.der_align / (hit.sigma**2) for der_i in hit.der_align])
                matrixC[wire*ALIGN_PARS : wire*ALIGN_PARS + ALIGN_PARS, wire*ALIGN_PARS : wire*ALIGN_PARS + ALIGN_PARS] += rows
                del rows, wire
                #gc.collect()
    return matrixC

def calculateCPrime(matrixCPrime, event):
    """
    CPrime is the matrix to invert, with both misalignment
    and track fitting parameters.
    """
    
    G = calculateG(event)
    C = calculateC(event)
    try:
        Gamma_inv = inv(calculateGamma(event))
        CorrectionMatrix = - (G @ Gamma_inv) @ G.T
        matrixCPrime += C + CorrectionMatrix
        del CorrectionMatrix, Gamma_inv, G, C
        #gc.collect()
    except np.linalg.LinAlgError:
        print("Error in Gamma inversion...")
        matrixCPrime += C
        del G, C
        #gc.collect()
    return matrixCPrime

def calculateBPrime(vectorBPrime, event):
    """
    Is the vector with the track and alignment residuals.
    """
    G = calculateG(event)
    b = calculateB(event)
    beta = calculateBeta(event)
    try:
        Gamma_inv = inv(calculateGamma(event))
        vectorBPrime += b - G @ (Gamma_inv @ beta)
        del G, b, beta, Gamma_inv
        #gc.collect()
    except np.linalg.LinAlgError:
        print("Error in Gamma inversion...")
        vectorBPrime += b
        del b, G, beta
        #gc.collect()
    return vectorBPrime

def survey(geom, matrixCPrime):
    """
    Insert in the C matrix the survey measurements.
    These are not the proper measurements, but are (x, y, z) points at w = +/- 95 cm
    obtained from software computation with CYLDCH 46.
    50 um error on measurements.
    """
    """
    id_wire, x_s, y_s, z_s, w_s = np.loadtxt("anode_CYLDCH46.txt", unpack=True)
    for wire, x, y, z, w in zip(id_wire, x_s, y_s, z_s, w_s):
        if int(wire) in good_wires:
            event = [CRHit(geom,
                           int(wire),
                           0.,
                           x,
                           0.,
                           z,
                           w,
                           0,
                           0.005)]
    """
    id_wire = np.linspace(0, 1919, 1920, dtype='int')
    z_survey = np.array([-95., 95.])
    for id in id_wire:
        if id not in good_wires:
            continue
        for z in z_survey:
            wire_params = list_wires_parameters[id]
            x_survey = fast_x([*wire_params, z])
            y_survey = fast_y([*wire_params, z])
            z_surv = fast_z([*wire_params, z])
            print(f"Survey point wire {id} at {z} cm : ({x_survey:.2f}, {y_survey:.2f}, {z_surv:.2f})")
            event = [CRHit(geom,
                          id,
                          0.,
                          x_survey + np.random.normal(loc=0., scale=0.003),
                          0.,
                          z_surv + np.random.normal(loc=0., scale=0.003),
                          z,
                          0.,
                          0.005)]
    
            C = calculateC(event)
            matrixCPrime += C
            del event, C
            #gc.collect()
    
    return matrixCPrime

def millepede(geom, tree):
    """
    This function computes and inverts CPrime
    hence it solve the alignment problem.
    """
    # Initialize matrices and vectors
    matrixCPrime = np.zeros(shape=(N_ALIGN, N_ALIGN))
    vectorBPrime = np.zeros(N_ALIGN)

    # Insert survey measurements 
    t_start = time.time()
    #matrixCPrime = survey(geom, matrixCPrime)
    t_stop = time.time()
    #print(f"Survey inserted in {((t_stop - t_start)/3600):.2f} h...")

    # Loop on event to fill the matrices
    #h = TH1D(f"hchi2{GEO_ID}", "Geometry = {GEO_ID}", 100, 0., 10.)
    for i, entry in enumerate(tree):
        #tstartloop = time.time()
        event = CRTrack(entry, geom)
        #h.Fill(event.chi2)
        
        if event.chi2 > 3:
            print("chi2 > 3")
            del event
            #gc.collect()
            continue
        if i%10000 == 0:
            print(f"Evento {i}")
        matrixCPrime = calculateCPrime(matrixCPrime, event.hits)
        vectorBPrime = calculateBPrime(vectorBPrime, event.hits)        
        del event
        #gc.collect()
        #tstoploop = time.time()
        #print(f"{tstoploop - tstartloop:.3f} s for 1 event in millepede loop")

    # Calculating determinant
    determinant = np.linalg.det(matrixCPrime)
    if abs(determinant) < 1e-14:
        print("CPrime Matrix is Singular")
    elif determinant > 0:
        print("CPrime Matrix POS DEF!")
    else:
        print("CPrime Matrix NOT POS DEF ...")

    # Final matrix inversion
    matrixCPrimeInverted = inv(matrixCPrime)

    return matrixCPrimeInverted @ vectorBPrime
    
    #h.SaveAs(f"h{GEO_ID}.C")

#############################################################################
#################           RUN MILLE             ###########################
#############################################################################

ITERATION = 0 # Step of iteration
outputfilename = f"mp2meg2_MC_0509.bin"

# TTree with cosmics events for millepede
crtree = TChain("trk")
inputfiles = f"residuals_iter_{ITERATION}/outTrack_*.root"
mcinput = f"mc/MCTrack*minchi2.root"
#crtree.Add(mcinput)
crtree.Add("mc/MCTrack_0_minchi2_newGeo.root")
crtree.Add("mc/MCTrack_10000_minchi2_newGeo.root")
crtree.Add("mc/MCTrack_20000_minchi2_newGeo.root")
crtree.Add("mc/MCTrack_30000_minchi2_newGeo.root")
crtree.Add("mc/MCTrack_40000_minchi2_newGeo.root")
crtree.Add("mc/MCTrack_50000_minchi2_newGeo.root")
crtree.Add("mc/MCTrack_60000_minchi2_newGeo.root")
crtree.Add("mc/MCTrack_70000_minchi2_newGeo.root")
crtree.Add("mc/MCTrack_80000_minchi2_newGeo.root")
crtree.Add("mc/MCTrack_90000_minchi2_newGeo.root")
crtree.Add("mc/MCTrack_100000_minchi2_newGeo.root")
crtree.Add("mc/MCTrack_110000_minchi2_newGeo.root")
crtree.Add("mc/MCTrack_120000_minchi2_newGeo.root")
crtree.Add("mc/MCTrack_130000_minchi2_newGeo.root")
crtree.Add("mc/MCTrack_140000_minchi2_newGeo.root")

print(f"Data File opened... GEOMETRY ID = {GEO_ID}, MillePede Step = {ITERATION} ...")

#plot_survey(GEO_ID)


# Write parameter file for this GEO_ID
"""
write_parameter_file(GEO_ID)
write_constraint_file()

# Start Mille
t_start = time.time()
with open(outputfilename, "wb") as aFile:
    for ev in crtree:
        a_track = CRTrack(ev, GEO_ID)
        chi2 = a_track.chi2
        #print(f"chi2 = {chi2}")
        if chi2 > 1e-10 and chi2 < 2:
            a_track.write_to_binary_file(aFile)

t_stop = time.time()
print(f"{outputfilename} produced. Time needed {(t_stop - t_start)/3600 :.1f} h")
"""
#############################################################################################
#
#                                     MILLEPEDE IN MY OWN WAY
#
############################################################################################


# Create events
#t_start = time.time()
#events = [CRTrack(ev, GEO_ID) for ev in crtree]
t_stop = time.time()
#print(f"{len(events)} eventi letti... tempo impiegato {(t_stop - t_start)/3600 :.1f} h")

# Get the chi2 scan for the sagitta parameter of some wires
"""
chi2_scan_sag1009 = np.zeros(100)
chi21009 = []
count1009 = 0
chi2_scan_sag1234 = np.zeros(100)
count1234 = 0
chi21234 = []
chi2_scan_sag955 = np.zeros(100)
count955 = 0
chi2955 = []
chi2_scan_sag634 = np.zeros(100)
count634 = 0
chi2634 = []
chi2_scan_sag580 = np.zeros(100)
count580 = 0
chi2580 = []

for track in events:
    for hit in track.hits:
        if hit.chi2 < 3:
            if hit.wire == 1009:
                chi2_scan_sag1009 += hit.chi2_scan
                count1009 += 1
                chi21009.append(hit.chi2_scan.sum())
            if hit.wire == 1234:
                chi2_scan_sag1234 += hit.chi2_scan
                count1234 += 1
                chi21234.append(hit.chi2_scan.sum())
            if hit.wire == 955:
                chi2_scan_sag955 += hit.chi2_scan
                count955 += 1
                chi2955.append(hit.chi2_scan.sum())
            if hit.wire == 634:
                chi2_scan_sag634 += hit.chi2_scan
                count634 += 1
                chi2634.append(hit.chi2_scan.sum())
            if hit.wire == 580:
                chi2_scan_sag580 += hit.chi2_scan
                count580 += 1
                chi2580.append(hit.chi2_scan.sum())

#print(f"580 {count580}, 634 {count634}, 955 {count955}, 1234 {count1234}, 1009 {count1009}")

# Plot profile of chi2

plt.figure(1)
plt.subplot(121)
plt.title(r"$\chi^2$ scan wire 1009")
plt.xlabel(r"$\delta$sag [cm]")
plt.ylabel(r"$\chi^2$")
plt.plot(np.linspace(-0.01, 0.01, 100), chi2_scan_sag1009/count1009)
plt.subplot(122)
plt.hist(chi21009, bins=100)

plt.figure(2)
plt.subplot(121)
plt.title(r"$\chi^2$ scan wire 1234")
plt.xlabel(r"$\delta$sag [cm]")
plt.ylabel(r"$\chi^2$")
plt.plot(np.linspace(-0.01, 0.01, 100), chi2_scan_sag1234/count1234)
plt.subplot(122)
plt.hist(chi21234, bins=100)


plt.figure(3)
plt.subplot(121)
plt.title(r"$\chi^2$ scan wire 955")
plt.xlabel(r"$\delta$sag [cm]")
plt.ylabel(r"$\chi^2$")
plt.plot(np.linspace(-0.01, 0.01, 100), chi2_scan_sag955/count955)
plt.subplot(122)
plt.hist(chi2955, bins=100)

plt.figure(5)
plt.subplot(121)
plt.title(r"$\chi^2$ scan wire 634")
plt.xlabel(r"$\delta$sag [cm]")
plt.ylabel(r"$\chi^2$")
plt.plot(np.linspace(-0.01, 0.01, 100), chi2_scan_sag634/count634)
plt.subplot(122)
plt.hist(chi2634, bins=100)

plt.figure(6)
plt.subplot(121)
plt.title(r"$\chi^2$ scan wire 580")
plt.xlabel(r"$\delta$sag [cm]")
plt.ylabel(r"$\chi^2$")
plt.plot(np.linspace(-0.01, 0.01, 100), chi2_scan_sag580/count580)
plt.subplot(122)
plt.hist(chi2580, bins=100)

plt.show()
"""


# MillePede
outputfile_name = "mc_results_millepede_newGeo.txt"
try:
    results = millepede(GEO_ID, crtree)
    print("------------- MILLEPEDE INVERTED MATRIX ---------------- \n")
    print(results)
    # Write results
    with open(outputfile_name, "w") as f:
        if ALIGN_PARS == 5:
            f.write('#dx0 #dy0 #dth #dph #ds\n')
        elif ALIGN_PARS == 6:
            f.write('#dx0 #dy0 #dth #dph #dgamma #ds\n')
        for i in range(int(len(results)/ALIGN_PARS)):
            entry = ""
            if ALIGN_PARS == 5:
                entry = f'{results[i*5]} {results[i*5+1]} {results[i*5+2]} {results[i*5+3]} {results[i*5+4]}\n'
            if ALIGN_PARS == 6:
                entry = f'{results[i*6]} {results[i*6+1]} {results[i*6+2]} {results[i*6+3]} {results[i*6+4]} {results[i*6+5]}\n'
            f.write(entry)
    print(f"\n{outputfile_name} scritto!")

except np.linalg.LinAlgError:
    print('Singular Matrix in Millepede execution')

print("\n--------------------------------------------------------")

t_stop2 = time.time()
print(f'{((t_stop2 - t_stop)/3600):.1f} h per terminare MillePede')

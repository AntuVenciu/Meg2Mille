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
import time
import struct
import array

from ROOT import TFile, TChain
import matplotlib.pyplot as plt
import numpy as np
from symengine import lambdify, diff, symbols, sin, cos, Matrix, sqrt # Package for symbolic calc in Python. Based on C and C++
import sympy as sym # Package for symbolic calc in Python


####################### GEOMETRY DEFINITION OF THE CDCH #################################

# Definition of variables for sym
sym.init_printing(use_unicode=True)
# global parameters
x0, y0, z0, s, gamma, theta, phi, L, z_ds, z_us, yt, zi, ti, sigma_i = symbols("x0 y0 z0 s gamma theta phi L z_ds z_us yt zi ti sigma_i")
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
    
# Definition of hit residual
def res(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, yt, zi, ti, sigma_i):
    """
    Calculates the correction to global chi square from a given data point.
    """
    # parameterization of the track
    a_track = Matrix([mxy*yt + qxy,
                      yt,
                      myz*yt + qyz])
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
                    0.])
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

    return sqrt((a_track - a_wire).dot(a_track - a_wire)) - ti

# Chi squared function calculation
def chi2(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, yt, zi, ti, sigma_i):
    """
    Calculates the correction to global chi square from a given data point.
    """
    a_res = res(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, yt, zi, ti, sigma_i)
    return 0.5 * a_res**2 / sigma_i**2

#In order to make calculation fast, the chi 2 function and its derivatives must be lambdified.
fast_chi2 = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, yt, zi, ti, sigma_i], [chi2(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, yt, zi, ti, sigma_i)])
fast_res = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, yt, zi, ti, sigma_i], [res(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, yt, zi, ti, sigma_i)])
fast_der_x0 = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, yt, zi, ti, sigma_i], [diff(res(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, yt, zi, ti, sigma_i), x0)])
fast_der_y0 = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, yt, zi, ti, sigma_i], [diff(res(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, yt, zi, ti, sigma_i), y0)])
fast_der_z0 = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, yt, zi, ti, sigma_i], [diff(res(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, yt, zi, ti, sigma_i), z0)])
fast_der_theta = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, yt, zi, ti, sigma_i], [diff(res(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, yt, zi, ti, sigma_i), theta)])
fast_der_phi = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, yt, zi, ti, sigma_i], [diff(res(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, yt, zi, ti, sigma_i), phi)])
fast_der_gamma = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, yt, zi, ti, sigma_i], [diff(res(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, yt, zi, ti, sigma_i), gamma)])
fast_der_s = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, yt, zi, ti, sigma_i], [diff(res(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, yt, zi, ti, sigma_i), s)])
fast_der_L = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, yt, zi, ti, sigma_i], [diff(res(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, yt, zi, ti, sigma_i), L)])
fast_der_mxy = lambdify([x0, y0, z0,theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, yt, zi, ti, sigma_i], [diff(res(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, yt, zi, ti, sigma_i), mxy)])
fast_der_qxy = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, yt, zi, ti, sigma_i], [diff(res(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, yt, zi, ti, sigma_i), qxy)])
fast_der_myz = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, yt, zi, ti, sigma_i], [diff(res(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, yt, zi, ti, sigma_i), myz)])
fast_der_qyz = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, yt, zi, ti, sigma_i], [diff(res(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, yt, zi, ti, sigma_i), qyz)])

fast_wire_coord = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, zi], [wire_coord(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, zi)])

# lists storing derivatives
fast_der_align = [fast_der_x0,
                  fast_der_y0,
             	  fast_der_theta,
                  fast_der_phi,
                  fast_der_gamma,
                  fast_der_s]

fast_der_track = [fast_der_mxy,
             	  fast_der_qxy,
                  fast_der_myz,
                  fast_der_qyz]

########################################################################

####################### GLOBAL VARIABLES ################################

# Definition of dictionary to map wires to a dense vector
good_wires = np.loadtxt("wire_list.txt", dtype='int', unpack=True) #np.loadtxt("wire_list_CYLDCH35_def.txt", dtype='int', unpack=True)
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

class Wire():
    """
    Wire class: contains all the informations about the wire
    geometry and alignment parameters.
    The alignment parameters are taken from the wire_parameters_CYLDCH*.txt file,
    ID = 46 for survey, ID = xxx for alignment step.
    the wire number must be declared.
    """
    def __init__(self, ID, wire):
        self.alignpars = np.loadtxt(f"wire_parameters_CYLDCH{ID}.txt")[wire]

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
                 yt,
                 z_i,
                 di,
                 sigma) :
        this_wire = Wire(ID, wire)
        wire_params = this_wire.alignpars
        self.params = [*wire_params, mx, qx, mz, qz, yt, z_i, di, sigma]
        self.wire = wire
        self.res = fast_res(self.params)
        self.sigma = sigma
        self.z_t = yt
        self.der_align = [der(self.params) for der in fast_der_align]
        if not any(self.der_align):
            print("Found zeros in GLB derivatives!")
        self.der_track = [der(self.params) for der in fast_der_track]
        if not any(self.der_track):
            print("Found zeros in local derivatives!")

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
        		   y_track,
        		   z_wire,
        		   doca,
        		   sigma)
                     for wire, y_track, z_wire, doca, sigma in zip(event.wire,
        							   event.y_track,
        		                                           event.w_wire,
        			 				   event.doca,
        			 				   event.sigma)
                     if wire in good_wires] # comment this part to fix the reference of some wires

        # Array format to write on binary file for Pede routine
        self.glder = array.array('f')
        self.inder = array.array('i')
        self.glder.append(0.)
        self.inder.append(0)
        for hit in self.hits:
            self.glder.append(hit.res)
            self.inder.append(0)
            self.glder.fromlist(hit.der_track)
            self.inder.fromlist([1, 2, 3, 4]) # counts from 1 to N_FIT
            self.glder.append(hit.sigma)
            self.inder.append(0)
            self.glder.fromlist(hit.der_align)
            glb_label = [dict_wire_to_label[hit.wire]*ALIGN_PARS + i + 1 for i in range(ALIGN_PARS)] 
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
    with open("desy/millepede-ii/meg2params.txt", "w") as aFile:
        aFile.write("Parameter\n")
        for w in good_wires:
            wire = Wire(geom, w)
            params = [par for i, par in enumerate(wire.alignpars) if i == 0 or i == 1 or (i > 2 and i < 7)]
            for i, par in enumerate(params):
                label = dict_wire_to_label[w] * ALIGN_PARS + i + 1
                pre_sigma = 0.0
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

def write_constrain_file(geom):
    """
    Write constraints on parameters.
    For each plane, the global shift in x0 and y0 should be zero (for example).
    """
    with open("desy/millepede-ii/meg2const.txt", "w") as aFile:
        # x0 constraints
        for iplane in range(10):
            aFile.write("Constraint 0.0")
            for w in good_wires:
                if int(w/192) == iplane:
                    aFile.write(f"{dict_wire_to_label[w] * ALIGN_PARS + 1} 1.0")
        # y0 constraints
        for iplane in range(10):
            aFile.write("Constraint 0.0")
            for w in good_wires:
                if int(w/192) == iplane:
                    aFile.write(f"{dict_wire_to_label[w] * ALIGN_PARS + 2} 1.0")

def write_measurement_file():
    """
    Write survey measurements to be used as gaussian constraints
    on parameters. 50 um sigma.
    """
    with open("meg2meas.txt", "w") as aFile:
        id_wire, x_s, y_s, z_s, w_s = np.loadtxt("anode_CYLDCH46.txt", unpack=True)
        for wire, x, y, z, w in zip(id_wire, x_s, y_s, z_s, w_s):
            if int(wire) in good_wires:
                wire_pars = Wire(66, int(wire)).alignpars
                # x measurement             
                aFile.write(f"Measurement {x} {0.005}")
                # y measurement
                # z measurement
                
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
    id_wire, x_s, y_s, z_s, w_s = np.loadtxt("anode_CYLDCH46.txt", unpack=True)
    for wire, x, y, z, w in zip(id_wire, x_s, y_s, z_s, w_s):
        if int(wire) in good_wires:
            event = [CRHit(66,
                           int(wire),
                           0.,
                           x,
                           0.,
                           z,
                           y,
                           w,
                           0,
                           0.005)]
            res.append(event[0].res)
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


def plot_geometry(geom, min_wire=192, max_wire=1920, plot_survey=True):
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
    if plot_survey:
        id_wire, x_s, y_s, z_s, w_s = np.loadtxt('anode_CYLDCH46.txt', unpack=True) 
        ax.scatter(x_s[w], y_s[w], z_s[w], marker='.', color='red', alpha=.3)

    plt.show()
        

###########################################################################


######################### MILLEPEDE ALGEBRA ###############################
#        Calculating matrices for MillePede.
#        The formulas are taken from the article on CMS alignment
###########################################################################

def calculateGamma(event):
    """
    Gamma is the matrix containint dervatives with
    respect to the track parameters.
    """
    matrixGamma = np.zeros(shape=(N_FIT, N_FIT))
    for hit in event:
        rows = np.array([der_i*hit.der_track/(hit.sigma**2) for der_i in hit.der_track])
        matrixGamma += rows
    return matrixGamma

def calculateG(event):
    """
    G is the matrix containing dervatives with
    respect to both track and alignment parameters.
    """
    matrixG = np.zeros(shape=(N_ALIGN, N_FIT))
    for hit in event:
        if hit.wire in good_wires:
            wire = dict_wire_to_idx[hit.wire]
            rows = np.array([der_i * hit.der_track / (hit.sigma**2) for der_i in hit.der_align])
            matrixG[wire*ALIGN_PARS : wire*ALIGN_PARS + ALIGN_PARS, : ] += rows

    return matrixG

def calculateB(event):
    """
    B is the vector containing the misalignment.
    """
    vectorB = np.zeros(N_ALIGN)
    for hit in event:
        if hit.wire in good_wires:
            wire = dict_wire_to_idx[hit.wire]
            vectorB[wire*ALIGN_PARS : wire*ALIGN_PARS + ALIGN_PARS] += hit.der_align * hit.res / (hit.sigma**2)

    return vectorB

def calculateBeta(event):
    """
    Beta is the vector containing the track residuals.
    """
    vectorBeta = np.zeros(N_FIT)
    for hit in event:
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
        if hit.wire in good_wires:
            wire = dict_wire_to_idx[hit.wire]
            rows = np.array([der_i*hit.der_align / (hit.sigma**2) for der_i in hit.der_align])
            matrixC[wire*ALIGN_PARS : wire*ALIGN_PARS + ALIGN_PARS, wire*ALIGN_PARS : wire*ALIGN_PARS + ALIGN_PARS] += rows
            #print(f"Updated C in position of wire {wire} with {rows}")
    return matrixC

def calculateCPrime(matrixCPrime, event):
    """
    CPrime is the matrix to invert, with both misalignment
    and track fitting parameters.
    """
    G = calculateG(event)
    C = calculateC(event)
    try:
        Gamma_inv = np.linalg.inv(calculateGamma(event))
        matrixCPrime += C - (G @ Gamma_inv) @ G.T
    except np.linalg.LinAlgError:
        print("Error in Gamma inversion...")
        matrixCPrime += C
    return matrixCPrime

def calculateBPrime(vectorBPrime, event):
    """
    Is the vector with the track and alignment residuals.
    """
    G = calculateG(event)
    b = calculateB(event)
    beta = calculateBeta(event)
    try:
        Gamma_inv = np.linalg.inv(calculateGamma(event))
        vectorBPrime += b - G @ (Gamma_inv @ beta)
    except np.linalg.LinAlgError:
        print("Error in Gamma inversion...")
        vectorBPrime += b
    return vectorBPrime

def survey(geom, matrixCPrime):
    """
    Insert in the C matrix the survey measurements.
    These are not the proper measurements, but are (x, y, z) points at w = +/- 95 cm
    obtained from software computation with CYLDCH 46.
    50 um error on measurements.
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
                           y,
                           w,
                           0,
                           0.005)]
            C = calculateC(event)
            matrixCPrime += C

    return matrixCPrime

def millepede(geom, data):
    """
    This function computes and inverts CPrime
    hence it solve the alignment problem.
    """
    # Initialize matrices and vectors
    matrixCPrime = np.zeros(shape=(N_ALIGN, N_ALIGN))
    vectorBPrime = np.zeros(N_ALIGN)
    # Insert survey measurements 
    t_start = time.time()
    matrixCPrime = survey(geom, matrixCPrime)
    t_stop = time.time()
    print(f"Survey inserted in {((t_stop - t_start)/3600):.2f} h...")
    # Loop on event to fill the matrices
    for i, event in enumerate(data):
        t_start = time.time()
        matrixCPrime = calculateCPrime(matrixCPrime, event.hits)
        vectorBPrime = calculateBPrime(vectorBPrime, event.hits)
        t_stop = time.time()
        if i%10000 == 0:
            print(f"Evento {i}")

    # Calculating determinant
    determinant = np.linalg.det(matrixCPrime)
    if abs(determinant) < 1e-14:
        print("CPrime Matrix is Singular")
    elif determinant > 0:
        print("CPrime Matrix POS DEF!")
    else:
        print("CPrime Matrix NOT POS DEF ...")

    # Final matrix inversion
    matrixCPrimeInverted = np.linalg.inv(matrixCPrime)

    return matrixCPrimeInverted @ vectorBPrime


#############################################################################
#################           RUN MILLE             ###########################
#############################################################################

GEO_ID = 66 # Geometry ID to start millepede
ITERATION = 0 # Step of iteration
outputfilename = f"desy/millepede-ii/mp2meg2_{ITERATION}.bin"

# TTree with cosmics events for millepede
crtree = TChain("trk")
crtree.Add(f"residuals_iter_{ITERATION}/outTrack_437*.root")
print(f"Data File opened... GEOMETRY ID = {GEO_ID}, MillePede Step = {ITERATION} ...")

# Write parameter file for this GEO_ID
write_parameter_file(GEO_ID)

# Start Mille
t_start = time.time()
with open(outputfilename, "ab") as aFile:
    for ev in crtree:
        CRTrack(ev, GEO_ID).write_to_binary_file(aFile)
t_stop = time.time()
print(f"{outputfilename} produced. Time needed {(t_stop - t_start)/3600 :.1f} h") 

#############################################################################################
#
#                                     MILLEPEDE IN MY OWN WAY
#
############################################################################################
"""

# Create events
t_start = time.time()
events = [CRTrack(ev, GEO_ID) for ev in crtree]
t_stop = time.time()
print(f"{len(events)} eventi letti... tempo impiegato {(t_stop - t_start)/3600 :.1f} h")

# MillePede
    
try:
    results = millepede(GEO_ID, events)
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
"""

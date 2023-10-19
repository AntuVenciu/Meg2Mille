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
gInterpreter.ProcessLine('#include "cosmics_includes_newGeo.h"')

import matplotlib.pyplot as plt
from numpy.linalg import inv, norm
import numpy as np
from scipy.optimize import curve_fit
from symengine import lambdify, diff, symbols, sin, cos, Matrix, sqrt # Package for symbolic calc in Python. Based on C and C++
import sympy as sym # Package for symbolic calc in Python


####################### GEOMETRY DEFINITION OF THE CDCH #################################

# Definition of variables for sym
sym.init_printing(use_unicode=True)
# global parameters
x0, y0, z0, s, gamma, theta, phi, L, z_ds, z_us, zi, ti, sigma_i = symbols("x0 y0 z0 s gamma theta phi L z_ds z_us zi ti sigma_i")
# local parameters
mxy, qxy, myz, qyz = symbols("mxy qxy myz qyz")


def der_mxy(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i):
    """
    Exact derivative of chi2 with respect to mxy
    """
    w = np.array([np.cos(phi)*np.sin(theta),
                  np.sin(phi)*np.sin(theta),
                  np.cos(theta)])
    w0 = np.array([x0,
                   y0,
                   z0])
    x0 = np.array([qxy,
                   0.,
                   qyz])
    l = np.array([mxy,
                  1.,
                  myz])
    n = np.cross(w, l)
    n_mag = norm(n)
    diff_pos = w0 - x0
    numerator = np.dot(n, diff_pos)
    doca = np.sqrt((numerator / n_mag)**2)
    #print(f"doca = {doca:.6f} - dmeas = {ti:.6f}")
    residual = (ti - doca)/sigma_i
    derivative = (1./doca) * ((numerator*(w[2]*diff_pos[1] - w[1]*diff_pos[2])*n_mag**2 - (numerator**2)*(n[1]*w[2] - n[2]*w[1]))/(n_mag**4))
    return - 2. * residual * (1. / sigma_i) * derivative

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
    """
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
    """
    pos0 = wirepos #+ sag * sag_v

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
    Use only a linear geometry for the wire.
    """
    # parameterization of the track
    line_vector = Matrix([mxy,
                          1,
                          myz])
    line_origin = Matrix([qxy,
                          0,
                          qyz])
    # Point on wire and wire vector
    wire_vector = Matrix([cos(phi)*sin(theta),
                    sin(phi)*sin(theta),
                    cos(theta)])
    point_on_wire = Matrix([x0,
                            y0,
                            z0])#wire_coord(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, zi)
    # The distance of closest approach between wire and line is given by:
    # doca = |n . (point_on_line - point_on_wire)| / |n|
    # n is the vector perpendicular to both line and wire: n = wire_vector x line_vector
    n = Matrix([sin(phi)*sin(theta)*myz - cos(theta),
                cos(theta)*mxy - myz * cos(phi)*sin(theta),
                cos(phi)*sin(theta) - sin(phi)*sin(theta)*mxy])
    n_mag2 = n[0]*n[0] + n[1]*n[1] + n[2]*n[2]
    n_points = n[0] * (point_on_wire[0] - line_origin[0]) + n[1] * (point_on_wire[1] - line_origin[1]) + n[2] * (point_on_wire[2] - line_origin[2])
    doca = sqrt((n_points)*(n_points) / (n_mag2))

    # Return the hit residual
    return (ti - doca) / sigma_i

# Chi squared function calculation
def chi2(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i):
    """
    Calculates the correction to global chi square from a given data point.
    """
    a_res = res(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i)
    return a_res**2

#In order to make calculation fast, the chi 2 function and its derivatives must be lambdified.
fast_chi2 = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i], [chi2(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i)])
fast_x = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, zi], [x_wire_coord(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, zi)])
fast_y = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, zi], [y_wire_coord(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, zi)])
fast_z = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, zi], [z_wire_coord(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, zi)])
fast_res = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i], [res(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i)])
fast_der_x0 = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i], [diff(chi2(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i), x0, 1)])
fast_der_y0 = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i], [diff(chi2(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i), y0, 1)])
fast_der_z0 = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i], [diff(chi2(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i), z0, 1)])
fast_der_theta = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i], [diff(chi2(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i), theta, 1)])
fast_der_phi = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i], [diff(chi2(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i), phi, 1)])
fast_der_gamma = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i], [diff(chi2(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i), gamma, 1)])
fast_der_s = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i], [diff(chi2(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i), s, 1)])
fast_der_mxy = lambdify([x0, y0, z0,theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i], [diff(res(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i), mxy, 1)])
fast_der_qxy = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i], [diff(res(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i), qxy, 1)])
fast_der_myz = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i], [diff(res(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i), myz, 1)])
fast_der_qyz = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i], [diff(res(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, mxy, qxy, myz, qyz, zi, ti, sigma_i), qyz, 1)])

fast_wire_coord = lambdify([x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, zi], [wire_coord(x0, y0, z0, theta, phi, gamma, s, L, z_ds, z_us, zi)])

# lists storing derivatives:
# do not use s for line wires
# do not use gamma at first iteration, when wires are initially straight
fast_der_align = [fast_der_x0,
                  fast_der_y0,
             	  fast_der_theta,
                  fast_der_phi]
                  #fast_der_gamma,
                  #fast_der_s]

fast_der_track = [fast_der_mxy,
             	  fast_der_qxy,
                  fast_der_myz,
                  fast_der_qyz]

def draw_scan_wire_geo(hit, par_idx):
    
    #Draw the scanning of the chi2 function as a function of the mxy parameter
    pars_labels = {0:"x0", 1:"y0", 3:"theta", 4:"phi"}
    pars_idx_list = {0:0, 1:1, 3:2, 4:3}
    p0 = (hit.get_params())[par_idx]
    print(f"Initial value of {pars_labels[par_idx]} = {p0:.6f}")
    dm = np.linspace(-0.001 * p0 + p0, 0.001 * p0 + p0, 100)
    chi2_vec = []
    true_chi2 = hit.chi2()
    der = hit.der_align()[pars_idx_list[par_idx]]
    print(f"chi2 = {true_chi2:.2f},  d{pars_labels[par_idx]} = {der:.4f}")
    der_vec = []
    
    for x in dm:
        parameters = hit.get_params()
        parameters[par_idx] = x
        chi2_vec.append(fast_chi2(parameters))
        der_vec.append(fast_der_align[pars_idx_list[par_idx]](parameters))
    
    plt.figure()
    plt.subplot(211)
    plt.errorbar(dm, np.array(chi2_vec), fmt='.', color='blue')
    plt.errorbar(p0, true_chi2, fmt='*', color='red', label='Best fit')
    plt.plot(dm, np.array(chi2_vec), linestyle='--', color='orange')
    plt.grid(True)
    plt.xlabel(f"{pars_labels[par_idx]}")
    plt.subplot(212)
    plt.errorbar(dm, np.array(der_vec), fmt='.', linestyle='dotted', color='black')
    plt.errorbar(p0, der, fmt='*', color='red')
    plt.grid(True)
    plt.xlabel(f"{pars_labels[par_idx]}")
    plt.legend()

    plt.show()
    
    return

########################################################################

####################### GLOBAL VARIABLES ################################

# Definition of dictionary to map wires to a dense vector
good_wires = np.loadtxt("MC_wire_list_fix_cosmics.txt", dtype='int', unpack=True) #np.loadtxt("wire_list_CYLDCH35_def.txt", dtype='int', unpack=True)
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

list_wires_parameters = np.loadtxt(f"linewire_parameters_CYLDCH{GEO_ID}.txt")

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
        print("I am initializing this hit on wire: ", wire )
        self._this_wire = Wire(ID, wire)
        self._wire_params = self._this_wire.alignpars
        self._params = [*self._wire_params, mx, qx, mz, qz, z_i, di, sigma]
        self._wire = wire
        self._sigma = sigma
        self._zi = z_i
    
    # Getters and setters
    # Wire
    def get_wire(self):
        return self._wire

    def set_wire(self, wire):
        self._wire = wire

    # Params
    def get_params(self):
        return self._params
    
    def set_params(self, parameters):
        self._params = parameters

     # Z
    def get_zi(self):
        return self._zi

    def set_zi(self, z):
        self._zi = z

    # Sigma
    def get_sigma(self):
        return self._sigma

    # Methods to compute properties of the hit

    # Hit residual
    def res(self):
        return fast_res(self._params)
    
    # Chi2
    def chi2(self):
        return fast_chi2(self._params)
    
    # Derivatives
    def der_align(self):
        ders = [der(self._params) for der in fast_der_align]
        return ders

    def der_track(self):
        return [der(self._params) for der in fast_der_track]

    def der_test(self):
        return der_mxy(*self._params)
    
    # Wire coordinates
    def x_wire(self):
        return fast_x([*self._wire_params, self._zi])
    
    def y_wire(self):
        return fast_y([*self._wire_params, self._zi])
    
    def z_wire(self):
        return fast_z([*self._wire_params, self._zi])
    

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
        		   event.GetLeaf("mxy").GetValue(),
        		   event.GetLeaf("qxy").GetValue(),
        		   event.GetLeaf("myz").GetValue(),
        		   event.GetLeaf("qyz").GetValue(),
        		   0.,
        		   doca,
        		   sigma)
                     for wire, doca, sigma in zip(event.wire,
                                                  event.doca,
                                                  event.sigma)]
        #print(f'Track pars : xi = {event.GetLeaf("mxy").GetValue():.6f}, x0 = {event.GetLeaf("qxy").GetValue():.6f}, eta = {event.GetLeaf("myz").GetValue():.6f}, z0 = {event.GetLeaf("qyz").GetValue():.6f}')
                     #if wire in good_wires] # comment this part to fix the reference of some wires
        if len(self.hits) < 5:
            self.chi2 = 1e30
        else:
            self.chi2 = np.array([hit.chi2() for hit in self.hits]).sum()/(len(self.hits) - 4.) # total chi2/dof
        self.der_glb_check = np.array([hit.der_align()[3] for hit in self.hits]).sum()
        # Array format to write on binary file for Pede routine
        self.glder = array.array('f')
        self.inder = array.array('i')
        self.glder.append(0.)
        self.inder.append(0)
        if False:
            for hit in self.hits:
                self.glder.append(hit.res())
                self.inder.append(0)
                self.glder.fromlist(hit.der_track())
                self.inder.fromlist([1, 2, 3, 4]) # counts from 1 to N_FIT
                self.glder.append(hit.get_sigma())
                self.inder.append(0)
                self.glder.fromlist(hit.der_align())
                glb_label = [hit.get_wire()*ALIGN_PARS + i + 1 for i in range(ALIGN_PARS)] #[dict_wire_to_label[hit.wire]*ALIGN_PARS + i + 1 for i in range(ALIGN_PARS)]
                self.inder.fromlist(glb_label)

    def der_glb(self, par_id):
        der = 0
        print(f"ID for glb der = {par_id}")
        for hit in self.hits:
            der += hit.der_align()[par_id]
            print(f"Der align {par_id} for a hit = {hit.der_align()[par_id]}")
        return der
    

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
            res.append(event.res())
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
            if hit.chi2() < 5e1:
                rows = np.array([der_i*hit.der_track() for der_i in hit.der_track()])
                matrixGamma += rows
                del rows
                #gc.collect()
        return matrixGamma

def calculateG(event):
        # matrix G
        matrixG = np.zeros(shape=(N_ALIGN, N_FIT))
        for hit in event:
            if hit.chi2() < 5e1:
                if hit.get_wire() in good_wires:
                    wire = dict_wire_to_idx[hit.get_wire()]
                    rows = np.array([der_i * hit.der_track() for der_i in hit.der_align()])
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
        if hit.chi2() < 5e1:
            if hit.get_wire() in good_wires:
                wire = dict_wire_to_idx[hit.get_wire()]
                vectorB[wire*ALIGN_PARS : wire*ALIGN_PARS + ALIGN_PARS] += hit.der_align() * hit.res()
                del wire
                #gc.collect()
    return vectorB

def calculateBeta(event):
    """
    Beta is the vector containing the track residuals.
    """
    vectorBeta = np.zeros(N_FIT)
    for hit in event:
        if hit.chi2() < 5e1:
            #if hit.wire in good_wires: #uncomment if no wires is fixed
            vectorBeta += hit.der_track() * hit.res()
    return vectorBeta

def calculateC(event):
    """
    C is the matrix with derivatives with respect
    to the alignment parameters.
    """
    matrixC = np.zeros(shape=(N_ALIGN, N_ALIGN))
    for hit in event:
        if hit.chi2() < 5e1:
            if hit.get_wire() in good_wires:
                wire = dict_wire_to_idx[hit.get_wire()]
                rows = np.array([der_i*hit.der_align() for der_i in hit.der_align()])
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
    for i, entry in enumerate(tree):
        #tstartloop = time.time()
        event = CRTrack(entry, geom)
        print(f"chi2 tot = {event.chi2:.2f}")
        ahit = event.hits[0]
        draw_scan_wire_geo(ahit, 4)
        draw_scan_wire_geo(ahit, 0)
        draw_scan_wire_geo(ahit, 3)
        draw_scan_wire_geo(ahit, 4)
        #print(f"dphi0 = {event.der_glb(3):.4f}")
        #print(f"dphi0 check = {event.der_glb_check:.4f}")
        ##h.Fill(event.chi2)
        if event.chi2 > 10:
            print(f"chi2 = {event.chi2:.4f}. Event skipped.")
        #print(f"chi2 = {event.chi2:.4f}")
        if i%10000 == 0:
            print(f"Evento {i}")
        matrixCPrime = calculateCPrime(matrixCPrime, event.hits)
        vectorBPrime = calculateBPrime(vectorBPrime, event.hits)
        del event

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


#############################################################################
#################           RUN MILLE             ###########################
#############################################################################

ITERATION = 0 # Step of iteration
print(f"Data File opened... GEOMETRY ID = {GEO_ID}, MillePede Step = {ITERATION} ...")

outputfilename = f"mp2meg2_MC_0509.bin"

# TTree with cosmics events for millepede
crtree = TChain("trk")
inputfiles = f"residuals_iter_{ITERATION}/outTrack_*.root"
mcinput = f"mc/MCTrack_lineWire*.root"
#crtree.Add(mcinput)
crtree.Add("mc/MCTrack_lineWire_test.root")
#crtree.Add("mc/MCTrack_lineWire_2*.root")
#crtree.Add("mc/MCTrack_lineWire_3*.root")

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

# MillePede
outputfile_name = f"mc_results_millepede_lineWire12Precision_noSurvey_fixed320-327-520-527-240-247-440-447_chi210_events200k.txt"
try:
    results = millepede(GEO_ID, crtree)
    print("------------- MILLEPEDE INVERTED MATRIX ---------------- \n")
    print(results)
    # Write results
    with open(outputfile_name, "w") as f:
        if ALIGN_PARS==4:
            f.write('#dx0 #dy0 #dth #dph\n')
        if ALIGN_PARS == 5:
            f.write('#dx0 #dy0 #dth #dph #ds\n')
        elif ALIGN_PARS == 6:
            f.write('#dx0 #dy0 #dth #dph #dgamma #ds\n')
        for i in range(int(len(results)/ALIGN_PARS)):
            entry = ""
            if ALIGN_PARS == 4:
                entry = f'{results[i*4]} {results[i*4+1]} {results[i*4+2]} {results[i*4+3]}\n'
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

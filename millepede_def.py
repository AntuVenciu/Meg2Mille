#!/usr/bin/env python3
"""
Author: Antoine Venturini
Date: 2022

Write MillePede files for MEGII drift chamber
alignment with cosmic rays.
Use autograd to calculate derivatives of geometry
and track parameters.
"""
import time
import array

from ROOT import TChain, gInterpreter
# Load trk library
gInterpreter.ProcessLine('#include "cosmics_includes_newGeo.h"')

import matplotlib.pyplot as plt
from numpy.linalg import inv
import numpy as np
from jax import grad
import jax.numpy as jnp


#######################################################################
####################### GEOMETRY DEFINITION OF THE CDCH ###############
#######################################################################

# Definition of hit residual
def res(x0, y0, z0, theta, phi, mxy, qxy, myz, qyz, ti, sigma_i):
    """
    The residual with respect to the track,
    when wires are considered as straight lines
    are calculated as res = (d_meas - d_i)/sigma_i
    where d_i = distance of closest approach between cosmic ray
    and wire.

    (Straight) Wires are parameterized by 4 geometrical variables:
    - x0, y0: position x, y of the wire @ z = 0
    - theta, phi : spherical angles for the orientation of the wire
    A straight track is parameterized as:
        l = (qxy, 0, qyz) + y * (mxy, 1, myz)
    """
    # parameterization of the track
    line_vector = jnp.array([mxy, 1, myz])
    line_origin = jnp.array([qxy, 0, qyz])
    # Point on wire and wire vector
    wire_vector = jnp.array([jnp.cos(phi)*jnp.sin(theta), jnp.sin(phi)*jnp.sin(theta), jnp.cos(theta)])
    point_on_wire = jnp.array([x0, y0, z0])
    # The distance of closest approach between wire and line is given by:
    # doca = |n . (point_on_line - point_on_wire)| / |n|
    # n is the vector perpendicular to both line and wire: n = wire_vector x line_vector
    n = jnp.cross(wire_vector, line_vector)
    n_mag2 = n[0]*n[0] + n[1]*n[1] + n[2]*n[2]
    n_points = n[0] * (point_on_wire[0] - line_origin[0]) + n[1] * (point_on_wire[1] - line_origin[1]) + n[2] * (point_on_wire[2] - line_origin[2])
    doca = jnp.sqrt((n_points)*(n_points) / (n_mag2))

    # Return the hit residual
    return (ti - doca) / sigma_i

# Chi squared function calculation
def chi2(x0, y0, z0, theta, phi, mxy, qxy, myz, qyz, ti, sigma_i):
    """
    Calculates the correction to global chi square from a given data point.
    """
    a_res = res(x0, y0, z0, theta, phi, mxy, qxy, myz, qyz, ti, sigma_i)
    return a_res**2

# Compute derivatives
der_x0 = grad(res, 0)
der_y0 = grad(res, 1)
der_th = grad(res, 3)
der_ph = grad(res, 4)
der_mxy = grad(res, 5)
der_qxy = grad(res, 6)
der_myz = grad(res, 7)
der_qyz = grad(res, 8)
der_align = [der_x0, der_y0, der_th, der_ph]
der_track = [der_mxy, der_qxy, der_myz, der_qyz]

########################################################################
####################### GLOBAL VARIABLES ###############################
########################################################################

# Definition of dictionary to map wires to a dense vector
good_wires = np.loadtxt("MC_wire_list_fix_cosmics.txt", dtype='int', unpack=True)
idx = np.arange(0, len(good_wires), 1)
label = np.arange(0, len(good_wires), 1)
dict_wire_to_label = dict(zip(good_wires, label))
dict_wire_to_idx = dict(zip(good_wires, idx))
dict_idx_to_wire = dict(zip(idx, good_wires))
# Number of global and local parameters
N_WIRES = len(good_wires)
ALIGN_PARS = 4
N_ALIGN = int(N_WIRES * ALIGN_PARS) # total number of align. parameters
N_FIT = 4 # number of fit parameters to a line in 3D

###########################################################################


####################### CLASSES FOR DATA HANDLING #########################

GEO_ID = 69 # Geometry ID to start millepede
list_wires_parameters = np.loadtxt(f"wire_params_CYLDCH{GEO_ID}.txt")

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
                 di,
                 sigma) :
        self._this_wire = Wire(ID, wire)
        self._wire_params = self._this_wire.alignpars
        self._params = [*self._wire_params, mx, qx, mz, qz, di, sigma]
        self._wire = wire
        self._sigma = sigma
        self.der_align = np.array([der(*self._params) for der in der_align])
        self.der_track = np.array([der(*self._params) for der in der_track])
        self.chi2 = chi2(*self._params)
        self.res = res(*self._params)
    
    # Getters and setters
    # Wire
    def get_wire(self):
        return self._wire

    def set_wire(self, wire):
        self._wire = wire
    """
    # Params
    def get_params(self):
        return self._params
    
    def set_params(self, parameters):
        self._params = parameters

    # Sigma
    def get_sigma(self):
        return self._sigma

    # Methods to compute properties of the hit

    # Hit residual
    def res(self):
        return res(*self._params)
    
    # Chi2
    def chi2(self):
        return chi2(*self._params)
    
    # Derivatives
    def der_align(self):
        ders = [der(*self._params) for der in der_align]
        return np.array(ders)

    def der_track(self):
        return np.array([der(*self._params) for der in der_track])
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
        		   event.GetLeaf("mxy").GetValue(),
        		   event.GetLeaf("qxy").GetValue(),
        		   event.GetLeaf("myz").GetValue(),
        		   event.GetLeaf("qyz").GetValue(),
        		   doca,
        		   sigma)
                     for wire, doca, sigma in zip(event.wire,
                                                  event.doca,
                                                  event.sigma)]

        if len(self.hits) < 5:
            self.chi2 = 1e30
        else:
            self.chi2 = np.array([hit.chi2 for hit in self.hits]).sum()/(len(self.hits) - 4.) # total chi2/dof
        self.der_glb_check = np.array([hit.der_align[3] for hit in self.hits]).sum()
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
                glb_label = [hit.get_wire()*ALIGN_PARS + i + 1 for i in range(ALIGN_PARS)]
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

###########################################################################
######################### MILLEPEDE ALGEBRA ###############################
#        Calculating matrices for MillePede.                              #
#        The formulas are taken from the article on CMS alignment         #
###########################################################################
###########################################################################

def calculateGamma(event):
        # matrix Gamma
        matrixGamma = np.zeros(shape=(N_FIT, N_FIT))
        for hit in event:
            if hit.chi2 < 5e10:
                der_track = hit.der_track
                for i, der_i in enumerate(der_track):
                    for j, der_j in enumerate(der_track):
                        matrixGamma[i, j] += der_i * der_j
        return matrixGamma

def calculateG(event):
        # matrix G
        matrixG = np.zeros(shape=(N_ALIGN, N_FIT))
        for hit in event:
            if hit.chi2 < 5e10:
                if hit.get_wire() in good_wires:
                    wire = dict_wire_to_idx[hit.get_wire()]
                    der_align = hit.der_align
                    der_track = hit.der_track
                    for i, der_a in enumerate(der_align):
                        for j, der_t in enumerate(der_track):
                            matrixG[i + wire, j] += der_a * der_t
        return matrixG

def calculateB(event):
    """
    B is the vector containing the misalignment.
    """
    vectorB = np.zeros(N_ALIGN)
    for hit in event:
        if hit.chi2 < 5e10:
            if hit.get_wire() in good_wires:
                wire = dict_wire_to_idx[hit.get_wire()]
                residual = hit.res
                der_align = hit.der_align
                for i, der_a in enumerate(der_align):
                    vectorB[wire + i] += residual * der_a
    return vectorB

def calculateBeta(event):
    """
    Beta is the vector containing the track residuals.
    """
    vectorBeta = np.zeros(N_FIT)
    for hit in event:
        if hit.chi2 < 5e10:
            residual = hit.res
            der_track = hit.der_track
            for i, der in enumerate(der_track):
                vectorBeta[i] += der * residual
    return vectorBeta

def calculateC(event):
    """
    C is the matrix with derivatives with respect
    to the alignment parameters.
    """
    matrixC = np.zeros(shape=(N_ALIGN, N_ALIGN))
    for hit in event:
        if hit.chi2 < 5e10:
            if hit.get_wire() in good_wires:
                wire = dict_wire_to_idx[hit.get_wire()]
                der_align = hit.der_align
                for i, der_i in enumerate(der_align):
                    for j, der_j in enumerate(der_align):
                        matrixC[wire + i, wire + j] += der_i * der_j
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

"""
def survey(geom, matrixCPrime):
    
    Insert in the C matrix the survey measurements.
    These are not the proper measurements, but are (x, y, z) points at w = +/- 95 cm
    obtained from software computation with CYLDCH 46.
    50 um error on measurements.
    

    id_wire = np.linspace(0, 1919, 1920, dtype='int')
    z_survey = np.array([-95., 95.])
    for id in id_wire:
        if id not in good_wires:
            continue
        for z in z_survey:
            list_wires_parameters_survey = np.loadtxt("linewire_parameters_CYLDCH46.txt")
            wire_params = list_wires_parameters_survey[id]
            x_survey = fast_x([*wire_params, z])
            y_survey = fast_y([*wire_params, z])
            z_surv = fast_z([*wire_params, z])
            #print(f"Survey point wire {id} at {z} cm : ({x_survey:.2f}, {y_survey:.2f}, {z_surv:.2f})")
            event = [CRHit(geom,
                          id,
                          0.,
                          x_survey + np.random.normal(loc=0., scale=0.0001),
                          0.,
                          z_surv + np.random.normal(loc=0., scale=0.0001),
                          z,
                          0.,
                          0.005)]

            C = calculateC(event)
            matrixCPrime += C
            del event, C
            #gc.collect()

    return matrixCPrime
"""

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
        #print(f"chi2 tot = {event.chi2:.2f}")
        #ahit = event.hits[0]
        #draw_scan_wire_geo(ahit, 4)
        #draw_scan_wire_geo(ahit, 0)
        #draw_scan_wire_geo(ahit, 3)
        #draw_scan_wire_geo(ahit, 4)
        #print(f"dphi0 = {event.der_glb(3):.4f}")
        #print(f"dphi0 check = {event.der_glb_check:.4f}")
        ##h.Fill(event.chi2)
        if event.chi2 > 1000:
            #print(f"chi2 = {event.chi2:.4f}. Event skipped.")
            continue
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

outputfilename = f"mp2meg2_MC_linWires.bin"

# TTree with cosmics events for millepede
crtree = TChain("trk")
inputfiles = f"residuals_iter_{ITERATION}/outTrack_*.root"
mcinput = f"mc/MCTrack_lineWire*.root"
#crtree.Add(mcinput)
crtree.Add("mc/rec_cosmics_69_1*.root")
#crtree.Add("mc/rec_cosmics_69_2*.root")
#crtree.Add("mc/rec_cosmics_69_3*.root")
#crtree.Add("mc/MCTrack_lineWire_4*.root")

# Write parameter file for this GEO_ID
"""
write_parameter_file(GEO_ID)
write_constraint_file()

# Start Mille
t_start = time.time()
with open(outputfilename, "wb") as aFile:
    for ev in crtree:
        a_track = CRTrack(ev, GEO_ID)
        #print(f"chi2 = {chi2}")
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
outputfile_name = f"mc_results_autograd_lineWire_Misaligned69_NoSurvey_FixedWires_chi2cut10_events300k.txt"
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


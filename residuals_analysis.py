"""
Draw residuals from cosmic track fit at each millepede iteration
"""
import os
import glob

import numpy as np
import matplotlib.pyplot as plt


WRITE_WIRE_LIST = 0

# iteration of millepede
ITERATION = 0
directory = f"residuals_iter_{ITERATION}"

# residuals and wire list
wires = np.array([])
dres = np.array([])
zvec = np.array([])

# Loop over all files res_437*.txt in the directory
for res_file in glob.iglob(f"{directory}/res_4*.txt"):
    doca, res, sigma, x, y, z, wire = np.loadtxt(res_file, unpack=True)
    wires = np.append(wires, wire)
    dres = np.append(dres, res)
    zvec.append(zvec, z)

if WRITE_WIRE_LIST:
    # Count hits on wire to establish good wires (with > 50 hits) to be aligned
    wire_list = np.zeros(1920)
    for w in wires:
        wire_list[int(w)] += 1
    with open(f"wire_list_iter{ITERATION}.txt", "w") as f:
        for w, entries in enumerate(wire_list):
            if entries > 50:
                f.write(f"{w}\n")
    print(f"Created file wire_list_iter{ITERATION}.txt")

# Create the histograms of residuals per layer and per wire
plt.figure(1)
for i in range(1, 10, 1):
    plt.subplot(3, 3, i)
    plt.xlim(0, 192)
    plt.ylim(-0.1, 0.1)
    plt.hist2d(wires[wires%10 == i]%192, dres[wires%10 == i], bins=[192, 100], range=[[-0.5, 192.5], [-0.1, 0.1]])

# Histogram of residuals VS z for four "average" wires
plt.figure(2)
wire_list = np.zeros(1920)
for w in wires:
    wire_list[int(w)] += 1
mean = np.mean(wire_list)
wires_plot = []
found = 0
for w in wires:
    if found = 4:
        break
    if abs(wire_list[int(w)] - mean) < 5:
        wires_plot.append(w)
        found += 1
for i, w in enumerate(wires_plot):
    plt.subplot(2, 2, i + 1)
    plt.ylabel("doca Res [cm]")
    plt.xlabel("z [cm]")
    plt.ylim(-0.2, 0.2)
    plt.xlim(-90, 90)
    plt.hist2d(dres[wires==w], zvec[wires==w], bins=[50, 100], range=[[-90, 90], [-0.2, 0.2]])

plt.show()

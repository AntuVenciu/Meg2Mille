"""
Draw residuals from cosmic track fit at each millepede iteration
"""
import os
import glob
import ROOT

import numpy as np
import matplotlib.pyplot as plt


WRITE_WIRE_LIST = 1

# iteration of millepede
ITERATION = 0
directory = "mc" #f"residuals_iter_{ITERATION}"

# residuals and wire list
wires = np.array([])
dres = np.array([])
zvec = np.array([])
xres = np.array([])
yres = np.array([])

# Loop over all files res_437*.txt in the directory
for res_file in glob.iglob(f"{directory}/res_MC*.txt"):
    if os.path.getsize(res_file) < 1000:
        continue
    doca, res, sigma, z, wire = np.loadtxt(res_file, unpack=True)
    wires = np.append(wires, wire)
    dres = np.append(dres, res)
    zvec = np.append(zvec, z)
    #xres = np.append(xres, x)
    #yres = np.append(yres, y)

if WRITE_WIRE_LIST:
    # Count hits on wire to establish good wires (with > 50 hits) to be aligned
    wire_list = np.zeros(1920)
    for w in wires:
        wire_list[int(w)] += 1
    with open(f"MC_wire_list_iter{ITERATION}.txt", "w") as f:
        for w, entries in enumerate(wire_list):
            if entries > 50:
                f.write(f"{w}\n")
    print(f"Created file MC_wire_list_iter{ITERATION}.txt")

# Create the histograms of residuals per layer and per wire
plt.figure(1)
for i in range(1, 10, 1):
    plt.subplot(3, 3, i)
    plt.xlim(0, 192)
    plt.ylim(-0.1, 0.1)
    plt.hist2d(wires[wires%10 == i]%192, dres[wires%10 == i], bins=[192, 100], range=[[-0.5, 192.5], [-0.1, 0.1]])
plt.savefig(f"MC_residuals_plane_per_plane_iter{ITERATION}.pdf")
"""
wire_list = np.zeros(1920)
for w in wires:
    wire_list[int(w)] += 1
mean = np.mean(wire_list)
wires_plot = []

found = 0
#hxresz = ROOT.TH2F("hxresz", "x Res VS z (#gamma determination); z [cm]; x Res [cm]", 40, -80, 80, 40, -0.2, 0.2)
for w in wires:
    if found == 1:
        break
    if abs(wire_list[int(w)] - mean) < np.sqrt(mean):
        wires_plot.append(w)
        found += 1

# ROOOT CODE

for x, z, w in zip(xres, zvec, wires):
    if w != wires_plot[0]:
        continue
    #print(f"Filling with {x} {z}")
    hxresz.Fill(z, x)

c = ROOT.TCanvas("c")
hxresz.Draw("colz")
prof = hxresz.ProfileX("pfx", 1, -1, "dsame")
prof.SetMarkerStyle(20)
prof.SetMarkerSize(1)
c.SaveAs("test.pdf")




# Histogram of residuals VS z for four "average" wires
plt.figure(2)
plt.title(r"$\gamma$ from y Res VS z")
for i, w in enumerate(wires_plot):
    plt.subplot(2, 2, i + 1)
    plt.ylabel("y Res [cm]")
    plt.xlabel("z [cm]")
    plt.ylim(-0.2, 0.2)
    plt.xlim(-80, 80)
    plt.hist2d(zvec[wires==w], yres[wires==w], bins=[50, 25], range=[[-80, 80], [-0.2 , 0.2]])
plt.savefig(f"gamma_yres_z_iter{ITERATION}.pdf")
    
plt.figure(3)
plt.title(r"$\gamma$ from x Res VS z")
for i, w in enumerate(wires_plot):
    plt.subplot(2, 2, i + 1)
    plt.ylabel("x Res [cm]")
    plt.xlabel("z [cm]")
    plt.ylim(-0.2, 0.2)
    plt.xlim(-80, 80)
    
    plt.hist2d(zvec[wires==w], xres[wires==w], bins=[50, 25], range=[[-80, 80], [-0.2 , 0.2]])
plt.savefig(f"gamma_xres_z_iter{ITERATION}.pdf")
    
plt.figure(4)
plt.title("DOCA Residuals")
plt.xlabel("$d_{hit} - d_{track}$ [cm]")
plt.hist(dres, bins=100, range=(-0.15, 0.15))
plt.savefig(f"doca_res_iter{ITERATION}.pdf")

plt.show()
"""

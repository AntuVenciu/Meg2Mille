# Meg2Mille
Code for software alignment of the MEG II Drift Chamber with MillePede II software by Claus Kelinwort. 
The complete alignment algorithm heavily relies on the structure of the MEG II software
(database, event analysis etc...). What is quite generic and may be exploit by others
is the definition of the geometry of the detector inside the millepede_def.py file
and the routine to write the binary file (equivalent to Mille routine in the official MillePede II software).

Most of these instructions are for future reference of the author
and other members of the MEG II experiment, therefore they may be quite useless to others.

## Instructions of usage

### Geometry parameters
- Wire geometrical parameters are stored in wire_parameters_CYLDCH{Conf_ID}.txt files,
  where each different Conf_ID refers to a different geometry. 66 is the starting point,
  then other ID will be created at subsequent iterations; 
- the survey measurements of the wires positions (constraints) are in anode_CYLDCH46.txt file
  We found an error in the anode_survey.txt files, therefore we need to use
  this other "measurements" (obtained with points at +/- 94 cm for each wire with software geom)

### Running millepede_def.py

To run the millepede macro:

1. Create a bad wire list (or read from database)
2. Data are stored in outTrack*.root files in residuals_iter_{ITERATION} directories
   (one for each iteration).
   They are written using the cosmics_track_fit_TXYnoB.C macro inside MEG II software.
   (This is parallelyzed using the slurm script: track_selection.sl)
3. Create the wire_parameter*.txt file suited for the stage
   of the alignment running the   ../write_wire_parameters.C macro
4. A list of good wires can be created just counting the wires with > tot hits.
   This is done in the macro "residuals_analysis.py". The output file can be modified eliminating a
   few wires to use a reference.
   In that case, just request in the "millepede_def.py" script that the
   hits on track are built for every wire, not requiring them to be in the good wires' list.
   In the first iteration (iter 0), wires 500, 1000, 1600 have been removed.
5. The script returns a binary file in Mille Style with name "mp2meg2_{ITERATION}.bin"
   that can be run using the Pede executable of
   [MillePedeII software](https://gitlab.desy.de/claus.kleinwort/millepede-ii/-/tree/main)
   $ ./pede steer.txt
   where "steer.txt" is a txt file written according to MillePede II requirements
   (contains at least the name of the binary file)   
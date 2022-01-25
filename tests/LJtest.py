# test LJ
# myManager = manager(forces=["LJ"],boundaries=[11.3,11.3,11.3],position="xyz",atomType="xyz",dimensions=3)
# myManager.initialize(positions={"positionsFile":"coord_ex.xyz"},masses={"Ns":[32],"masses":[6.64e-26*kg]},
#                      momentum={"temperature":298*K})
# positions = myManager.run(1000,LJ={"Ar-Ar":{"sigma":3.4,"epsilon":9.977346068146507e-05}},printStats=10,savePositions=10)
# # myManager.makePositionsFile(positions,"testArgon.xyz")
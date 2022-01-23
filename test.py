from manager import manager
from unum.units import *
import matplotlib.pyplot as plt

# test potential
# myManager = manager(N=1,forces=["CoordsEquationPotential"],boundaries=[100])
# myManager.initialize(positions={"positionsList":[[1.]]},masses={"Ns":[1],"masses":[1e-26*kg]},
#                      momentum={"temperature":300*K},atomTypes={"Ns":[1],"types":[1]})
# positions = myManager.run(1,equations=["-1e-11*q*(1-100*numpy.exp(-q**2))"],printStats=1000,savePositions=1,Langevin=True,gamma=10**25,temperature=300*K)
# plt.plot([p[0] for p in positions])
# plt.show()

# test LJ
myManager = manager(forces=["LJ"],boundaries=[11.3,11.3,11.3],position="xyz",atomType="xyz",dimensions=3)
myManager.initialize(positions={"positionsFile":"coord_ex.xyz"},masses={"Ns":[32],"masses":[6.64e-26*kg]},
                     momentum={"temperature":298*K})
positions = myManager.run(1000,LJ={"Ar-Ar":{"sigma":3.4,"epsilon":9.977346068146507e-05}},printStats=10,savePositions=10)
myManager.makePositionsFile(positions,"testArgon.xyz")
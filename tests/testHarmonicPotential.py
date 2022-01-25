from pythonMD import manager
from unum.units import *
import matplotlib.pyplot as plt

# test single particle in 1d

# Increase timestep and check energy convergence

# myManager = manager(forces=["CoordsEquationPotential"],boundariesType=None,dimensions=1)
#
# for dt in range(1,20,2):
#     myManager.dt = dt
#     myManager.initialize(positions=[[5.]], masses=1e-26 * kg,
#                          temperature=0 * K, types=[1])
#     data = myManager.run(10000,forceEquations=["-1e-5*q"],
#                               potentialEquations=["5e-6*(q**2)"],
#                               printStats=1,savePositions=1)
#     totalRelativeEnergy = [abs((energy-data["totalEnergy"][0])*100/energy) for energy in data["totalEnergy"]]
#     totalRelativeEnergy.sort()
#     print(f"maximum change in energy for dt={dt}: {totalRelativeEnergy[-1]}%")

# Change initial position

myManager = manager(forces=["CoordsEquationPotential"],boundariesType=None,dimensions=1)

for p in range(1,11,2):

    myManager.initialize(positions=[[float(p)]], masses=1e-26 * kg,
                         temperature=0 * K, types=[1])
    data = myManager.run(10000,forceEquations=["-1e-5*q"],
                              potentialEquations=["5e-6*(q**2)"],
                              printStats=1,savePositions=1)
    data["positions"].sort()
    print(data["positions"][0],data["positions"][-1])

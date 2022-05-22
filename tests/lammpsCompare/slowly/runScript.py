import pandas as pd
from pythonMD import manager
import numpy as np
unitFit = 2391.1776000000004
prop = [("VelocityVerlet",1.)]
grid=np.linspace(0,10,100)
grid = 0.5*(grid[1:]+grid[:-1])
metaDynamicsDict = {"CVs":[{"type":"singleParticlePosition","axis":0}],"sigma":0.5,"height":0.0002091,"pace":1,"biasFactor":10,"temperature":300,
                    "grid":grid}
myManager = manager(dt=1,forces=[],boundariesType=None,dimensions=2,prop=prop,metaDynamicsDict=metaDynamicsDict,momentum="initialVelocities")
myManager.initialize(positions=[[0.1,0.]],masses=1.,velocities=[[0.1,0.]],types=[1])
data = myManager.run(10, printStats=1,saveStats=1,temperature=300)
transposed = np.transpose(np.array(data["hills"]['singleParticlePosition']))
data = pd.DataFrame({"x":transposed[0],"height":transposed[1]*unitFit,"bias":transposed[2]*unitFit,"cbias":transposed[3]*unitFit*300*8.310549580257024e-7})
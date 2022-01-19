import numpy
from unum.units import *
class forceCalculator():

    _forceList = None
    _manager = None
    _cutoff = None

    @property
    def manager(self):
        return self._manager

    @property
    def forceList(self):
        return self._forceList
    @forceList.setter
    def forceList(self,newforceList):
        self._forceList=newforceList

    def __init__(self,forces,manager,cutoff):
        self._forceList=forces
        self._manager = manager
        self._cutoff=cutoff

    def calculateForce(self,**kwargs):
        Fs = []
        for fstyle in self.forceList:
            Fs.append(getattr(self,f"calculateForce_{fstyle}")(**kwargs))
        return sum(Fs)

    def calculateForce_CoordsEquationPotential(self,equations):
        Fs = []
        for qs, m in zip(self._manager.positions, self._manager.masses):
            newF = []
            for q,equation in zip(qs,equations):
                newF.append(eval(equation))
            Fs.append(newF)
        return numpy.array(Fs)

    def calculateForce_LJ(self,LJ):
        Fs = [[0] for n in range(self._manager.N)]
        neighboursList, neighboursDistList = self.findNeighbours()
        for particle in range(self._manager.N):
            for couple in range(len(neighboursList[particle])):
                Fs[particle] += 24*LJ[f"{particle}-{couple}"]["epsilon"]*(
                                2*LJ[f"{particle}-{couple}"]["sigma6"]**2/(
                                neighboursDistList[particle][couple]**13)+LJ[f"{particle}-{couple}"]["sigma6"]/(
                                neighboursDistList[particle][couple] ** 7))
        return numpy.array(Fs)

    def findNeighbours(self):
        neighboursList = [[] for n in range(self._manager.N)]
        neighboursDistList = [[] for n in range(self._manager.N)]
        for i in range(self._manager.N-1):
            for j in range(i+1,self._manager.N):
                rij = self._manager.positions[i]-self._manager.positions[j]
                rij -= self._manager.boundaries*numpy.rint(rij/self._manager.boundaries)
                rijval = sum([r**2 for r in rij])**0.5
                if abs(rijval) < self._cutoff:
                    neighboursList[i].append(j)
                    neighboursList[j].append(i)
                    neighboursDistList[i].append(rij)
                    neighboursDistList[j].append(rij)
        return neighboursList, neighboursDistList
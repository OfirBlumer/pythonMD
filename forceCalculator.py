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
        self._cutoff=cutoff.asNumber(ANGSTROM)

    def calculateForce(self,**kwargs):
        Fs = []
        for fstyle in self.forceList:
            Fs.append(getattr(self,f"calculateForce_{fstyle}")(**kwargs))
        return sum(Fs)

    def calculateForce_CoordsEquationPotential(self, forceEquations,**kwargs):
        Fs = []
        for qs, m in zip(self._manager.positions, self._manager.masses):
            newF = []
            for q,equation in zip(qs, forceEquations):
                newF.append(eval(equation))
            Fs.append(newF)
        return numpy.array(Fs)

    def calculateForce_LJ(self,LJ,**kwargs):
        Fs = [[0] for n in range(self._manager.N)]
        neighboursList, neighboursDistList = self.findNeighbours()
        for particle in range(self._manager.N):
            for couple in range(len(neighboursList[particle])):
                rad = sum([r**2 for r in neighboursDistList[particle][couple]])**0.5
                epsilon = LJ[f"{self._manager.atomTypes[particle]}-{self._manager.atomTypes[neighboursList[particle][couple]]}"]["epsilon"]
                sigma6 = (LJ[f"{self._manager.atomTypes[particle]}-{self._manager.atomTypes[neighboursList[particle][couple]]}"]["sigma"])**6
                Fs[particle] += neighboursDistList[particle][couple]/rad*24*epsilon*(2*sigma6**2/(rad**13)-sigma6/(rad ** 7))
        return numpy.array(Fs)

    def calculatePotentialEnergy(self, **kwargs):
        potE = 0
        for fstyle in self.forceList:
            potE += getattr(self, f"calculatePotentialEnergy_{fstyle}")(**kwargs)
        return potE

    def calculatePotentialEnergy_CoordsEquationPotential(self, potentialEquations,**kwargs):
        potE = 0
        for qs, m in zip(self._manager.positions, self._manager.masses):
            newF = []
            for q, equation in zip(qs, potentialEquations):
                newF.append(eval(equation)**2)
            potE += sum(newF)**0.5
        return potE

    def calculatePotentialEnergy_LJ(self, LJ,**kwargs):
        potE = 0
        neighboursList, neighboursDistList = self.findNeighbours()
        for particle in range(self._manager.N):
            for couple in range(len(neighboursList[particle])):
                rad = sum([r ** 2 for r in neighboursDistList[particle][couple]]) ** 0.5
                epsilon = LJ[f"{self._manager.atomTypes[particle]}-{self._manager.atomTypes[couple]}"]["epsilon"]
                sigma6 = (LJ[f"{self._manager.atomTypes[particle]}-{self._manager.atomTypes[couple]}"]["sigma"]) ** 6
                potE += 4 * epsilon * (sigma6**2/ (rad**12) - sigma6/(rad**6))
        return numpy.array(potE)

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
                    neighboursDistList[j].append(-rij)
        return neighboursList, neighboursDistList
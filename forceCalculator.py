withUnits = True
import numpy
if withUnits:
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
        """
        This class performs the calculations of the forces and potentials in the system
        :param forces: A list of forces types;
                       Currently the available forces are Lennard-Jones ("LJ") and
                       a user-defined equation ("CoordsEquationPotential") (list of str)
        :param manager: The simulation's main manager (manager class)
        :param cutoff: The cutoff used in the forces (int/float*unum length unit)
        """
        self._forceList=forces
        self._manager = manager
        if withUnits:
            self._cutoff=10 if cutoff is None else cutoff.asNumber(ANGSTROM)
        else:
            self._cutoff = 10 if cutoff is None else cutoff

    def calculateForce(self,**kwargs):
        """
        Calculates and sums all the forces.
        :param kwargs: Any parameters needed for the calculations
        :return: a numpy array with the force acting on each atom
        """
        Fs = []
        for fstyle in self.forceList:
            Fs.append(getattr(self,f"calculateForce_{fstyle}")(**kwargs))
        return sum(Fs)

    def calculateForce_CoordsEquationPotential(self, forceEquations,**kwargs):
        """
        Calculates the force acting on a particle in coordinates q using an equation for q
        :param forceEquations: a list of equations for each component of the position;
                               each equation is an expression may include "q" as the component and "m" as the mass
                               (list of str)
        :return: a numpy array with the force acting on each atom
        """
        Fs = []
        for qs, m in zip(self._manager.positions, self._manager.masses):
            newF = []
            for q,equation in zip(qs, forceEquations):
                newF.append(eval(equation))
            Fs.append(newF)
        return numpy.array(Fs)

    def calculateForce_LJ(self,LJ,**kwargs):
        """
        Calculates the Lennard Jones force acting on each atom
        :param LJ: A dictionary with the LJ parameters. holds a key named "{type1}-{type2}"
                   for each pair of types in the simulation. Each pair holds two keys with LJ parameters:
                   "epsilon", which should contain a parameter with value in units of U*ANGSTROM**2*fs**(-2) and
                   "sigma6", which should contain a parameter with value in units of ANGSTROM**6

        :return: a numpy array with the force acting on each atom
        """
        Fs = [numpy.zeros(self._manager.dimensions) for n in range(self._manager.N)]
        neighboursList, neighboursDistList = self.findNeighbours()
        rad=10
        for particle in range(self._manager.N):
            for couple in range(len(neighboursList[particle])):
                rad = sum([r**2 for r in neighboursDistList[particle][couple]])**0.5
                epsilon = (LJ[f"{self._manager.atomTypes[particle]}-"
                              f"{self._manager.atomTypes[neighboursList[particle][couple]]}"]["epsilon"])
                sigma6 = (LJ[f"{self._manager.atomTypes[particle]}-{self._manager.atomTypes[neighboursList[particle][couple]]}"]["sigma6"])
                Fs[particle] += neighboursDistList[particle][couple]/rad*24*epsilon*(2*sigma6**2/(rad**13)-sigma6/(rad ** 7))
        return numpy.array(Fs)

    def calculatePotentialEnergy(self, **kwargs):
        """
        Calculates and sums all the potential energies.
        :param kwargs: Any parameters needed for the calculations
        :return: The value of the potential energy in the system
        """
        potE = 0
        for fstyle in self.forceList:
            potE += getattr(self, f"calculatePotentialEnergy_{fstyle}")(**kwargs)
        return potE

    def calculatePotentialEnergy_CoordsEquationPotential(self, potentialEquations,**kwargs):
        """
        Calculates the potential energy in coordinates q using an equation for q
        :param potentialEquations: a list of equations for each component of the position;
                               each equation is an expression may include "q" as the component and "m" as the mass
                               (list of str)
        :return: a value of potential energy calculated using the given equations
        """
        potE = 0
        for qs, m in zip(self._manager.positions, self._manager.masses):
            newF = []
            for q, equation in zip(qs, potentialEquations):
                newF.append(eval(equation))
            potE += sum(newF)
        return potE

    def calculatePotentialEnergy_LJ(self, LJ,**kwargs):
        """
        Calculates the total Lennard-Jones potential in the system
        :param LJ: see the description in calculateForce_LJ
        :return: The value of the LJ potential energy
        """
        potE = 0
        radiusList = self.findRadius()
        for raduis in radiusList:
                epsilon = (LJ[raduis[1]]["epsilon"])
                sigma6 = (LJ[raduis[1]]["sigma6"])
                potE += 4 * epsilon * (sigma6**2/ (raduis[0]**12) - sigma6/(raduis[0]**6))
        return numpy.array(potE)

    def findNeighbours(self):
        """
        finds the neighbours in the system.
        :return: neighboursList, neighboursDistList:
                 neighboursList is a list that contains an inner list for each atom in the system.
                 Each inner list contains the indexes of its neighbors.
                 neighboursDistList is also a list that contains an inner list for each atom in the system.
                 Each inner list contains the distance from the neighbors specified in neighboursList.
        """
        neighboursList = [[] for n in range(self._manager.N)]
        neighboursDistList = [[] for n in range(self._manager.N)]
        for i in range(self._manager.N-1):
            for j in range(i+1,self._manager.N):
                rij = self._manager.positions[i]-self._manager.positions[j]
                change = self._manager.boundaries*numpy.fix(rij*2/self._manager.boundaries)
                rij -= change
                rijval = sum([r**2 for r in rij])**0.5
                if abs(rijval) < self._cutoff:
                    neighboursList[i].append(j)
                    neighboursList[j].append(i)
                    neighboursDistList[i].append(rij)
                    neighboursDistList[j].append(-rij)
        return neighboursList, neighboursDistList

    def findRadius(self):
        """
        finds the radius in the system.
        :return: radiusList, a list of radii
        """
        radiusList = []

        for i in range(self._manager.N-1):
            for j in range(i+1,self._manager.N):
                rij = self._manager.positions[i]-self._manager.positions[j]
                change = self._manager.boundaries*numpy.fix(rij*2/self._manager.boundaries)
                rij -= change
                rijval = sum([r**2 for r in rij])**0.5
                if abs(rijval) < self._cutoff:
                    radiusList.append((rijval,f"{self._manager.atomTypes[i]}-{self._manager.atomTypes[j]}"))

        return radiusList

    #### Specific potentials, because eval is slow...

    def calculatePotentialEnergy_RessetingFirstPotential(self, qConstant,gaussConstant,gaussWidth,**kwargs):
        width = gaussWidth ** 2
        potE = 0
        for qs, m in zip(self._manager.positions, self._manager.masses):
            newF = []
            for q in qs:
                newF.append(qConstant*q**2+gaussConstant*numpy.exp(-q**2/(2*width)))
            potE += sum(newF)
        return potE

    def calculateForce_RessetingFirstPotential(self,qConstant,gaussConstant,gaussWidth,**kwargs):

        Fs = []
        for qs, m in zip(self._manager.positions, self._manager.masses):
            newF = []
            for q in qs:
                newF.append(-q*(qConstant*2-gaussConstant*numpy.exp(-q**2/(2*gaussWidth))/gaussWidth))
            Fs.append(newF)
        return numpy.array(Fs)

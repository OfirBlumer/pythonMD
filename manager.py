withUnits = True
withUnits = False
from .initialization import initialization
from .propagator import propagator
from .forceCalculator import forceCalculator
from .metaDynamics import metaDynamics
if withUnits:
    from unum.units import *
import numpy
import os
Na = 6.02e23
kb_si = 1.38e-23
energyToJ = 1.66054e-17

class manager():

    _prop = None
    _initialize = None
    _forces = None
    _positions = None
    _momentums = None
    _masses = None
    _dimensions = None
    _N = None
    _dt = None
    _boundaries = None
    _boundariesType = None
    _atomTypes = None
    _initialProps = None
    _metaDynamics = None

    @property
    def metaDynamics(self):
        return self._metaDynamics
    @property
    def atomTypes(self):
        return self._atomTypes
    @atomTypes.setter
    def atomTypes(self,newatomTypes):
        self._atomTypes=newatomTypes
    @property
    def boundaries(self):
        return self._boundaries
    @property
    def boundariesType(self):
        return self._boundariesType
    @property
    def dimensions(self):
        return self._dimensions
    @dimensions.setter
    def dimensions(self,newdimensions):
        self._dimensions=newdimensions
    @property
    def positions(self):
        return self._positions
    @positions.setter
    def positions(self,newpositions):
        self._positions=newpositions
    @property
    def momentums(self):
        return self._momentums
    @momentums.setter
    def momentums(self,newmomentums):
        self._momentums=newmomentums
    @property
    def masses(self):
        return self._masses
    @masses.setter
    def masses(self,newProp):
        self._masses=newProp
    @property
    def N(self):
        return self._N
    @N.setter
    def N(self,newProp):
        self._N=newProp
    @property
    def forces(self):
        return self._forces
    @forces.setter
    def forces(self,newforces):
        self._forces=newforces
    @property
    def dt(self):
        return self._dt
    @dt.setter
    def dt(self,newdt):
        self._dt=newdt

    def __init__(self,dt,boundaries=None,boundariesType="periodic",prop=[("VelocityVerlet",1.)],momentum="MaxwellBoltzmann",
                 forces=["LJ"],dimensions=1,cutoff=None,seed=0,metaDynamicsDict={}):
        """
        This class manage the simulation. It holds the simulation data and
        calls the acting functions that propagates the simulation.

        :param boundaries: A list of the high boundaries of the simulation box coordinates (the minimum is 0).
                           The list must match in length the number of dimensions. The default is None,
                           which doesn't fit periodic boundary conditions. (None or list of int/float)

        :param boundariesType: The type of boundaries condition. Currently, only periodic conditions
                               are implemented. Any input other than the default "periodic" would
                               lead to a simulation with no boundaries. (None or str)

        :param prop: The types of propagator used in the simulation and their relative timesteps.
                     Currently, VelocityVerlet, Langevin and CSVR are available. (list of (str,float))

        :param momentum: The methods by which the momentum values are calculated;
                         currently the only option is "MaxwellBoltzmann". (str)

        :param forces: A list of forces types;
                       Currently the available forces are Lennard-Jones ("LJ") and
                       a user-defined equation ("CoordsEquationPotential") (list of str)

        :param dimensions: The number of dimensions in the simulation (int)

        :param dt: The time step's value; default is 1fs (int/float*unum time unit)

        :param cutoff: The cutoff used in the forces; default is 10 Angstrom (int/float*unum length unit)
        """
        self._initialize = initialization(momentum=momentum,manager=self)
        self._dimensions = dimensions
        if withUnits:
            self._dt = dt.asNumber(fs)
        else:
            self._dt = dt
        self._prop = propagator(prop=prop,manager=self)
        self._forces = forceCalculator(forces=forces,manager=self,cutoff=cutoff)
        self._boundaries=boundaries
        self._boundariesType=boundariesType
        numpy.random.seed(seed)
        self._metaDynamics = metaDynamics(manager=self,**metaDynamicsDict)

    def initialize(self,positions,masses,types=None,**kwargs):
        """
        Initialize the system

        :param positions: see initialization class getPositions doc
        :param masses: see initialization class getMasses doc
        :param types: see initialization class getAtomTypes doc
        :param kwargs: Additional parameters that may be required by initialization class' getMomentums
        """
        # print("Initializing Molecular Dynamics Simulation")
        self.positions = self._initialize.getPositions(positions,**kwargs)
        self.masses = self._initialize.getMasses(masses)
        self.momentums = self._initialize.getMomentums(**kwargs)
        self.atomTypes = self._initialize.getAtomTypes(positions=positions,types=types)
        self._initialProps = {"positions":numpy.copy(self.positions),"momentum":numpy.copy(self.momentums),"momentumArgs":kwargs}

    def run(self, Niterations, savePositions=100,saveMomentum=100,saveStats=100, printStats=100,
            resetMethod=None,resetSameMomentum=True, stopCriterion=None, LJ=None,**kwargs):
        if isinstance(LJ,dict):
            newkeys = []
            newkeysvals = []
            for key in LJ.keys():
                if withUnits:
                    LJ[key]["epsilon"] = (LJ[key]["epsilon"]).asNumber(U * ANGSTROM ** 2 * fs ** (-2))
                    LJ[key]["sigma6"] = (LJ[key]["sigma"]).asNumber(ANGSTROM)**6
                else:
                    LJ[key]["epsilon"] = LJ[key]["epsilon"]
                    LJ[key]["sigma6"] = LJ[key]["sigma"]
                newkeys.append(f"{key.split('-')[1]}-{key.split('-')[0]}")
                newkeysvals.append({"epsilon":LJ[key]["epsilon"],"sigma6":LJ[key]["sigma6"]})
            for i in range(len(newkeys)):
                LJ[newkeys[i]] = newkeysvals[i]
        positions = []
        Ts = []
        kineticEnergyList = []
        potentialEnergyList = []
        totalEnergyList = []
        momentum = []
        totnstep = None
        i = 0
        while i < Niterations - 1:
            startStep = i
            if resetMethod is not None:
                nextReset = self._prop.reset(resetMethod=resetMethod, **kwargs)
                self.positions = numpy.copy(self._initialProps["positions"])
                if resetSameMomentum:
                    self.momentums = numpy.copy(self._initialProps["momentum"])
                else:
                    self.momentums = numpy.copy(self._initialize.getMomentums(**self.initialProps["momentumArgs"]))
                print(f"next reset in {nextReset} steps")
                finalStep = min(i + nextReset, Niterations)
            else:
                finalStep = Niterations
            for j in range(startStep,finalStep):
                i+=1
                # if self.positions[0][0]<0:
                #     print(f"Stopped because fulfilled criterion after {i} steps")
                #     totnstep = i
                #     i = Niterations
                #     break
                self.metaDynamics.metaManager(i)
                self._prop.propagate(LJ=LJ,**kwargs)
                if i >= savePositions and i%savePositions==0:
                    positions.append(numpy.copy(self.positions))
                if i >= saveMomentum and i%saveMomentum==0:
                    momentum.append(numpy.copy(self.momentums))
                if i >= saveStats and i%saveStats==0:
                    kineticEnergies = self.momentums**2/2/self.masses
                    kineticEnergy = sum(sum(kineticEnergies))*energyToJ
                    potentialEnergy = self.forces.calculatePotentialEnergy(LJ=LJ,**kwargs)*energyToJ
                    T = 2*kineticEnergy/(kb_si)/self.dimensions/self.N
                    Ts.append(T),
                    kineticEnergyList.append(kineticEnergy)
                    potentialEnergyList.append(potentialEnergy)
                    totalEnergyList.append((kineticEnergy+potentialEnergy))
                    if i >= printStats and i%printStats==0:
                        print(i, T, kineticEnergy, potentialEnergy ,(kineticEnergy + potentialEnergy) )
        totnstep = i if totnstep is None else totnstep
        return {"positions":positions,"momenta":momentum,"T":Ts,"kineticEnergy":kineticEnergyList,
                "potentialEnergy":potentialEnergyList, "totalEnergy":totalEnergyList,"nsteps":totnstep,"hills":self.metaDynamics.hills}

    def makePositionsFile(self,positions,save=None):
        """
        Generates a xyz format string from the positions output of a run.
        :param positions: The positions output
        :param save: the name of a file in which the xyz format string is saved;
                     if it is None the string isn't saved (str or None)
        :return: The xyz format string
        """
        fileString = ""
        for step in range(len(positions)):
            fileString += f"{len(positions[0])}\nStep {step}\n"
            for atom in range(self.N):
                fileString += f"{self.atomTypes[atom]}"
                for d in range(self.dimensions):
                    fileString += f"\t{positions[step][atom][d]}"
                if d<3:
                    for i in range(3-d):
                        fileString += "\t0"
                fileString += "\n"
        if save is not None:
            with open(save,"w") as file:
                file.write(fileString)
        return fileString

from .initialization import initialization
from .propagator import propagator
from .forceCalculator import forceCalculator
from unum.units import *
import numpy
import os

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

    def __init__(self,boundaries=None,boundariesType="periodic",prop="VelocityVerlet",momentum="MaxwellBoltzmann",
                 forces=["LJ"],dimensions=1,dt=1*fs,cutoff=10*ANGSTROM):
        """
        This class manage the simulation. It holds the simulation data and
        calls the acting functions that propagates the simulation.

        :param boundaries: A list of the high boundaries of the simulation box coordinates (the minimum is 0).
                           The list must match in length the number of dimensions. The default is None,
                           which doesn't fit periodic boundary conditions. (None or list of int/float)

        :param boundariesType: The type of boundaries condition. Currently, only periodic conditions
                               are implemented. Any input other than the default "periodic" would
                               lead to a simulation with no boundaries. (None or str)

        :param prop: The type of propagator used in the simulation.
                     Currently, only VelocityVerlet is available. (str)

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
        self._dt = dt.asNumber(fs)
        self._prop = propagator(prop=prop,manager=self)
        self._forces = forceCalculator(forces=forces,manager=self,cutoff=cutoff)
        self._boundaries=boundaries
        self._boundariesType=boundariesType

    def initialize(self,positions,masses,types=None,**kwargs):
        """
        Initialize the system

        :param positions: see initialization class getPositions doc
        :param masses: see initialization class getMasses doc
        :param types: see initialization class getAtomTypes doc
        :param kwargs: Additional parameters that may be required by initialization class' getMomentums
        """
        print("Initializing Molecular Dynamics Simulation")
        self.positions = self._initialize.getPositions(positions)
        self.masses = self._initialize.getMasses(masses)
        self.momentums = self._initialize.getMomentums(**kwargs)
        self.atomTypes = self._initialize.getAtomTypes(positions=positions,types=types)

    def run(self, Niterations, savePositions=100,saveStats=100, printStats=100, resetMethod=None, stopCriterion=None, LJ=None,**kwargs):
        """
        The simulation's main function, in which the main loop is executed

        :param Niterations: number of steps in the simulation (int)
        :param savePositions: the rate at which the positions are saved, each savePositions steps (int)
        :param printStats: the rate at which the temperatures and energies are printed and saved, each savePositions steps (int)
        :param resetMethod: optional, default is None.
                            Else, the method by which resetting occurs (see propagator class' reset function)
        :param stopCriterion: A str of a Boolean statement that is used for an if check;
                              stops the simulation of the criterion is reached
        :param LJ: optional, for LJ potential. A dictionary with the LJ parameters. holds a key named "{type1}-{type2}"
                   for each pair of types in the simulation. Each pair holds two keys with LJ parameters:
                   "epsilon", which should contain a parameter with units of energy, and
                   "sigma", which should contain a parameter with units of length
        :param kwargs: Any additional parameters that may be needed for the resetting,
                       propagation or calculation of potential energy
        :return: a dictionary with a list of positions , temperatures and energies at different times
        """
        print(f"Start running for {Niterations} iterations")
        print("Timestep Temperature KineticEnergy PotentialEnergy TotalEnergy")
        if isinstance(LJ,dict):
            for key in LJ.keys():
                LJ[key]["epsilon"] = (LJ[key]["epsilon"]).asNumber(U * ANGSTROM ** 2 * fs ** (-2))
                LJ[key]["sigma6"] = (LJ[key]["sigma"]).asNumber(ANGSTROM)**6
        positions = []
        Ts = []
        kineticEnergyList = []
        potentialEnergyList = []
        totalEnergyList = []
        for i in range(Niterations):
            if stopCriterion is not None:
                if eval(stopCriterion):
                    print(f"Stopped because fulfilled criterion after {i} steps")
                    break
            restarted = self._prop.reset(resetMethod=resetMethod, iterationStep=i, **kwargs)
            if restarted:
                self.positions = self._initialize.getPositions(**self._initialProps["positions"])
                self.momentums = self._initialize.getMomentums(**self._initialProps["momentum"])
            else:
                self._prop.propagate(LJ=LJ,**kwargs)
                if i >= savePositions and i%savePositions==0:
                    positions.append(numpy.copy(self.positions))
                if i >= saveStats and i%saveStats==0:
                    kineticEnergies = self.momentums**2/2/self.masses
                    kineticEnergy = (sum(sum(kineticEnergies))*U*ANGSTROM**2*fs**(-2)).asNumber(J)
                    potentialEnergy = (self.forces.calculatePotentialEnergy(LJ=LJ,**kwargs)*U*ANGSTROM**2*fs**(-2)).asNumber(J)
                    T = 2*kineticEnergy/(1.38e-23)/self.dimensions/self.N
                    Ts.append(T),
                    kineticEnergyList.append(kineticEnergy*6.02e23/self.N)
                    potentialEnergyList.append(potentialEnergy*6.02e23/self.N)
                    totalEnergyList.append((kineticEnergy+potentialEnergy)*6.02e23/self.N)
                    if i >= printStats and i%printStats==0:
                        print(i,T,kineticEnergy*6.02e23/self.N,potentialEnergy*6.02e23/self.N,(kineticEnergy+potentialEnergy)*6.02e23/self.N)
        return {"positions":positions,"T":Ts,"kineticEnergy":kineticEnergyList,
                "potentialEnergy":potentialEnergyList, "totalEnergy":totalEnergyList}

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
                fileString += "\n"
        if save is not None:
            with open(os.path.join(os.path.dirname(__file__),save),"w") as file:
                file.write(fileString)
        return fileString

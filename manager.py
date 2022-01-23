from initialization import initialization
from propagator import propagator
from forceCalculator import forceCalculator
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

    def __init__(self,N=None,boundaries=None,boundariesType="periodic",prop="VelocityVerlet",position="list",
                 momentum="MaxwellBoltzmann",mass="lists",atomType="lists",forces=["LJ"],dimensions=1,dt=1*fs,cutoff=10*ANGSTROM):
        self._initialize = initialization(initializ={"position":position,"momentum":momentum,"mass":mass,"atomType":atomType},manager=self)
        self._dimensions = dimensions
        self._N=N
        self._dt = dt.asNumber(fs)
        self._prop = propagator(prop=prop,manager=self)
        self._forces = forceCalculator(forces=forces,manager=self,cutoff=cutoff)
        self._boundaries=boundaries
        self._boundariesType=boundariesType

    def initialize(self,positions,masses,momentum,atomTypes=None):
        print("Initializing Molecular Dynamics Simulation")
        self.positions = self._initialize.getPositions(**positions)
        self.masses = self._initialize.getMasses(**masses)
        self.momentums = self._initialize.getMomentums(**momentum)
        atomTypes = positions if atomTypes is None else atomTypes
        self.atomTypes = self._initialize.getAtomTypes(**atomTypes)

    def run(self,Niterations,savePositions=100,printStats=100,dt=None,**kwargs):
        print(f"Start running for {Niterations} iterations")
        print("Timestep Temperature KineticEnergy PotentialEnergy TotalEnergy")
        self.dt = self.dt if dt is None else dt
        positions = []
        for i in range(Niterations):
            self._prop.propagate(**kwargs)
            if i >= savePositions and i%savePositions==0:
                positions.append(numpy.copy(self.positions))
            if i >= printStats and i%printStats==0:
                kineticEnergies = self.momentums**2/2/self.masses
                kineticEnergy = (sum(kineticEnergies) if self.dimensions==1 else \
                                sum([sum([e**2 for e in energy])**0.5 for energy in kineticEnergies])*U*ANGSTROM**2*fs**(-2)).asNumber(J)
                potentialEnergy = (self.forces.calculatePotentialEnergy(**kwargs)*U*ANGSTROM**2*fs**(-2)).asNumber(J)
                T = 2*kineticEnergy/(1.38e-23)/self.dimensions/self.N
                print(i,T,kineticEnergy*6.02e23/self.N,potentialEnergy*6.02e23/self.N,(kineticEnergy+potentialEnergy)*6.02e23/self.N)
        return positions

    def makePositionsFile(self,positions,save=None):
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

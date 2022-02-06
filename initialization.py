from unum.units import *
import numpy

class initialization():

    _momentum = None
    _manager = None

    def __init__(self,momentum,manager):
        """
        This class manages the initialization of the simulation's system
        :param momentum: The methods by which the momentum values are calculated;
                         currently the only option is "MaxwellBoltzmann". (str)
        :param manager:  The simulation's main manager (manager class)
        """
        self._momentum=momentum
        self._manager=manager

    def getPositions(self,positions):
        """
        Sets the number of atoms in the system and the initial positions
        :param positions: The positions; currently supports a list of list or a name of a xyz file (list or str)
        :return: the positions as a numpy array
        """
        print("Reading positions")
        if isinstance(positions,list):
            ret = numpy.array(positions)
            self._manager.N=len(positions)
        elif isinstance(positions,str):
            if positions.split(".")[-1]=="xyz":
                with open(positions, "r") as file:
                    lines = file.readlines()
                self._manager.N = len(lines) - 2
                print(f"Reading {self._manager.N} atoms")
                positionsList = [[] for l in range(len(lines) - 2)]
                for i in range(len(lines) - 2):
                    for val in lines[i + 2].split()[1:]:
                        positionsList[i].append(float(val))
                ret = numpy.array(positionsList)
            else:
                raise ValueError("Currently, only .xyz are available")
        return ret

    def getMasses(self,mass):
        """
        Sets the masses in the system.
        :param mass: The mass of atoms; currently supports a single value for uniform mass
                     or a list of masses for all atoms. (int/float*unum mass unit or list of those)
        :return: the masses as a numpy array
        """
        newMasses = []
        try:
            mass = mass.asNumber(U)
        except:
            mass = [mas.asNumber(U) for mas in mass]
        if isinstance(mass, int) or isinstance(mass, float):
            for n in range(self._manager.N):
                newMasses.append([mass for d in range(self._manager.dimensions)])
        elif isinstance(mass, list):
            for massVal in mass:
                newMasses.append([massVal for d in range(self._manager.dimensions)])
        return numpy.array(newMasses)

    def getAtomTypes(self,positions,types=None):
        """
        Sets the atom types in the system.
        :param positions: The positions delivered for getPositions; if it is a xyz file it is used for extracting the atoms types
        :param types: If the types weren't set by the positions, it is set by this param;
                      currently only supports a list of types as strings for all atoms
        :param kwargs:
        :return:
        """
        if isinstance(positions,str):
            if positions.split(".")[-1]=="xyz":
                with open(positions, "r") as file:
                    lines = file.readlines()
                types = []
                for l in lines[2:]:
                    types.append(l.split()[0])
                ret = numpy.array(types)
        elif isinstance(types,list):
            ret = numpy.array(types)
        return ret

    def getMomentums(self,**kwargs):
        """
        Sets the momentum values in the system.
        :param kwargs: any parameters needed for the calculation of the momentum
        :return: the momentum as a numpy array
        """
        print("Calculating Momentums using ",self._momentum)
        return getattr(self,f"getMomentums_{self._momentum}")(**kwargs)

    def getMomentums_MaxwellBoltzmann(self, temperature):
        """
        Sets the momentum using the Maxwell-Boltzmann distribution
        :param temperature: The temperature of the system (int/float*unum units of temperature)
        :return: the momentum as a numpy array
        """
        kT = 8.310549580257024e-7*temperature.asNumber(K)
        Momentums = []
        d = self._manager.dimensions
        for i in range(self._manager.N):
            mass = self._manager.masses[i][0]
            std = numpy.sqrt((kT / mass))
            V = numpy.sqrt(numpy.random.normal(0,std)**2+numpy.random.normal(0,std)**2+numpy.random.normal(0,std)**2)
            V = V if numpy.random.uniform() > 0.5 else -V
            if d==1:
                Momentums.append([V*mass])
            elif d==2:
                phi = numpy.random.uniform()*2*numpy.pi
                Momentums.append([numpy.cos(phi)*V*mass,numpy.sin(phi)*V*mass])
            elif d==3:
                phi = numpy.random.uniform() * 2 * numpy.pi
                psi = numpy.random.uniform() * numpy.pi
                Momentums.append([numpy.cos(phi) * numpy.sin(psi) * V*mass, numpy.sin(phi) * numpy.sin(psi) * V*mass, numpy.cos(psi) * V*mass])
        return numpy.array(Momentums)

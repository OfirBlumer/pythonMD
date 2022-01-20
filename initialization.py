from unum.units import *
import numpy
import pandas
class initialization():

    _initializ = None
    _manager = None

    def __init__(self,initializ,manager):
        self._initializ=initializ
        self._manager=manager

    def getPositions(self,**kwargs):
        print("Getting positions using ", self._initializ['position'])
        return getattr(self,f"getPositions_{self._initializ['position']}")(**kwargs)

    def getMasses(self,**kwargs):
        print("Getting masses using ", self._initializ['mass'])
        return getattr(self,f"getMasses_{self._initializ['mass']}")(**kwargs)

    def getMasses_lists(self,Ns,masses,**kwargs):
        newMasses = []
        for i in range(len(Ns)):
            for n in range(Ns[i]):
                newMasses.append([masses[i].asNumber(U) for d in range(self._manager.dimensions)])
        return numpy.array(newMasses)

    def getAtomTypes(self,**kwargs):
        return getattr(self,f"getAtomTypes_{self._initializ['atomType']}")(**kwargs)

    def getAtomTypes_lists(self,Ns,types,**kwargs):
        newTypes = []
        for i in range(len(Ns)):
            for n in range(Ns[i]):
                newTypes.append(types[i])
        return numpy.array(newTypes)

    def getAtomTypes_xyz(self, positionsFile, **kwargs):
        with open(positionsFile,"r") as file:
            lines = file.readlines()
        types = []
        for l in lines[2:]:
            types.append(l.split()[0])
        return numpy.array(types)

    def getPositions_list(self,positionsList,**kwargs):
        return numpy.array(positionsList)

    def getPositions_xyz(self,positionsFile,**kwargs):
        with open(positionsFile,"r") as file:
            lines = file.readlines()
        self._manager.N = len(lines)-2
        print(f"Reading {self._manager.N} atoms")
        positionsList =[[] for l in range(len(lines)-2)]
        for i in range(len(lines)-2):
            for val in lines[i+2].split()[1:]:
                positionsList[i].append(float(val))
        return numpy.array(positionsList)

    def getMomentums(self,**kwargs):
        print("Calculating Momentums using ",self._initializ['momentum'])
        return getattr(self,f"getMomentums_{self._initializ['momentum']}")(**kwargs)

    def getMomentums_MaxwellBoltzmann(self, temperature):
        kT = 8.310549580257024e-7*temperature.asNumber(K)
        Momentums = []
        d = self._manager.dimensions
        for i in range(self._manager.N):
            mass = self._manager.masses[i][0]
            std = numpy.sqrt((kT / mass))
            V = numpy.sqrt(numpy.random.normal(0,std)**2+numpy.random.normal(0,std)**2+numpy.random.normal(0,std)**2)
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

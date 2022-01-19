from unum.units import *
import numpy
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
        print("Getting masses using ", self._initializ['position'])
        return getattr(self,f"getMasses_{self._initializ['mass']}")(**kwargs)

    def getMasses_lists(self,Ns,masses,**kwargs):
        newMasses = []
        for i in range(len(Ns)):
            for n in range(Ns[i]):
                newMasses.append(masses[i].asNumber(kg))
        return numpy.array(newMasses)

    def getAtomTypes(self,**kwargs):
        return getattr(self,f"getAtomTypes_{self._initializ['atomType']}")(**kwargs)

    def getAtomTypes_lists(self,Ns,types,**kwargs):
        newTypes = []
        for i in range(len(Ns)):
            for n in range(Ns[i]):
                newTypes.append(types[i])
        return numpy.array(newTypes)

    def getPositions_list(self,positionsList,**kwargs):
        return numpy.array(positionsList)

    def getMomentums(self,**kwargs):
        print("Calculating Momentums using ",self._initializ['momentum'])
        return getattr(self,f"getMomentums_{self._initializ['momentum']}")(**kwargs)

    def getMomentums_MaxwellBoltzmann(self, temperature):
        kT = 1.38e-23*temperature.asNumber(K)
        Momentums = []
        d = self._manager.dimensions
        for i in range(self._manager.N):
            mass = self._manager.masses[i]
            std = numpy.sqrt((kT / self._manager.masses[i]))
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

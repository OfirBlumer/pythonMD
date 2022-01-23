import numpy
from unum.units import *
class propagator():

    _prop = None
    _manager = None

    def __init__(self,prop,manager):
        self._prop=prop
        self._manager = manager

    def propagate(self, **kwargs):
        getattr(self, f"propagate_{self._prop}")(**kwargs)
        if self._manager.boundariesType=="periodic":
            self._manager.positions -= self._manager.boundaries*numpy.rint(self._manager.positions/self._manager.boundaries)

    def propagate_VelocityVerlet(self, Langevin=False,gamma=None,temperature=None,**kwargs):

        if Langevin:
            kT = 1.38e-23 * temperature.asNumber(K)
            exponent = numpy.exp((-gamma*self._manager.dt/2))
            self._manager.momentums = self._manager.momentums*exponent+((self._manager.masses*kT)*(1-exponent**2))**0.5*numpy.random.standard_normal()
        Fs = self._manager.forces.calculateForce(**kwargs)
        self._manager.momentums -= self._manager.dt*Fs/2
        self._manager.positions+=self._manager.dt*self._manager.momentums/self._manager.masses
        Fs = self._manager.forces.calculateForce(**kwargs)
        self._manager.momentums -= self._manager.dt*Fs/2
        if Langevin:
            self._manager.momentums = self._manager.momentums * exponent + (
                        (self._manager.masses * kT) * (1 - exponent ** 2)) ** 0.5 * numpy.random.standard_normal()
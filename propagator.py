import numpy
from unum.units import *
class propagator():

    _prop = None
    _manager = None

    def __init__(self,prop,manager):
        """
        This class supports propagating the simulation in time
        :param prop: The propagator type; currently supports "VelocityVerlet", "VelocityVerletLangevin","Langevin" (str)
        :param manager: The simulation's main manager (manager class)
        """
        self._prop=prop
        self._manager = manager

    def propagate(self,**kwargs):
        """
        The main function that propagates the simulation in time
        :param kwargs: any parameters needed for the specific propagator
        """
        getattr(self, f"propagate_{self._prop}")(dt=self._manager.dt,**kwargs)
        if self._manager.boundariesType=="periodic":
            self._manager.positions -= self._manager.boundaries*numpy.rint(self._manager.positions/self._manager.boundaries)

    def propagate_VelocityVerlet(self, dt,**kwargs):
        """
        Propagates a single step using velocity Verlet
        :param dt: The time step
        :param kwargs: any parameters needed for calculating the forces
        """
        Fs = self._manager.forces.calculateForce(**kwargs)
        self._manager.momentums += dt*Fs/2
        self._manager.positions+= dt*self._manager.momentums/self._manager.masses
        Fs = self._manager.forces.calculateForce(**kwargs)
        self._manager.momentums += dt*Fs/2

    def propagate_Langevin(self, dt, gamma=None,temperature=None,**kwargs):
        """
        Implements a Langevin thermostat
        :param dt: The time step
        :param gamma: The gamma parameter for the thermostat
        :param temperature: The temperature of the system (int/float*unum units of temperature)
        """
        kT = 8.310549580257024e-7 * temperature.asNumber(K)
        exponent = numpy.exp((-gamma*dt/2))
        self._manager.momentums = self._manager.momentums*exponent+((self._manager.masses*kT)*(1-exponent**2))**0.5*numpy.random.standard_normal()

    def propagate_VelocityVerletLangevin(self, dt, gamma=None, temperature=None, **kwargs):
        """
        Performs Langevin for 1/2 dt, velocity Verlet for dt, Langevin for 1/2 dt;
        More info in these propagators' docs.
        """
        self.propagate_Langevin(dt=dt/2, gamma=gamma,temperature=temperature)
        self.propagate_VelocityVerlet(dt=dt,**kwargs)
        self.propagate_Langevin(dt=dt / 2, gamma=gamma, temperature=temperature)

    def reset(self, resetMethod, **kwargs):
        return False if resetMethod is None else getattr(self, f"reset_{resetMethod}")(**kwargs)

    def reset_Poisson(self,resetRate,**kwargs):
        return self._manager.dt*resetRate>numpy.random.uniform()
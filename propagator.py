import numpy
from unum.units import *

kb_simUnits = 8.310549580257024e-7


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
        :param kwargs: any parameters needed for the specific propagators
        """
        for propStyle in self._prop:
            getattr(self, f"propagate_{propStyle[0]}")(dt=self._manager.dt*propStyle[1],**kwargs)
        if self._manager.boundariesType=="periodic":
            self._manager.positions -= self._manager.boundaries*numpy.fix(self._manager.positions/self._manager.boundaries)

    def propagate_VelocityVerlet(self, dt,**kwargs):
        """
        Propagates a single step using velocity Verlet
        :param dt: The time step
        :param kwargs: any parameters needed for calculating the forces
        """
        Fs = self._manager.forces.calculateForce(**kwargs)
        # print(self._manager.positions)
        # print(self._manager.forces.calculatePotentialEnergy(**kwargs))
        # print(Fs,"\n")
        self._manager.momentums += dt*Fs/2
        self._manager.positions+= dt*self._manager.momentums/self._manager.masses
        Fs = self._manager.forces.calculateForce(**kwargs)
        self._manager.momentums += dt*Fs/2

    def propagate_Langevin(self, dt, gamma=None,temperature=None,**kwargs):
        """
        Implements a Langevin thermostat
        :param dt: The time step
        :param gamma: The gamma parameter for the thermostat
        :param temperature: The temperature of the system in K
        """
        kT = kb_simUnits * temperature
        exponent = numpy.exp((-gamma*dt/2))
        self._manager.momentums = self._manager.momentums*exponent+((self._manager.masses*kT)*(1-exponent**2))**0.5*numpy.random.standard_normal()

    def propagate_CSVR(self,dt, gamma=None, temperature=None,**kwargs):
        """
        THIS FUNCTION DOES A CSVR THERMOSTAT TIME STEP.
        CAN BE EASILY COMBINED WITH VVSTEP.
        Bussi, G.; Parrinello, M. Comput. Phys. Commun. 2008, 179, 26-2.
        Bussi, G.; Donadio, D.; Parrinello, M. J. Chem. Phys. 2007, 126, 014101.

        Parameters
        ----------

        gamma : float
            friction coefficient, units of inverse time.
        dt : float
            Time step for integration.
        temperature : float
            desired temperature.
        """
        kineticEnergies = self._manager.momentums ** 2 / 2 / self._manager.masses
        kineticEnergy = sum(sum(kineticEnergies))
        Ktarget = 0.5*self._manager.N*self._manager.dimensions*kb_simUnits*temperature
        Ndof = self._manager.N*self._manager.dimensions
        c1 = numpy.exp(-2.0 * gamma * dt)
        c2 = (1.0 - c1) * Ktarget / kineticEnergy / Ndof
        r1 = numpy.random.standard_normal()
        nn = Ndof - 1
        if (nn) == 0:
            r2 = 0
        elif (nn) % 2 == 0:
            r2 = 2 * numpy.random.gamma(nn / 2, 1.)
        else:
            rr = numpy.random.standard_normal()
            r2 = 2 * numpy.random.gamma((nn - 1) / 2, 1.) + rr ** 2

        alpha_sq = c1 + c2 * (r1 ** 2 + r2) + 2.0 * r1 * numpy.sqrt(c1 * c2)
        signalpha = numpy.sign(r1 + numpy.sqrt(2 * c1 / c2))
        alpha = signalpha * numpy.sqrt(alpha_sq)

        self._manager.momentums *= alpha


    def reset(self, resetMethod, **kwargs):
        return False if resetMethod is None else getattr(self, f"reset_{resetMethod}")(**kwargs)

    def reset_Poisson(self,resetRate,**kwargs):
        return self._manager.dt*resetRate>numpy.random.uniform()
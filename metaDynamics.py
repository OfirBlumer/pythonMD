withUnits = True
withUnits = False
import numpy
if withUnits:
    from unum.units import *
class metaDynamics():

    _manager = None
    _CVs = None
    _hills = None
    _width = None
    _height = None
    _biasFactor = None
    _pace = None

    @property
    def manager(self):
        return self._manager

    @property
    def CVs(self):
        return self._CVs

    @property
    def hills(self):
        return self._hills

    def __init__(self,manager,CVs=[],sigma=1,height=None,pace=1e100,biasFactor=None):
        """

        """

        self._manager = manager
        self._CVs = CVs
        self._hills = {}
        for CV in CVs:
            self._hills[CV["type"]]=[]
        self._width = sigma**2
        self._height = height
        self._biasFactor = biasFactor
        self._pace = pace

    def metaManager(self,step):
        if step>=self._pace and step%self._pace==0:
            for CV in self.CVs:
                getattr(self,CV["type"])(**CV)

    def singleParticlePosition(self,axis=0,**kwargs):

        # print(f"added hill in {self.manager.positions[0][axis]}")
        self._hills["singleParticlePosition"].append(self.manager.positions[0][axis])

    def calculateForce_singleParticlePosition(self,axis=0,**kwargs):

        Fs = [0 for i in self.manager.positions[0]]
        newF = 0
        q = self.manager.positions[0][axis]
        for center in self._hills["singleParticlePosition"]:
            newF += (q-center)*self._height * numpy.exp(-(q-center)**2/(2*self._width))/self._width
        Fs[axis] = newF
        return numpy.array(Fs)

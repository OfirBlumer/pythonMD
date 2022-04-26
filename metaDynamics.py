withUnits = True
withUnits = False
import numpy
import xarray
if withUnits:
    from unum.units import *
kb_simUnits = 8.310549580257024e-7

class metaDynamics():

    _manager = None
    _CVs = None
    _hills = None
    _width = None
    _sigma = None
    _height = None
    _biasFactor = None
    _T0 = None
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

    def __init__(self,manager,CVs=[],sigma=1,height=1,pace=500,biasFactor=None,temperature=None):
        """
        This class supports performing metaDynamics simulations
        :param CVs: A list of CVs, each a dictionary with the CV type and its parameters.
                       Currently, the available CV is singleParticlePosition (list of dicts)
        :param manager: The simulation's main manager (manager class)
        :param sigma: The width of the added Gaussians (float)
        :param height: The height of the added Gaussians (float)
        :param pace: The pace of the addition of Gaussians (int)
        :param biasFactor: The biasFactor by which the added Gaussians's height decreases with time,
                        not implemented yet (float)
        """

        self._manager = manager
        self._CVs = CVs
        self._hills = {}
        for CV in CVs:
            self._hills[CV["type"]]=[]
        self._sigma = sigma
        self._width = sigma**2
        self._height = height
        self._biasFactor = biasFactor
        self._pace = pace
        if biasFactor is not None:
            self._T0 = temperature*(biasFactor-1)

    def metaManager(self,step):
        """
        Adds a gaussian when needed
        :param step: The current time step of the simulation (int)
        """
        if step>=self._pace and step%self._pace==0:
            for CV in self.CVs:
                getattr(self,CV["type"])(**CV)

    def singleParticlePosition(self,axis=0,**kwargs):
        """
        Adds a Gaussian based on one of the spacial coordinations of a single particle.
        :param axis: The index of the coordinate (int)
        """
        height = self._height
        if self._biasFactor is not None:
            V = 0
            q = self.manager.positions[0][axis]
            for hill in self._hills["singleParticlePosition"]:
                if abs(q - hill[0]) < 3 * self._sigma:
                    V += hill[1] * numpy.exp(-(q - hill[0]) ** 2 / (2 * self._width))
                    height *= numpy.exp(-V/(kb_simUnits*self._T0))
        if height/self._height>0.01:
            self._hills["singleParticlePosition"].append((self.manager.positions[0][axis],height))

    def calculateForce_singleParticlePosition(self,axis=0,**kwargs):
        """
        Calculates the force due to the Gaussians which are based on one of the spacial coordinations of a single particle.
        :param axis: The index of the coordinate (int)
        """
        Fs = [0 for i in self.manager.positions[0]]
        newF = 0
        q = self.manager.positions[0][axis]
        for hill in self._hills["singleParticlePosition"]:
            if abs(q-hill[0]) < 3*self._sigma:
                newF += (q-hill[0])*hill[1] * numpy.exp(-(q-hill[0])**2/(2*self._width))/self._width
        Fs[axis] = newF
        return numpy.array(Fs)

    def getFES(self,file,hillsFile,temperature,bins=50):
        """
        Get the free energy surface based on the different CVs
        :param file: The file which is used to calculate the FES (str)
        :param paceOverFileStep: The pace of the addition of Gaussians over the pace of
                    written lines in the file (float)
        :param temperature: The temperature of the simulation in Kelvin (float)
        :return: A list of pandas dataframes or a single dataframe if there is only one CV.
                    Each dataframe contains the histogram and FES related to its CV.
        """
        fesDataframes = []
        for CV in self.CVs:
            fesDataframes.append(getattr(self, f"getFES_{CV['type']}")(file,hillsFile,temperature,bins=bins,**CV))

        return fesDataframes if len(fesDataframes)>1 else fesDataframes[0]

    def getFES_singleParticlePosition(self, file,hillsFile,temperature,axis=0,bins=50,**kwargs):
        """
        Get the free energy surface based on the singleParticlePosition CV.
        Currently works only for 2D simulations.
        :param file: The file which is used to calculate the FES (str)
        :param paceOverFileStep: The pace of the addition of Gaussians over the pace of
                    written lines in the file (float)
        :param temperature: The temperature of the simulation in Kelvin (float)
        :param axis: The index of the coordinate (int)
        :return: The dataframe which contains the histogram and FES related to this CV.
        """
        with open(file, "r") as newfile:
            lines = newfile.readlines()
        coords = []
        for dim in range(self._manager.dimensions):
            coords.append([float(line.split(" ")[dim]) for line in lines[2:]])

        with open(hillsFile, "r") as newfile:
            lines = newfile.readlines()
        hills = []
        if len(lines)>0:
            if len(lines[0].split(" "))==1:
                for line in lines:
                    line = line.replace("(","").replace(")","").replace(",","")
                    hills.append((float(line.split(" ")[0]), self._height))
            else:
                for line in lines:
                    line = line.replace("(", "").replace(")", "").replace(",", "")
                    hills.append((float(line.split(" ")[0]),float(line.split(" ")[1])))
        weights = []
        for i in range(len(coords[axis])):
            coord = coords[axis][i]
            bias = 0
            for hill in hills[:int(i*len(hills)/len(coords[axis]))]:
                if abs(coord - hill[0]) < 3*self._sigma:
                    bias += hill[1] * numpy.exp(-(coord - hill[0]) ** 2 / 2*(self._width))
            weights.append(numpy.exp(bias / (temperature*kb_simUnits)))

        if self._manager.dimensions==2:

            H, xedges, yedges = numpy.histogram2d(coords[0],coords[1], density=True, bins=bins, weights=weights)
            gridxs = [(xedges[i] + xedges[i - 1]) / 2 for i in range(1, len(xedges))]
            gridys = [(yedges[i] + yedges[i - 1]) / 2 for i in range(1, len(yedges))]
            dataframe = xarray.DataArray(data=H, coords={"x": gridxs, "y": gridys}).to_dataframe(name="Pr").reset_index()
            dataframe = dataframe.loc[dataframe.Pr > 0.0]
            dataframe["fes"] = -temperature*kb_simUnits * numpy.log(dataframe.Pr)
        elif self._manager.dimensions==1:
            H, xedges = numpy.histogram(coords[0], density=True, bins=bins, weights=weights)
            gridxs = [(xedges[i] + xedges[i - 1]) / 2 for i in range(1, len(xedges))]
            dataframe = xarray.DataArray(data=H, coords={"x": gridxs}).to_dataframe(name="Pr").reset_index()
            dataframe = dataframe.loc[dataframe.Pr > 0.0]
            dataframe["fes"] = -temperature*kb_simUnits * numpy.log(dataframe.Pr)
        else:
            raise NotImplementedError("Currently, calculations of FES are implemented only for 1/2D simulations")
        return dataframe
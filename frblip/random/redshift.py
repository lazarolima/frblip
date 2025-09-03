import numpy
from astropy import units
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.stats import rv_continuous
from scipy.stats import gamma # added by Lázaro Lima 09/12/2024


class Redshift(rv_continuous):
    # def __init__(
    #     self,
    #     zmin: float = 0.0,
    #     zmax: float = 6.0,
    #     cosmology: str | None = None,
    #     eps: float = 1e-3,
    # ):

    def __init__(
            self, 
            k: float = 1.4597, 
            theta: float = 0.1542, 
            zmin: float = 0, 
            zmax: float = 6, 
            eps: float = 1e-3, 
            loc: float = -0.0018, 
            cosmology: str | None = None,):
        
        self.k = k  # Shape
        self.theta = theta  # Scale
        self.loc = loc  # Location

        self.zmin = zmin
        self.zmax = zmax
        self.eps = eps

        super().__init__()

        # self.zmin = zmin
        # self.zmax = zmax
        self.cosmology = cosmology

        self.pdf_norm = 1.0 / units.Mpc**3

        ngrid = 1 + int((zmax - zmin) / eps)
        self.zgrid = numpy.linspace(zmin, zmax, ngrid)

        pdf = self.pdf(self.zgrid)

        self.cdf_grid = cumtrapz(x=self.zgrid, y=pdf, initial=0.0)
        self.pdf_norm = self.pdf_norm / self.cdf_grid[-1]
        self.cdf_grid = self.cdf_grid / self.cdf_grid[-1]
        self.isf_grid = 1 - self.cdf_grid

    def _get_support(self) -> tuple[float, float]:

        return self.zmin, self.zmax

    def angular_density(self, z: float) -> units.Quantity:

        diff_co_vol = self.cosmology.differential_comoving_volume(z)
        return diff_co_vol / (1 + z)

    def density(self, z: numpy.ndarray) -> units.Quantity:

        angular_density = self.angular_density(z)
        return units.spat * angular_density

    # def _pdf(self, z: numpy.ndarray) -> numpy.ndarray:

    #     density = self.density(z)
    #     return (self.pdf_norm * density).to(1).value

    def _pdf(self, z: numpy.ndarray) -> numpy.ndarray:
        
        pdf_values = gamma.pdf(z, self.k, scale=self.theta, loc=self.loc)
        
        return pdf_values

    def _logpdf(self, x: numpy.ndarray) -> numpy.ndarray:

        return numpy.log(self._pdf(x))

    def _cdf(self, z: numpy.ndarray) -> numpy.ndarray:

        return numpy.interp(x=z, xp=self.zgrid, fp=self.cdf_grid)

    def _logcdf(self, x: numpy.ndarray) -> numpy.ndarray:

        return numpy.log(self._cdf(x))

    def _sf(self, x: numpy.ndarray) -> numpy.ndarray:

        return 1 - self._cdf(x)

    def _logsf(self, x: numpy.ndarray) -> numpy.ndarray:

        return numpy.log(self._sf(x))

    def _ppf(self, u: numpy.ndarray) -> numpy.ndarray:

        return numpy.interp(x=u, xp=self.cdf_grid, fp=self.zgrid)

    def _isf(self, u: numpy.ndarray) -> numpy.ndarray:

        return numpy.interp(x=u, xp=self.isf_grid, fp=self.zgrid)
    
# Part modified by Lázaro Lima 09/12/2024

class CustomRedshift:
    def __init__(self, k=1.1519, theta=0.1739, zmin=0, zmax=6):
        self.k = k  # Parâmetro de forma
        self.theta = theta  # Parâmetro de escala
        self.zmin = zmin
        self.zmax = zmax


    def generate_samples(self, size):
        """
        Gera amostras da distribuição Gama entre zmin e zmax.
        """
        # Usando a função rvs da distribuição Gama
        samples = gamma.rvs(self.k, scale=self.theta, size=size) 
        
        # Filtrar amostras no intervalo desejado
        samples = samples[(samples >= self.zmin) & (samples <= self.zmax)]
        
        # Garantir o número exato de amostras retornadas
        while len(samples) < size:
            additional_samples = gamma.rvs(self.k, scale=self.theta, size=(size - len(samples)))
            additional_samples = additional_samples[(additional_samples >= self.zmin) & (additional_samples <= self.zmax)]
            samples = numpy.concatenate([samples, additional_samples])

        return samples[:size]
    



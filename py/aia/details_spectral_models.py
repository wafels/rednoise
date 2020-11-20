import numpy as np

from astropy import units as u
from astropy.units import Quantity, UnitsError
from astropy.modeling.core import (Fittable1DModel, Fittable2DModel)
from astropy.modeling.parameters import Parameter, InputParameterError
from astropy.modeling import models


class Const1DAsExponential(Fittable1DModel):
    """
    One dimensional Constant model.

    Parameters
    ----------
    amplitude : float
        Value of the constant function

    See Also
    --------
    Const1D

    Notes
    -----
    Model formula:

        .. math:: f(x) = e^A

    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt

        from astropy.modeling.models import Const1D

        plt.figure()
        s1 = Const1D()
        r = np.arange(-5, 5, .01)

        for factor in range(1, 4):
            s1.amplitude = factor
            plt.plot(r, s1(r), color=str(0.25 * factor), lw=2)

        plt.axis([-5, 5, -1, 4])
        plt.show()
    """

    amplitude = Parameter(default=0)

    @staticmethod
    def evaluate(x, amplitude):
        """One dimensional Constant model function"""

        if amplitude.size == 1:
            # This is slightly faster than using ones_like and multiplying
            x = np.empty_like(x, subok=False)
            x.fill(np.exp(amplitude.item()))
        else:
            # This case is less likely but could occur if the amplitude
            # parameter is given an array-like value
            x = np.exp(amplitude) * np.ones_like(x, subok=False)

        if isinstance(amplitude, Quantity):
            return Quantity(x, unit=amplitude.unit, copy=False)
        return x

    @staticmethod
    def fit_deriv(x, amplitude):
        """One dimensional Constant model derivative with respect to parameters"""

        d_amplitude = np.exp(amplitude) * np.ones_like(x)
        return [d_amplitude]

    @property
    def input_units(self):
        return None

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        return {'amplitude': outputs_unit[self.outputs[0]]}


class SelectModel:
    def __init__(self, model_type):
        self.model_type = model_type

        if self.model_type.lower() == 'pl_c':
            # Power Law plus constant model

            # Power law component and limits
            power_law = models.PowerLaw1D()
            power_law.amplitude.min = 0.0
            power_law.amplitude.max = None
            power_law.alpha.min = 0.0
            power_law.alpha.max = 20.0

            # fix x_0 of power law component
            power_law.x_0.fixed = True

            # Constant component
            constant = models.Const1D()
            constant.amplitude.min = 0.0
            constant.amplitude.max = None

            # Create the model
            self.observation_model = power_law + constant
            self.observation_model.name = 'pl_c'
        elif self.model_type.lower() == 'pl_c_as_exponentials':
            # Power Law plus constant model

            # Power law component and limits
            power_law = models.PowerLaw1D()
            power_law.amplitude.min = 0.0
            power_law.amplitude.max = None
            power_law.alpha.min = 0.0
            power_law.alpha.max = 20.0

            # fix x_0 of power law component
            power_law.x_0.fixed = True

            # Fix the amplitude of the power law component
            power_law.amplitude.fixed = True

            # Amplitude of the power law as an exponential
            amplitude_as_exponential = Const1DAsExponential()
            amplitude_as_exponential.amplitude.min = -100
            amplitude_as_exponential.amplitude.max = None

            # Constant component
            constant = Const1DAsExponential()
            constant.amplitude.min = -100
            constant.amplitude.max = None

            # Create the model
            self.observation_model = amplitude_as_exponential*power_law + constant
            self.observation_model.name = 'pl_c_as_exponentials'
        elif model_type.lower() == 'smoothlybroken_c':
            power_law = models.SmoothlyBrokenPowerLaw1D()

            # Constant component
            constant = models.Const1D()
            constant.amplitude.min = 0.0
            constant.amplitude.max = None

            self.observation_model = power_law + constant
        else:
            raise ValueError('Model not known')

    @property
    def scipy_optimize_options(self):
        # Return the parameter bounds in the format scipy.optimize
        # can use
        param_names = self.observation_model.param_names
        fixed = self.observation_model.fixed
        bounds = []
        for param_name in param_names:
            if not fixed[param_name]:
                bounds.append((self.observation_model.bounds[param_name]))
        return {"bounds": bounds}

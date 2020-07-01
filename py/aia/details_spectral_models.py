
from astropy.modeling import models


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

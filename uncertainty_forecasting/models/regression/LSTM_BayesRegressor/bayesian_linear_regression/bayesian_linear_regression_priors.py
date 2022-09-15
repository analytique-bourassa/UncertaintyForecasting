from pydantic import BaseModel, validator
from typing import Optional

from uncertainty_forecasting.utils.validator import Validator



class BayesianLinearRegressionPriors(BaseModel):
    mean_theta_0: Optional[float] = 1.0
    mean_thetas: Optional[float] = 1.0

    standard_deviation_theta_0: Optional[float] = 2.0
    standard_deviation_thetas: Optional[float] = 2.0

    standard_deviation_sigma: Optional[float] = 10.0
    mean_mu: Optional[float] = 0.0

    @validator('standard_deviation_theta_0',"standard_deviation_thetas", 'standard_deviation_sigma', pre=True, always=True)
    def must_be_positive(cls, v):
        Validator.check_value_is_a_number(v)
        Validator.check_value_strictly_positive(v)
        return v

    @validator('mean_mu', "mean_theta_0", 'mean_thetas', pre=True, always=True)
    def must_be_number(cls, v):
        Validator.check_value_is_a_number(v)
        return v

    class Config:
        validate_assignment = True
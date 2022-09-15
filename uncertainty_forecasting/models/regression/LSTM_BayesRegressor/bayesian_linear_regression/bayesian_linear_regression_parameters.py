from dataclasses import dataclass
from pydantic import BaseModel, validator

from uncertainty_forecasting.utils.validator import Validator

class BayesianLinearRegressionParameters(BaseModel):

    number_of_samples_for_predictions: int = 1000
    number_of_samples_for_posterior: int = 10000
    number_of_tuning_steps: int = 1000
    number_of_iterations: int = 500000

    def set_for_smoke_test(self):
        self.number_of_samples_for_predictions = 1
        self.number_of_samples_for_posterior = 1
        self.number_of_tuning_steps = 1
        self.number_of_iterations = 1

    @validator('number_of_samples_for_predictions',
               "number_of_samples_for_posterior",
               "number_of_tuning_steps",
               "number_of_iterations",
               pre=True, always=True)
    def validate_positive(cls, v):
        Validator.check_value_is_a_number(v)
        Validator.check_value_strictly_positive(v)
        return v

    class Config:
        validate_assignment = True





from enum import Enum


class SamplingMethod(Enum):
    NUTS = "NUTS"
    ADVI_MEAN_FIELD = "ADVI-Mean-Field"
    ADVI_FULL_RANK = "ADVI-full-rank"
    HYBRID = "Hybrid"

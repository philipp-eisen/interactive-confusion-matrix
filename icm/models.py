from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, root_validator, validator


class ENormalize(str, Enum):
    PREDICTED = "predicted"
    ACTUAL = "actual"
    NO = "no"


class ConfusionMatrixRequest(BaseModel):
    predicted: List[float]
    actual: List[float]

    features: Optional[List[str]]
    class_names: Optional[List[str]] = Field(alias="classNames")
    title: Optional[str]
    normalize: Optional[ENormalize] = ENormalize.PREDICTED

    @root_validator
    def validate_lengths_are_equal(cls, values):
        predicted = values.get("predicted")
        actual = values.get("actual")

        assert len(predicted) == len(
            actual
        ), "Length of `predicted` must equal length of `actual`"

        return values

    @validator("class_names")
    def validate_class_names(cls, v, values, **kwargs):
        predicted = values.get("actual")

        assert len(set(predicted)) == len(v), (
            "If provided, there must be as many values in `classNames` as the "
            "cardinality of `actual`"
        )

        return v

    @validator("features")
    def passwords_match(cls, v, values, **kwargs):
        actual = values.get("actual")
        assert len(v) == len(actual), (
            "If provided, length of `features` must equal the "
            "length of `actual` and `predicted`"
        )
        return v

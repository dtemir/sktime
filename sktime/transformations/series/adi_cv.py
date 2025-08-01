# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Feature transformer that returns features of time series including categories."""

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer

__author__ = ["shlok191"]


class ADICVTransformer(BaseTransformer):
    r"""Transformer categorizing series into ADI-CV2 classes after Syntetos/Boylan.

    Transforms a time series into a category label, which is one of:
    ``"smooth"``, ``"erratic"``, ``"intermittent"``, or ``"lumpy"``.

    The labels are based on the Average Demand Interval (ADI) and the Coefficient of
    Variation squared (CV^2) of the time series, using simple thresholding.
    Optionally, the transformer can return the ADI and CV^2 values as well.

    Let :math:`x_t` be the value of the time series at times :math:`t = 1, 2, \dots, T`,
    and let :math:`N` be the number of non-zero values in the time series, i.e.,
    :math:`N = \text{card}\{t: x_t \neq 0\}`.

    The ADI and CV^2 are calculated as follows:

    1. Average Demand Interval (ADI): The average number of periods between
    periods with non-zero demand, mathematically defined as:

    .. math:: ADI = \frac{T}{N}

    2. Coefficient of Variation squared (CV^2): CV is calculated on non-zero values
    in the time series. Mathematical definition is:

    .. math:: CV2 = \frac{1}{N}\sum_{t=1}^{T} x_t^2 - \left(\frac{1}{N}\sum_{t=1}^{T} x_t\right)^2

    3. Class: Classification of time series on basis of ADI threshold
    and CV^2 threshold.

    For thresholds ``ADI_threshold`` and ``CV2_threshold``, the classes are:

    1. Smooth: If ``ADI <= adi_threshold`` and ``CV2 <= cv2_threshold``
    2. Erratic: If ``ADI <= adi_threshold`` and ``CV2 > cv2_threshold``
    3. Intermittent: If ``ADI > adi_threshold`` and ``cv2 <= cv2_threshold``
    4. Lumpy: if ``ADI > adi_threshold`` and ``CV2 > cv2_threshold``

    Default values for the thresholds are taken from the paper by Syntetos/Boylan [1].
    namely, ``adi_threshold = 1.32`` and ``cv2_threshold = 0.49``.
    They can also be adjusted by passing them as parameters to the transformer.

    Parameters
    ----------
    adi_threshold : float (default = 1.32)
        Specifies the ADI threshold utilized for classifying the time series

    cv2_threshold : float (default = 0.49)
        Specifies the CV2 threshold utilized for classifying the time series

    features : list[str] | None (default = ['adi', 'cv2', 'class'])
        Specifies all of the feature values to be calculated

    adi_trim_handling: string (default = pool)
        Specifies the method for reconciling leading/trailing zeros in the series.
        Allowable values are ('pool', 'trim', and 'ignore'), corresponding to the
        following treatment of leading/trailing zeros:
        pool => L / N
        trim => (Last N - First N) / N
        ignore => L / (N - 1)
            - where L is the set of All Observations, and N is the set of Non-Zero observations

    Examples
    --------
    >>> from sktime.transformations.series.adi_cv import ADICVTransformer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = ADICVTransformer()
    >>> y_hat = transformer.fit_transform(y)

    References
    ----------
    [1]: John E. Boylan, Aris Syntetos: The Accuracy of Intermittent
    Demand Estimates. International Journal of Forecasting, 1 Apr. 2005
    """  # noqa: E501

    _tags = {
        # packaging info
        # --------------
        "authors": ["shlok191", "sbhobbes"],
        "maintainers": ["shlok191"],
        # estimator type
        # --------------
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Primitives",
        "scitype:instancewise": False,
        "scitype:transform-labels": "None",
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "None",
        "univariate-only": True,  # Demand being the only variable
        "requires_y": False,
        "fit_is_empty": True,
        "capability:inverse_transform": False,
        "capability:unequal_length": False,
        "capability:missing_values": False,
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
    }

    def __init__(
        self,
        features=None,
        adi_threshold=1.32,
        cv_threshold=0.49,
        adi_trim_handling="pool",
    ):
        """Initialize the transformer and process any provided parameters.

        Parameters
        ----------
        features : List[str] | None, optional
            List of features to compute. Defaults to None (all features)

        adi_threshold : float, optional
            Threshold for Average Demand Interval. Defaults to 1.32.

        cv_threshold : float, optional
            Threshold for Variance. Defaults to 0.49.

        adi_trim_handling : str, optional
            Determines how leading and trailing zeros are handled for ADI.
            Defaults to 'pool'

        Raises
        ------
            ValueError: If features is provided and does not
            contain 'adi','cv2', or 'class'.
        """
        self.adi_threshold = adi_threshold
        self.cv_threshold = cv_threshold
        self.features = features
        self.adi_trim_handling = adi_trim_handling

        self.features_internal = features

        # Initialize the parent class
        super().__init__()

        # Checking if the features parameter is valid
        if features is not None:
            if not all(feature in ["adi", "cv2", "class"] for feature in features):
                raise ValueError(
                    "Error in ADICVTransformer: Invalid features list provided. "
                    "The features argument must either be None, or a list of str "
                    "containing one or multiple of the valid feature strings "
                    "('adi', 'cv2', 'class')."
                )

        else:
            # Helpful to transform None to default list for transform function
            self.features_internal = ["adi", "cv2", "class"]

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : Series
            Series of time series data this transform function aims to classify

        y : Series | None, default=None
            Not required for the classification of the time series

        Returns
        -------
        X_transformed : pd.DataFrame

            The returned DataFrame consists of the columns in the features list passed
            during initialization. Specifically, the columns include (by default):

                1. Average Demand Interval (ADI)
                2. Coefficient of Variation squared (CV2)
                3. categorical class
        """
        X_non_zero_indices = X.to_numpy().nonzero()[0]
        X_non_zero_values = X.iloc[X_non_zero_indices].values

        # Calculating ADI value based on adi_trim_handling parameter
        n_non_zero_observations = len(X_non_zero_indices)
        n_observations = len(X)

        if self.adi_trim_handling == "trim":
            first_non_zero = X_non_zero_indices[0]
            last_non_zero = X_non_zero_indices[-1]

            adi_numerator = last_non_zero - first_non_zero
            adi_denominator = n_non_zero_observations - 1

        elif self.adi_trim_handling == "pool":
            adi_numerator = n_observations
            adi_denominator = n_non_zero_observations

        elif self.adi_trim_handling == "ignore":
            adi_numerator = n_observations
            adi_denominator = n_non_zero_observations - 1

        adi_value = adi_numerator / adi_denominator

        # Calculating coefficient of variation squared for all non-zero values
        sigma = np.std(X_non_zero_values)
        mu = np.mean(X_non_zero_values)
        cv2_value = (sigma / mu) ** 2

        # Calculating the class type
        adi_low = adi_value <= self.adi_threshold
        cv2_low = cv2_value <= self.cv_threshold
        adi_high = not adi_low
        cv2_high = not cv2_low

        if adi_low and cv2_low:
            class_type = "smooth"
        elif adi_low and cv2_high:
            class_type = "erratic"
        elif adi_high and cv2_low:
            class_type = "intermittent"
        elif adi_high and cv2_high:
            class_type = "lumpy"

        # Collecting all values together into dict and converting to DF
        return_dict = {}

        if "adi" in self.features_internal:
            return_dict["adi"] = [adi_value]

        if "cv2" in self.features_internal:
            return_dict["cv2"] = [cv2_value]

        if "class" in self.features_internal:
            return_dict["class"] = [class_type]

        df = pd.DataFrame(return_dict)

        # Ordering the dataframe in the correct order
        df = df.loc[:, self.features_internal]

        return df

    # todo: return default parameters, so that a test instance can be created
    #   required for automated unit and integration testing of estimator
    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests.
            If no special parameters are defined for a value, will return
            `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test case
        """
        # Testing with 0 thresholds for both thresholds
        # in independent test cases!

        params1 = {}
        params2 = {"features": None, "adi_threshold": 1.32, "cv_threshold": 0.49}
        params3 = {
            "features": ["adi", "class"],
            "adi_threshold": 1.5,
            "cv_threshold": 0.2,
            "adi_trim_handling": "trim",
        }

        return [params1, params2, params3]

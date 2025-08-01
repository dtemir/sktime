"""BOSS test code."""

import numpy as np
import pytest

from sktime.classification.dictionary_based import BOSSEnsemble, IndividualBOSS
from sktime.datasets import load_unit_test
from sktime.tests.test_switch import run_test_for_class


@pytest.fixture
def dataset():
    """Load unit_test train and test data set from sktime.

    :return: tuple, (X_train, y_train, X_test, y_test).
    """
    X_train, y_train = load_unit_test(split="train")
    X_test, y_test = load_unit_test(split="test")
    return (X_train, y_train, X_test, y_test)


@pytest.mark.skipif(
    not run_test_for_class(IndividualBOSS),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "new_class,expected_dtype",
    [
        ({"1": "Class1", "2": "Class2"}, "same"),
        ({"1": 1, "2": 2}, int),
        ({"1": 1.0, "2": 2.0}, float),
        ({"1": True, "2": False}, bool),
    ],
)
def test_individual_boss_classes(dataset, new_class, expected_dtype):
    """Test Individual BOSS on unit_test data with different datatypes as classes."""
    # load unit test data
    X_train, y_train, X_test, y_test = dataset

    # change class
    y_train = np.array([new_class[y] for y in y_train])

    if expected_dtype == "same":
        expected_dtype = y_train.dtype

    # train iboss and predict X_test
    iboss = IndividualBOSS()
    iboss.fit(X_train, y_train)
    y_pred = iboss.predict(X_test)

    # assert class type and names
    assert y_pred.dtype == expected_dtype
    assert set(y_pred) == set(y_train)


@pytest.mark.skipif(
    not run_test_for_class(BOSSEnsemble),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "new_class,expected_dtype",
    [
        ({"1": "Class1", "2": "Class2"}, "<U6"),
        ({"1": 1, "2": 2}, int),
        ({"1": 1.0, "2": 2.0}, float),
        ({"1": True, "2": False}, bool),
    ],
)
def test_boss_ensemble_classes(dataset, new_class, expected_dtype):
    """Test BOSS Ensemble on unit_test data with different datatypes as classes."""
    # load unit test data
    X_train, y_train, X_test, y_test = dataset

    # change class
    y_train = np.array([new_class[y] for y in y_train])

    # train boss_ensemble and predict X_test
    boss_ensemble = BOSSEnsemble()
    boss_ensemble.fit(X_train, y_train)
    y_pred = boss_ensemble.predict(X_test)

    # assert class type and names
    assert y_pred.dtype == expected_dtype
    assert set(y_pred) == set(y_train)


@pytest.mark.skipif(
    not run_test_for_class(IndividualBOSS),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_individual_boss_histograms(dataset):
    """Test Individual BOSS histograms_ attribute type and size."""
    X_train, y_train, _, _ = dataset

    iboss = IndividualBOSS(
        window_size=10, word_length=4, alphabet_size=2, store_histogram=True
    )
    iboss.fit(X_train, y_train)

    assert isinstance(iboss.histograms_, list), "histograms_ should be a list"
    assert len(iboss.histograms_) == len(X_train), (
        f"histograms_ length ({len(iboss.histograms_)}) should be equal to "
        f"number of training instances ({len(X_train)})"
    )

    for hist in iboss.histograms_:
        assert isinstance(hist, dict), "each histogram should be a dict"
        if hist:  # only check if not empty
            key = next(iter(hist.keys()))
            value = next(iter(hist.values()))
            assert isinstance(key, str), (
                f"histogram key should be str, found {type(key)}"
            )
            assert isinstance(value, int), (
                f"histogram value should be int, found {type(value)}"
            )


@pytest.mark.skipif(
    not run_test_for_class(IndividualBOSS),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_individual_boss_no_histograms(dataset):
    """Test Individual BOSS with store_histogram=False."""
    X_train, y_train, _, _ = dataset

    iboss = IndividualBOSS(
        window_size=10, word_length=4, alphabet_size=2, store_histogram=False
    )
    iboss.fit(X_train, y_train)

    assert iboss.histograms_ == [], "histograms_ should be an empty list"

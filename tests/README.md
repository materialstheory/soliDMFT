# Automated testing

This folder contains unit tests, i.e. short snippets of code that test certain cases for specific functions in the code.
In the future, integration tests, which test the program as a whole by for example performing a standard DMFT iteration, could be implemented as well.

## Unit tests

The framework we use is [pytest](https://docs.pytest.org/en/latest/).
If you create a docker image with the docker files in the directory `/Docker`, pytest is already installed.
Inside the docker (or any environment with the triqs libraries and pytest) and from the main folder `uni-dmft`, you can execute it by running `python -m pytest`.
pytest will find every file starting with "test" and execute every function in there that itself starts with "test".
Therefore, every function starting with "test" has to contain one or multiple standalone tests, including setup of input parameters and checking output with the `assert` keyword.

You can run a single test, for example the test `test_add_dft_values_one_impurity_one_band` in `test_observables.py`,  by going to the main directory and execute it as:
```
python -m pytest tests/test_observables.py::test_add_dft_values_one_impurity_one_band
```

Notes:

* When developing code, please write tests as soon as you know what a function is supposed to do.
This saves debugging later, which potentially makes developing faster, and we get a better test coverage of the overall code.
* When comparing floats, `numpy.isclose()` with reasonable tolerance parameters `atol` and `rtol` is more stable than float comparison with `==`
* The `Dummy()` object is the same as a python `Object()` except that we can actually add attributes to it.
We use it to set up for example mock SumkDFT object.

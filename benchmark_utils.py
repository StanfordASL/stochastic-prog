import numpy as np


def compute_variance_metric(xs: np.array) -> float:
    """Computes a variance metric.

    Given an array of N variables xi contained in xs, 
    returns the variance metric
        sigma_hat_N^2 = (1 / (N - 1)) * 
            sum_{i=1}^N ||xi - mean(xi)||^2.
    Returns NaN if a single element is provided (N is 1).

    Args:
        xs: array of variables batched over the first axis
            (batch_size, x_size) array

    Returns:
        variance_metric: sigma_hat_N^2
            scalar
    """
    if not(isinstance(xs, np.ndarray)):
        raise ValueError('xs should be a numpy array.')
    if xs.ndim != 2:
        raise ValueError('xs should have two dimensions.')
    batch_size = xs.shape[0]
    if batch_size < 2:
        variance_metric = float("nan")
        return variance_metric

    xs_mean = np.mean(xs, axis=0)
    xs_norms = np.linalg.norm(xs - xs_mean, axis=-1)
    variance_metric = (
        1 / (batch_size - 1)) * np.sum(xs_norms**2)
    return variance_metric


def compute_error_metric(xs: np.array, x_true) -> float:
    """Computes an error metric between xs and x_true.

    Given an array of N variables xi contained in xs, 
    returns the median (over i) of the error metric
                    ||xi - x_true||.

    Args:
        xs: array of variables batched over the first axis
            (batch_size, x_size) array
        x: reference variable
            (x_size) array

    Returns:
        error_metric: err_hat_N
            scalar
    """
    if not(isinstance(xs, np.ndarray)):
        raise ValueError('xs should be a numpy array.')
    batch_size = xs.shape[0]

    xs_err = xs - x_true
    xs_err_norms = np.linalg.norm(xs_err, axis=-1) / np.linalg.norm(x_true)
    error_metric = np.median(xs_err_norms)
    return error_metric


if __name__=="__main__":
    print("[benchmark_utils.py]")
    test_success = True
    error_msg = "compute_variance_metric should catch"
    error_msg += "this wrong input."

    xs = np.ones((100, 5))
    variance_metric = compute_variance_metric(xs)
    try:
        compute_variance_metric(4.0)
        raise ValueError(error_msg)
    except ValueError:
        pass
    try:
        compute_variance_metric(np.ones((6, 6, 7)))
        raise ValueError(error_msg)
    except ValueError:
        pass
    nan_handling_success = np.isnan(
        compute_variance_metric(np.ones((1, 6))))
    if not(nan_handling_success):
        raise ValueError("Should return nan if only one xi.")
    test_success = test_success & nan_handling_success
    if test_success:
        print("Test successful.")
    else:
        print("Test failed.")
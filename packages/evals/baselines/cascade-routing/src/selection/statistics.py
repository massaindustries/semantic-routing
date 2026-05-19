from scipy.stats import norm, qmc
import numpy as np

computed_random_numbers_sobol = {

}

def transform_samples(samples, qualities, sigma_qualities, independent=False):
    """
    Transform the given samples to compute the maximum values of the transformed samples.
    Args:
        samples: numpy.ndarray
                The input samples to be transformed.
        qualities: numpy.ndarray
                The qualities to be added to the transformed samples.
        sigma_qualities: numpy.ndarray
                The covariance matrix of the qualities.
        independent: bool, optional
                If True, the transformation is done independently for each model.
                If False, the transformation is done using the covariance matrix.
    Returns:
        mean_max: float
            The mean of the maximum values of the transformed samples.
        var_max: float
            The variance of the maximum values of the transformed samples.
    """
    if independent:  # Handle independent case
        transformed_samples = samples * np.diag(sigma_qualities) + qualities
    else:
        L = np.linalg.cholesky(sigma_qualities + 1e-12 * np.eye(sigma_qualities.shape[0]))
        transformed_samples = samples @ L.T + qualities
    
    max_samples = np.max(transformed_samples, axis=1)
    return float(np.mean(max_samples)), float(np.var(max_samples))

def compute_expected_max(qualities, sigma_qualities, independent=False, n_samples=100):
    """
    Compute the expected maximum value of qualities.
    Args:
        qualities (ndarray): Array of qualities.
        sigma_qualities (ndarray or None): Array of standard deviations of qualities, or covariance matrix. 
            If None, the maximum value of qualities is returned without uncertainty.
        independent (bool): Whether the qualities are independent. Default is False.
        n_samples (int): Number of samples used for Monte Carlo integration. Default is 100.
    Returns:
        expected_max (float): Expected maximum value of qualities.
        uncertainty (float or None): Uncertainty of the expected maximum value. None if sigma_qualities is None.
    """
    if sigma_qualities is None:
        return float(np.max(qualities)), None
    n = qualities.shape[0]
    if n == 1:
        return float(qualities), float(sigma_qualities[0, 0])
    elif n == 0:
        return -np.inf, None
    
    if (n, n_samples) in computed_random_numbers_sobol:
        normal_samples = computed_random_numbers_sobol[(n, n_samples)]
    else:
        sampler = qmc.Sobol(d=n, scramble=True)
        qmc_samples = sampler.random(n_samples)
        normal_samples = norm.ppf(qmc_samples)
        computed_random_numbers_sobol[(n, n_samples)] = normal_samples
    
    return transform_samples(normal_samples, qualities, sigma_qualities, independent)
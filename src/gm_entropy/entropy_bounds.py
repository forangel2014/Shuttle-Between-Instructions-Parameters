# import jax.numpy as jnp
# from jax.scipy.stats import multivariate_normal
import torch
from torch.distributions import MultivariateNormal
import math

def calculate_z_ij(mean_i, mean_j, cov_i, cov_j):
    combined_cov = cov_i + cov_j  # Sum of covariance matrices
    #z_ij = multivariate_normal.pdf(mean_i, mean_j, combined_cov)
    
    diff = mean_i - mean_j
    d = len(diff)
    det_cov = torch.prod(combined_cov)
    inv_cov = 1.0 / combined_cov
    exponent = -0.5 * torch.sum(diff * inv_cov * diff)
    z_ij = (2 * torch.pi) ** (-d/2) * det_cov ** (-0.5) * torch.exp(exponent)
    return z_ij

def calculate_log_z_ij(mean_i, mean_j, cov_i, cov_j):
    combined_cov = cov_i + cov_j  # 协方差矩阵之和
    
    diff = mean_i - mean_j
    d = len(diff)
    log_det_cov = torch.sum(torch.log(combined_cov))
    inv_cov = 1.0 / combined_cov
    
    log_pdf = -0.5 * (d * math.log(2 * torch.pi) + log_det_cov + torch.sum(diff * inv_cov * diff))
    return log_pdf

def EntropyLowerBoundEst(gmm_params):
    weights, means, covariances = gmm_params
    
    L = len(weights)
    H_l = 0.0

    for i in range(L):
        sum_term = 0.0
        for j in range(L):
            z_ij = calculate_z_ij(means[i], means[j], covariances[i], covariances[j])
            sum_term += weights[j] * z_ij
        #H_l += weights[i] * jnp.log(sum_term)
        H_l += weights[i] * torch.log(sum_term)
    H_l = -H_l
    return H_l

def EntropyLowerBoundEstLogScale(gmm_params):
    weights, means, covariances = gmm_params
    
    L = len(weights)
    H_l = 0.0

    sum_terms = torch.zeros(L, device=weights.device)
    for i in range(L):
        log_terms = torch.zeros(L, device=weights.device)
        for j in range(L):
            log_z_ij = calculate_log_z_ij(means[i], means[j], covariances[i], covariances[j])
            log_terms[j] = torch.log(weights[j]) + log_z_ij
        sum_terms[i] = torch.logsumexp(log_terms, dim=0)
    
    H_l = -torch.sum(weights * sum_terms)
    return H_l

def EntropyUpperBoundEst(weights, covariances):
    L = len(weights)
    N = covariances[0].shape[0] 
    H_u = 0.0

    for i in range(L):
        
        # Log determinant of covariance matrix
        log_det_cov = jnp.linalg.slogdet(covariances[i])[1]  
        term = -jnp.log(weights[i]) + 0.5 * jnp.log((2 * jnp.pi * jnp.e) ** N * jnp.exp(log_det_cov))
        H_u += weights[i] * term

    return H_u

if __name__ == '__main__':
    # GMM parameters
    weights = jnp.array([0.125 for _ in range(8)]) 
    means = [jnp.array([0.0, 0.0, 0.0]) for _ in range(8)] 
    covariances = [jnp.eye(3) for _ in range(8)]  
    
    
    # Calculate the entropy lower bound
    gmm_params = (weights, means, covariances)
    H_l = EntropyLowerBoundEst(gmm_params)
    H_u = EntropyUpperBoundEst(weights, covariances)
    print(H_l)
    print(H_u)



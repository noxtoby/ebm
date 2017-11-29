# Authors: Nicholas C. Firth <ncfirth87@gmail.com>, Neil P. Oxtoby <noxtoby@gmail.com>
# License: TBC
from ebm import mixture_model
from ebm import mcmc
from ebm import plotting
from ebm import datasets
from ebm import distributions
from matplotlib import pyplot as plt


def main():
    # An example of how to produce a more conservative estimate of uncertainty 
    # in the EBM sequence, by fitting multiple EBMs to bootstrap samples from 
    # the data
    
    # Load the data
    X, y, bmname, cname = datasets.load_synthetic('synthetic_1500_10.csv')
    
    # Fit the mixture models
    mixture_models = mixture_model.fit_all_gmm_models(X, y)
    fig, ax = plotting.mixture_model_grid(X, y, mixture_models,
                                          score_names=bmname,
                                          class_names=cname)
    fig.show()
    
    # Fit our disease sequence, using greedy ascent followed by MCMC optimisation
    samples = mcmc.mcmc(X, mixture_models, n_iter=200,
                        greedy_n_iter=10, greedy_n_init=2)
    ml_order = samples[0] # MLE of sequence
    
    # Plot a positional variance diagram to visualise uncertainty in the sequence
    # The uncertainty here is over-confident: it's uncertainty in the mean
    fig, ax = plotting.mcmc_uncert_mat(samples, score_names=bmname)
    fig.show()
    
    # Bootstrapping
    bs_samples = mcmc.bootstrap_ebm(X, y, n_mcmc_iter=200,
                                    n_bootstrap=10, greedy_n_init=2,
                                    greedy_n_iter=10)
    # Plot a positional variance diagram to visualise uncertainty in the bootstrap sequence
    # The uncertainty here is more conservative: it's uncertainty in the mean, combined across bootstrapped samples
    fig, ax = plotting.mcmc_uncert_mat(bs_samples, ml_order=ml_order,
                                       score_names=bmname)
    fig.show()
    plt.show()


if __name__ == '__main__':
    import numpy
    numpy.random.seed(42)
    main()

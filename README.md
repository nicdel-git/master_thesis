# Gaussian process regression methods for option pricing

This git repository groups my master thesis (in pdf and latex form), the presentation slides of the oral, and the code used for the implementation.

The thesis is divided into three parts/implementations:
1. The first is a Gaussian process that takes as input a stock price $S$ and a time $t$, and outputs the posterior distribution of the associated call price. It then derives this posterior distribution to obtain the Greeks of that call price (namely Delta and Theta).
2. The second is a functional Gaussian process that takes as input a payoff function, and optionally also a time-to-maturity $T$, and outputs the price of the European option described by that payoff function. The payoff function can be chosen freely.
3. The third is a multitask Gaussian process that takes as input a strike price $K$ and a time-to-maturity $T$, and outputs both the American call option price and the European call option price associated to those inputs.

The code is also divided into three jupyter notebook files:
1. The code for the first part is in the "greeks_GP" file.
2. The code for the second part is in the "functional_GP" file.
3. The code for the third part is in the "multitask_GP" file.

The additional files in the code section are there to keep the notebooks less cluttered:
1. The "BS_data" file is used to generate Black-Scholes training and test data.
2. The "analytic_variance" file is used to compute the analytic derivatives of the GP model in the first part/notebook.
3. The "custom_kernels" file is used to define custom kernels for the Gaussian processes, using the TensorFlow framework as a basis.
4. The "finite_elements" file is not used in any of the notebooks, but is included in case a user would like to use the finite element method to estimate the price of any European payoff function (instead of using the Functional GP from the second part).

The following packages are required:
* numpy
* pandas
* scipy
* math
* tensorflow
* tensorflow_probability
* matplotlib
* sklearn
* torch
* gpytorch

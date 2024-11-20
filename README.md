```{r}
#question 1 part a
# Define the true values of the regression coefficients
beta_0 <- 0.1
beta_1 <- 1.1
beta_2 <- -0.9

# Define the sample sizes for N = 10, 50, 100
sample_sizes <- c(10, 50, 100)

# Print the selected regression coefficients
cat("The true values of the regression coefficients are:\n")
cat("Beta_0 = ", beta_0, "\n")
cat("Beta_1 = ", beta_1, "\n")
cat("Beta_2 = ", beta_2, "\n")

# Print the sample sizes
cat("\nSynthetic sample sizes to use: N = ", sample_sizes, "\n")
```





```{r}
#question 1 part b


# Define the true values of the regression coefficients (from Part a)
beta_0 <- 0.1
beta_1 <- 1.1
beta_2 <- -0.9

# Define the sample sizes for N = 10, 50, 100
sample_sizes <- c(10, 50, 100)

# Loop through each sample size
for (N in sample_sizes) {
  # Generate N random values for x1 and x2 from Uniform[-2, 2]
  X1 <- runif(N, min = -2, max = 2)  # Covariate x1
  X2 <- runif(N, min = -2, max = 2)  # Covariate x2
  
  # Print the generated covariates for the current sample size
  cat("\nGenerated covariates (X1, X2) for N =", N, "samples:\n")
  print(data.frame(X1, X2))
}
```


```{r}
#question 1 part c 
# Part (c) code to generate binary responses
# Define the true values of the regression coefficients
beta_0 <- 0.1
beta_1 <- 1.1
beta_2 <- -0.9

# Generate the binary responses for each sample size
for (N in sample_sizes) {
  # Generate covariates x1 and x2 for the n-th sample
  X1 <- runif(N, min = -2, max = 2)  # Covariate x1 from Uniform[-2, 2]
  X2 <- runif(N, min = -2, max = 2)  # Covariate x2 from Uniform[-2, 2]
  
  # Calculate the logit (linear predictor) for each observation n
  logit <- beta_0 + beta_1 * X1 + beta_2 * X2
  
  # Calculate the probability of y_i^{(n)} = 1 for each (x_1^{(n)}, x_2^{(n)})
  prob_y1 <- 1 / (1 + exp(-logit))
  
  # Generate binary responses y_i^{(n)} based on the calculated probabilities
  Y <- rbinom(N, size = 1, prob = prob_y1)
  
  # Output the generated data
  cat("\nGenerated data for N =", N, "samples:\n")
  cat("Covariates (X1, X2):\n")
  print(data.frame(X1, X2))
  cat("Binary responses (Y):\n")
  print(Y)
  
  # Report the proportion of 1's and 0's in the sample
  cat("\nProportion of 1's and 0's in the sample:", table(Y) / N, "\n")
}

```
```{r}
#question 2 
# Load required libraries
library(MASS)  # For mvrnorm and dmvnorm
library(mvtnorm)

# Set true parameter values for logistic regression
beta_true <- c(0.1, 1.1, -0.9)

# Define logistic regression model to generate data
logistic_model <- function(X1, X2, beta) {
  z <- beta[1] + beta[2] * X1 + beta[3] * X2
  return(1 / (1 + exp(-z)))
}

# Set the three sample sizes (N)
sample_sizes <- c(10, 50, 100)

# Number of samples for importance sampling
n_samples <- 20000  # Further increased to improve stability

# Number of resamples to estimate the posterior
n_resamples <- 10000

# Define spherical Gaussian proposal (mean 0, variance controlled by c)
c_value <- 15  # Further increased variance for better coverage
proposal_cov <- c_value * diag(3)  # Spherical covariance matrix for 3 parameters

# Function to perform Bayesian inference via importance sampling
importance_sampling <- function(N) {
  # Generate synthetic data
  set.seed(42)
  X1 <- rnorm(N)
  X2 <- rnorm(N)
  probs <- logistic_model(X1, X2, beta_true)
  Y <- rbinom(N, 1, probs)
  
  # Define the likelihood function for logistic regression
  likelihood <- function(beta, X1, X2, Y) {
    p <- logistic_model(X1, X2, beta)
    return(prod((p^Y) * ((1 - p)^(1 - Y))))  # Likelihood for all data points
  }
  
  # Importance sampling step: draw samples from the spherical Gaussian proposal
  beta_samples <- mvrnorm(n_samples, mu = rep(0, 3), Sigma = proposal_cov)
  
  # Compute importance weights for each sample
  weights <- numeric(n_samples)
  for (i in 1:n_samples) {
    weights[i] <- likelihood(beta_samples[i, ], X1, X2, Y) / dmvnorm(beta_samples[i, ], mean = rep(0, 3), sigma = proposal_cov)
  }
  
  # Regularize weights to prevent extreme values
  threshold <- quantile(weights, 0.95)  # Regularize only the top 5% of weights
  weights[weights > threshold] <- threshold
  
  # Normalize weights
  weights <- weights / sum(weights)
  
  # Estimate posterior means and standard errors using the weighted samples
  posterior_means <- colSums(beta_samples * weights)
  posterior_vars <- colSums((beta_samples - posterior_means)^2 * weights)
  posterior_ses <- sqrt(posterior_vars)
  
  # Resample from the weighted samples
  resampled_indices <- sample(1:n_samples, n_resamples, replace = TRUE, prob = weights)
  resampled_betas <- beta_samples[resampled_indices, ]
  
  # Set up a layout for plotting on a single sheet
  dev.new()
  par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))  # Adjust margins and layout
  
  hist(resampled_betas[, 1], main = paste("Posterior of β0 (N=", N, ")"), xlab = "β0", breaks = 30, col = 'skyblue', border = 'white')
  hist(resampled_betas[, 2], main = paste("Posterior of β1 (N=", N, ")"), xlab = "β1", breaks = 30, col = 'skyblue', border = 'white')
  hist(resampled_betas[, 3], main = paste("Posterior of β2 (N=", N, ")"), xlab = "β2", breaks = 30, col = 'skyblue', border = 'white')
  
  # Effective sample size
  ess <- (sum(weights)^2) / sum(weights^2)
  
  # Fit logistic regression using standard optimization (MLE)
  data <- data.frame(X1 = X1, X2 = X2, Y = Y)
  fit <- glm(Y ~ X1 + X2, family = binomial, data = data)
  mle_means <- coef(fit)
  mle_ses <- summary(fit)$coefficients[, "Std. Error"]
  
  return(list(posterior_means = posterior_means, posterior_ses = posterior_ses, ess = ess, mle_means = mle_means, mle_ses = mle_ses, resampled_betas = resampled_betas))
}

# Run for each sample size
results <- list()

for (N in sample_sizes) {
  cat("Processing for N =", N, "...\n")
  result <- importance_sampling(N)
  results[[as.character(N)]] <- result
}

# Compare results for different sample sizes
for (N in sample_sizes) {
  cat("\nPosterior means and standard errors for N =", N, ":\n")
  print(results[[as.character(N)]]$posterior_means)
  print(results[[as.character(N)]]$posterior_ses)
  cat("Effective sample size for N =", N, ":\n")
  print(results[[as.character(N)]]$ess)
  
  cat("\nMLE means and standard errors for N =", N, ":\n")
  print(results[[as.character(N)]]$mle_means)
  print(results[[as.character(N)]]$mle_ses)
}
```


```{r}
#question 3 

library(MASS)  # For mvrnorm

# Extend true parameters to 9 dimensions
set.seed(42)
beta_additional <- runif(6, -1, 1)  # Generate six additional beta values from Uniform[-1, 1]
beta_true <- c(0.1, 1.1, -0.9, beta_additional)  # Combine original 3 betas with the additional 6 betas

cat("The six additional beta values are:\n")
print(beta_additional)

# Function to simulate logistic regression probabilities
logistic_model <- function(X, beta) {
  z <- X %*% beta
  return(1 / (1 + exp(-z)))
}

# Extended sample sizes and importance sampling setup
sample_sizes <- c(10, 50, 100)
n_samples <- 20000
n_resamples <- 10000
c_value <- 1
proposal_cov <- c_value * diag(9)  # 9x9 covariance matrix for the proposal

# Function to perform Bayesian inference via importance sampling using NIS
importance_sampling <- function(N) {
  # Simulate data with intercept and 8 covariates (from Uniform[-2, 2])
  set.seed(42)
  X <- cbind(1, matrix(runif(N * 8, min = -2, max = 2), ncol = 8))  # Add intercept as the first column
  
  # Compute probabilities using the logistic model
  probs <- logistic_model(X, beta_true)
  
  # Simulate the response variable Y from a binomial distribution
  Y <- rbinom(N, 1, probs)
  
  # Log-likelihood function for logistic regression
  log_likelihood <- function(beta, X, Y) {
    p <- logistic_model(X, beta)
    return(sum(Y * log(p) + (1 - Y) * log(1 - p)))  # Log-likelihood calculation
  }
  
  # Importance sampling
  beta_samples <- mvrnorm(n_samples, mu = rep(0, 9), Sigma = proposal_cov)  # Sample from the proposal distribution
  log_weights <- numeric(n_samples)
  
  # Compute log-weights based on log-likelihood and proposal log-density
  for (i in 1:n_samples) {
    log_weights[i] <- log_likelihood(beta_samples[i, ], X, Y) - dmvnorm(beta_samples[i, ], mean = rep(0, 9), sigma = proposal_cov, log = TRUE)
  }
  
  # Normalize log-weights to prevent numerical issues
  max_log_weight <- max(log_weights)
  weights <- exp(log_weights - max_log_weight)
  weights <- weights / sum(weights)  # Normalize the weights
  
  # Posterior means
  posterior_means <- colSums(beta_samples * weights)
  
  # Effective sample size (ESS)
  ESS <- (sum(weights)^2) / sum(weights^2)
  
  # Resample to estimate posterior
  resampled_indices <- sample(1:n_samples, n_resamples, replace = TRUE, prob = weights)
  resampled_betas <- beta_samples[resampled_indices, ]
  
  # Plotting
  dev.new()  # Open a new plotting window for each sample size
  par(mfrow = c(3, 3), mar = c(4, 4, 2, 1))  # Adjust margins and layout
  hist(resampled_betas[, 1], main = paste("Posterior of β0 (N=", N, ")"), xlab = "β0", breaks = 30, col = 'skyblue', border = 'white')
  hist(resampled_betas[, 2], main = paste("Posterior of β1 (N=", N, ")"), xlab = "β1", breaks = 30, col = 'skyblue', border = 'white')
  hist(resampled_betas[, 3], main = paste("Posterior of β2 (N=", N, ")"), xlab = "β2", breaks = 30, col = 'skyblue', border = 'white')
  hist(resampled_betas[, 4], main = paste("Posterior of β3 (N=", N, ")"), xlab = "β3", breaks = 30, col = 'skyblue', border = 'white')
  hist(resampled_betas[, 5], main = paste("Posterior of β4 (N=", N, ")"), xlab = "β4", breaks = 30, col = 'skyblue', border = 'white')
  hist(resampled_betas[, 6], main = paste("Posterior of β5 (N=", N, ")"), xlab = "β5", breaks = 30, col = 'skyblue', border = 'white')
  hist(resampled_betas[, 7], main = paste("Posterior of β6 (N=", N, ")"), xlab = "β6", breaks = 30, col = 'skyblue', border = 'white')
  hist(resampled_betas[, 8], main = paste("Posterior of β7 (N=", N, ")"), xlab = "β7", breaks = 30, col = 'skyblue', border = 'white')
  hist(resampled_betas[, 9], main = paste("Posterior of β8 (N=", N, ")"), xlab = "β8", breaks = 30, col = 'skyblue', border = 'white')
  
  return(list(posterior_means = posterior_means, ESS = ESS, resampled_betas = resampled_betas))
}

# Run for each sample size
results <- list()
for (N in sample_sizes) {
  cat("Processing for N =", N, "...\n")
  result <- importance_sampling(N)
  results[[as.character(N)]] <- result
}

# Print results
for (N in sample_sizes) {
  cat("\nSample size N =", N, "\n")
  cat("Posterior means:\n")
  print(results[[as.character(N)]]$posterior_means)
  cat("Effective sample size (ESS):\n")
  print(results[[as.character(N)]]$ESS)
}

```
```{r}
#Question 4

library(MASS)  # For mvrnorm
library(stats)  # For optim()

# Define the logistic model function
logistic_model <- function(X, beta) {
  z <- X %*% beta
  return(1 / (1 + exp(-z)))
}

# Define the log-posterior function (log-likelihood + log-prior)
log_posterior <- function(beta, X, Y, prior_mean, prior_cov) {
  # Log-likelihood for logistic regression
  log_likelihood <- sum(Y * log(logistic_model(X, beta)) + (1 - Y) * log(1 - logistic_model(X, beta)))
  
  # Log-prior (assume normal prior for each beta with zero mean and unit variance)
  log_prior <- sum(dmvnorm(beta, mean = prior_mean, sigma = prior_cov, log = TRUE))  # Use dmvnorm for prior
  
  return(log_likelihood + log_prior)
}

# Smarter proposal using posterior mode
importance_sampling_smart_proposal <- function(N, proposal_cov) {
  # Simulate data with intercept and 8 covariates (from Uniform[-2, 2])
  set.seed(42)
  X <- cbind(1, matrix(runif(N * 8, min = -2, max = 2), ncol = 8))  # Add intercept as the first column
  
  # True beta values (for this example)
  beta_true <- c(0.1, 1.1, -0.9, runif(6, -1, 1))  # Example: true beta values
  
  # Compute probabilities using the logistic model
  probs <- logistic_model(X, beta_true)
  
  # Simulate the response variable Y from a binomial distribution
  Y <- rbinom(N, 1, probs)
  
  # Define the log-posterior function for optimization
  log_posterior_fn <- function(beta) {
    return(-log_posterior(beta, X, Y, prior_mean = rep(0, 9), prior_cov = diag(9)))  # Negative log-posterior for minimization
  }
  
  # Initial guess for optimization
  init_guess <- rep(0, 9)
  
  # Optimize to find the posterior mode (MAP estimate)
  mode_result <- optim(init_guess, log_posterior_fn, method = "BFGS")  # Using BFGS method
  if (mode_result$convergence != 0) {
    warning("Optimization did not converge!")
  }
  posterior_mode <- mode_result$par  # Extract the posterior mode
  
  cat("Posterior mode: ", posterior_mode, "\n")
  
  # Smarter proposal centered at posterior mode
  n_samples <- 20000
  beta_samples <- mvrnorm(n_samples, mu = posterior_mode, Sigma = proposal_cov)  # Sample from the proposal
  log_weights <- numeric(n_samples)
  
  # Compute log-weights based on the log-posterior and the proposal log-density
  for (i in 1:n_samples) {
    log_weights[i] <- log_posterior(beta_samples[i, ], X, Y, prior_mean = rep(0, 9), prior_cov = diag(9)) - 
      dmvnorm(beta_samples[i, ], mean = posterior_mode, sigma = proposal_cov, log = TRUE)  # Log of the importance weight
  }
  
  # Normalize the weights to prevent numerical instability
  max_log_weight <- max(log_weights)
  weights <- exp(log_weights - max_log_weight)
  weights <- weights / sum(weights)  # Normalize the weights
  
  # Compute posterior mean from the weighted samples
  posterior_means <- colSums(beta_samples * weights)
  
  # Compute the effective sample size (ESS)
  ESS <- (sum(weights)^2) / sum(weights^2)
  
  return(list(posterior_means = posterior_means, ESS = ESS, resampled_betas = beta_samples))
}

# Define sample sizes for the experiment
sample_sizes <- c(10, 50, 100)

# Run the experiment for each sample size
results_smart_proposal <- list()
for (N in sample_sizes) {
  cat("Processing for N =", N, "...\n")
  result <- importance_sampling_smart_proposal(N, proposal_cov = diag(9))  # Use identity matrix as the covariance
  results_smart_proposal[[as.character(N)]] <- result
}

# Print the results for each sample size
for (N in sample_sizes) {
  cat("\nSample size N =", N, "\n")
  cat("Posterior means (smart proposal):\n")
  print(results_smart_proposal[[as.character(N)]]$posterior_means)
  cat("Effective sample size (ESS) (smart proposal):\n")
  print(results_smart_proposal[[as.character(N)]]$ESS)
}


```

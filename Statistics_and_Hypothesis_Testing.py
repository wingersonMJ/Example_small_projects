import numpy as np
from scipy import stats


def sampling(low, high, n):
    rng = np.random.default_rng(1989)
    return rng.uniform(low, high, n)


# simple weighted mean
sample = sampling(0, 50, 100)
weights = sampling(0, 1, 100)

mean = np.mean(sample)
wt_mean = sum(sample * weights) / sum(weights)
print("Mean:", mean)
print("Weighted Mean:", wt_mean)

# simple variance and standard deviation
variance = sum((s - mean) ** 2 for s in sample) / (len(sample) - 1)
std_dev = np.sqrt(variance)

print("Variance:", variance)
print("Standard Deviation:", std_dev)

# outlier detection
upper = mean + 1.75 * std_dev
lower = mean - 1.75 * std_dev

outliers = [s for s in sample if s < lower or s > upper]
print("Number of outliers:", len(outliers))
print("Outliers:", outliers)

# z-score
z_scores = [(s - mean) / std_dev for s in sample]

print(z_scores[:2])

# EXAMPLE Problem #1:
x_30 = stats.norm.cdf(50, mean, std_dev) - stats.norm.cdf(30, mean, std_dev)
x_40 = stats.norm.cdf(50, mean, std_dev) - stats.norm.cdf(40, mean, std_dev)

print("Probability of x between 30 and 50:", x_30)
print("Probability of x between 40 and 50:", x_40)

# confidence intervals
lower_95CI = mean - (2 * std_dev)
upper_95CI = mean + (2 * std_dev)
print("95% Confidence Interval:", (lower_95CI, upper_95CI))
print(
    stats.norm.cdf(upper_95CI, mean, std_dev)
    - stats.norm.cdf(lower_95CI, mean, std_dev)
)

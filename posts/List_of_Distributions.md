---
title: "Common Distributions"
date: "2022-03-09" 
---

In this post, I will write about some of the key distributions that are used by statisticians. The aim is to not just have the formulas, but explain intuition, proofs and uses for these distributions. These will be added in as my time permits. I will mainly be using this page as a reference.

Each distribution will have its probability mass function (pmf) or probability density function (pdf) listed, mean and variance, and some notes on relations to other distributions and derivations. 

# Notation and Abbreviations
- $X$: generic random variable.
- Cumulative distribution function: The function $F_X(x) := P(X \leq x)$.
- Probability mass function (PMF): For a discrete random variable $X$, the function $f_X(x) = P(X = x)$. 
- Probability distribution function (PDF): For a continuous random variable $X$, the derivative $\frac{d}{dx} F_X(x)$ of the CDF $F_X$.  
- Moment generating function (MGF): The function given by $m_X(t) = \mathbb{E}\exp(tX)$.

## Discrete Probability Distributions 

# Bernoulli, Binomial and Multinomial Distributions
Bernoulli$$(p)$$: The Bernoulli random variable takes values $$1$$ with probability $$p$$ and $$0$$ with probability $$1-p$$ for $$0 \leq p \leq 1$$. The probability of getting $$x$$ for $$x = 0$$ or $$x = 1$$ is given by 

$$
P(X=x|p) = p^x (1-p)^{1-x}.
$$

- Mean: $$\mathbb{E}X = p$$. 
- Variance: $$\text{Var} X = p(1-p)$$.
- MGF: $$M_X(t) = (1-p) + p\exp(t)$$.

<br>

Binomial$$(n,p)$$: The binomial random variable is the sum of $$n$$ Bernoulli random variables. By the Binomial theorem, we have 

$$
P(X=x|n,p) = {n \choose x} p^x (1-p)^{1-x} \quad \text{for } x = 0, 1, \dots, n \text{ and } 0 \leq p \leq 1.
$$

- Mean: $$\mathbb{E}X = np$$. 
- Variance: $$\text{Var}(X) = np(1-p)$$.
- MGF: $$m_X(t) = (p\exp(t) + (1-p))^n$$.

<br> 

Multinomial$$(n, p_1, p_2, \dots, p_k)$$: The multinomial distribution is a generalisation of the binomial random variable. Suppose we have $$k$$ distinct events, each occurring with probability $$p_i$$, and we have $$n$$ total experiments. The multinomial coefficient is defined by 

$$ 
{n \choose k_1, k_2, \dots, k_n} = \frac{n!}{k_1! k_2! \dots k_n!} \quad \text{with } \sum_{i=1}^n k_i = n.
$$

Write $$X_i$$ for the random variable representing the number of times event $$i$$ occurs in $$n$$ experiments and write $$X = (X_1, X_2, \dots, X_n)$$. Then the PMF of $$X$$ is given by

$$ 
P(X=(x_1, x_2,\dots, x_k)|k,n,p) = {n \choose x_1, x_2, \dots, x_k} p_1^{x_1} p_2^{x_2} \dots p_k^{x_k}.
$$

We must have $\sum_{i=1}^k x_k = n$ and $\sum_{i=1}^k p_k = 1$.

- Mean: $$\mathbb{E}X_i = np_i$$. 
- Variance: $$\text{Var}(X) = np_i(1-p_i)$$.
- MGF: $$m_X(t) = (\sum_{i=1}^k p_i \exp(t_i))^n$$. 

# Geometric Distribution

Suppose we are doing an experiment to see the number of binomial trials required to see one success, with $$p$$ being the probability of success. The probability that the first success occurs on experiment $$x$$ is given by the probability of failing $$x-1$$ times and succeeding on the $$x$$th experiment:

$$
P(X=x | p) = p(1-p)^{x-1}, \quad x = 1, 2, \dots; \quad 0 \leq p \leq 1.
$$

- Mean: $$\mathbb{E}X = 1/p$$. 
- Variance: $$\text{Var}(X) = \frac{1-p}{p^2}.$$
- MGF: $$m_X(t) = \frac{p\exp(t)}{1-(1-p)\exp(t)}, \quad t < -\log(1-p)$$.

<br>

The geometric distribution is memoryless, in the sense that $$P(X > s | X > t) = P(X > s-t)$$: given that we know the success must occur after $$t$$ experiments, the probability of $$X$$ occuring after $$s$$ experiments (where $$s > t$$) is exactly as if we had started the experiments from 0: that is, it is equal to the probability that $$X > s - t$$.

# Hypergeometric Distribution

Suppose there are a total of $$M$$ tagged fish in a total population of $$N$$ fishes. Suppose further we have trawled up $$K$$ fishes (with $$K \leq N$$). Then the probability of getting exactly $$x$$ tagged fish is given by 

$$
P(X = x | N, M, K) = \frac{ \binom{M}{x}\binom{N-M}{K - x}}{\binom{N}{K}}
$$

This represents the probability of getting $$x$$ of the fish from the tagged population $$M$$, the remaining $$K - x$$ fish from the non-tagged $$N-M$$ population, divided by the total number of ways of getting $$K$$ fish from $$N$$ total fishes.

- Mean: $$\mathbb{E}X = KM/N$$. 
- Variance: $$\text{Var}(X) = \frac{KM}{N} \frac{(N-M)(N-K)}{N(N-1)}.$$

# Poisson Distribution

Suppose some set of discrete events occur at an average fixed rate $$\lambda$$, with no two events occurring at the same instantaneous moment and each event occurring independently of the other. Then the number of such events $$X$$ which occur over a fixed period of time follows a Poisson distribution (written $X \sim \text{Pois}(\lambda)$), with

$$
P(X= x | \lambda) = \frac{\exp(-\lambda) \lambda^x}{x!}, \quad x = 0,1 , \dots; \quad 0 \leq \lambda < \infty.
$$

- Mean: $$\mathbb{E}X = \lambda$$. 
- Variance: $$\text{Var}(X) = \lambda$$.
- MGF: $$m_X(t) = \exp(\lambda(\exp^t-1))$$.

<!-- ## Continuous Probability Distributions

# Normal Distribution

# Beta Distribution

# Laplace Distribution

# Exponential Distribution


# Gamma Distribution
 -->


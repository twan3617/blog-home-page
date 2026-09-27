---
title:  "Bayesian Inference, and a basic Changepoint Detection Algorithm"
date:   '2024-01-09'
description: "An introduction to Bayesian reasoning through a practical changepoint-detection algorithm."
topics: [Bayesian Statistics, Probability, Algorithms]
featured: true
---


## Background
Any Bayesian analysis of data must start with Bayes' rule: 

$$
P(A|B) = \frac{P(B|A) P(A)}{P(B)},
$$

where $$A$$ and $$B$$ are two measurable events with $$P(B) \neq 0$$ . In fact, it has been proven that Bayes' rule continues to hold when $$A$$ and $$B$$ are instead replaced with random variables and their corresponding distributions. Let $$\theta$$ be the parameter of interest (conceptualised as a random variable under the Bayesian framework\*) and $$X$$ be the observed data. Bayes' rule for distributions can be written as 

$$
\pi(\theta | X) = \frac{L(X | \theta) \pi(\theta)}{m(X)},
$$

where $\pi(\theta | X)$ is the _posterior distribution_ of $$\theta$$ after observing the data $$X$$, $$L(X | \theta)$$ is called the _likelihood function_ which is (proportional to) the probability of observing the data $$X$$ given $$\theta$$, $$\pi(\theta)$$ is the _prior_ which describes our initial belief in the distribution of the parameter $$\theta$$, and $$m(X)$$ is a constant which normalises the right-hand expression into a probability distribution that integrates to $$1$$.

<br> 

We can interpret the above equation as follows: after having observed the data $$X$$, we are taking our prior belief of what $$\theta$$ could be and updating it to more accurately fit the observations. This update is encoded in the posterior $$\pi(\theta | X)$$ using Bayes' rule and can only be calculated once we can evaluate $$\pi(\theta)$$ (i.e. specifiying the prior distribution) and $$L(X|\theta)$$ (i.e. specifying the likelihood).

<br> 

Note that the normalisation constant $$m(X)$$ is usually not of theoretical interest, since the distributional information on $$X$$ and $$\theta$$ is entirely encapsulated within the prior and likelihood terms and the integration required to find $${m(X) = \int_\Theta L(X|\theta)\pi(\theta) \: d\theta}$$ is usually intractable by hand anyway. Hence, you might see Bayes' rule being written and applied in proportional form:

$$
\pi(\theta | X) \propto L(X | \theta) \pi(\theta) = \text{Likelihood} \times \text{Prior},
$$

and many Bayesian calculations done by hand only do so up to some fixed multiplicative constant, with the final distribution determined by the functional form of the posterior, as we will discuss in the next section.

<br> 

\*: _I've completely skipped a very important point here, a point which is entirely responsible for the difference between the Frequentist and Bayesian statistical treatise. Fitting statistical and probabilistic models onto data requires finding parameters of interest which tell us something about the underlying data generating process. From the Frequentist perspective, these parameters are unknown but fixed, and the goal is usually to optimise some probability function to find these parameters. The Bayesian framework posits that these parameters are not only unknown, but are inherently random in itself. In the absense of data, the probability distribution of $$\theta$$ represents our underlying belief of what values $$\theta$$ can take, which is then updated to take into account observed data $$X$$._

## Obtaining the Posterior Distribution

To do any sort of inference in the Bayesian framework, we need to compute the posterior $$\pi(\theta | X)$$ (usually only up to some normalisation constant). This requires us to: 

- Specify the _prior distribution_ of $$\theta$$, $$\pi(\theta)$$.
- Specify the _distributional form_ relating $$\theta$$ to the data generation process behind $$X$$, and hence the likelihood $$L(X | \theta)$$.

There is a lot of nuance in each of these steps which we should elaborate on.

### Picking a Prior
This is the modelling step with the most flexibility, and represents a way for "external information" to make its way into the model. Essentially, here, we need to give our view on what values we _think_ $$\theta$$ could take on. Depending on the strength of evidence that one chooses to encode in the prior, it could either heavily impact the inferential step, or not. At a minimum, the prior should be supported on values that make sense: for example, a prior for the probability $$p$$ of flipping heads on a coin should be in the interval $$[0,1]$$, and nowhere else; a prior on the number of patients at a hospital should, likewise, only have positive probability on the natural numbers. 

<br>

However, how _certain_ we are about parameter $$\theta$$, and what value $$\theta$$ would most likely take, is entirely up to us. A common approach when there is no strong opinion or past experience to incorporate is to take an "objective" prior which has high variance and (ideally) doesn't weight any particular value higher than any other - for example, choosing a uniform $$U[0,1]$$ distribution for a probability $$p$$, or a normal $$N(0, 100)$$ prior for a regression parameter.

<br> 

One might consider whether the subjectivity of choosing a prior means that Bayesian analyses are, in some sense, not rigorous. Certainly, this type of choice is not (explicitly) present in Frequentist analyses! One argument against this is as follows: In the "big data limit" as the number of data points $$n \to \infty$$, the effect of the prior on the posterior distribution vanishes. Intuitively, in the formula given by Bayes' rule, the prior $$\pi(\theta)$$ remains constant in $$n$$ while the likelihood term $$L(X|\theta)$$ scales multiplicatively with the number of data points observed (assuming independence) and hence dominates in the big data limit. Explicit examples can also be worked out to concretely demonstrate this fact (a common exercise is to show the equivalence of Frequentist and Bayesian ordinary least squares regression, one which I may write out in a future post). 

<br> 

Hence, we can be confident that despite the inherently subjective nature of picking a prior in the Bayesian world, both frameworks ultimately converge to a consistent underlying truth, with the added flexibility of being able to encode prior, expert knowledge into the model with the Bayesian framework.


### Specifying the Distributional Form
Most of the time, the distributional form relating $$\theta$$ and $$X$$ can be inferred by the type of data that is being modelled. For example, data that comes through as counts could be described by a Poisson distribution; waiting times between events would naturally follow an Exponential distribution, and heights could (roughly) be approximated by a Normal distribution. 

<br> 

That being said, there is no requirement that the likelihood $$L(X | \theta)$$ has to be of a known distributional form - as long as the likelihood function can be evaluated given $$X$$ and $$\theta$$, there are still methods to obtaining the posterior distribution; it simply means that a calculation by hand won't be possible, which is perfectly fine with 21st century computational power. Indeed, I describe some methods to obtain the posterior distribution in the sections below. This means that non-parametric methods of modelling the likelihood $$L(X|\theta)$$ is possible, albeit harder to analyse theoretically.

### Computations by Hand, or by Computer

Ultimately, we must be able to obtain the posterior distribution $$\pi(\theta | X)$$ in order to do Bayesian inference. As with all mathematical methods developed before modern computing, there are ways to compute the posterior distribution by hand. This is done via a clever choice of a prior, called the conjugate prior, which is essentially a choice of distribution whose probability density has the same core form as the likelihood function. Conjugate priors exist for exponential family distributions, which encompass a large number of commonly used distributions in statistical modelling. Since priors can be designed to contain as little information as possible (e.g. by increasing variance), this should not theoretically impact any posterior inference. 

<br> 

New algorithms focused on running simulations to obtain _samples_ from the posterior distribution, such as Gibbs sampling, the Metropolis-Hastings algorithm and Hamiltonian Monte Carlo also work well and open up opportunities for many statistical problems to be tractable in the Bayesian framework, albeit requiring more computational power and hence also restricting its potential industrial applications. Below, I will describe an example of a conjugate prior computation using a simple probability modelling example.

#### Computations by Hand with Conjugate Priors

Suppose we have a coin that can either be heads or tails, and we wish to find out the probability $$p$$ that it will turn up heads. To do so, we conduct an experiment to flip the coin $$10$$ times and observe $$6$$ heads and $$4$$ tails. The _likelihood_ of observing this result, given that we know $$p$$, is given by the binomial formula: 

$$
P(6H \text{ and } 4T | p) =  {10 \choose 6} p^6 (1-p)^{4}.
$$

Using Bayes' rule, we can obtain an expression for the posterior distribution of $$p$$ (up to normalisation constant):

$$
\begin{aligned}
\pi(p \mid X) &\propto P(X \mid p)\pi(p) \\
&= {10 \choose 6} p^6 (1-p)^4\pi(p) \\
&\propto p^6 (1-p)^4\pi(p).
\end{aligned}
$$

By choosing the prior $$\pi(p)$$ to be of the same _form_ as the binomial, we can ensure that the posterior will be a known distribution that can be worked out by hand. As it turns out, the conjugate distribution for the binomial is the beta distribution, which has form 

$$
f(x; \alpha, \beta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha) \Gamma(\beta)} x^{\alpha-1} (1-x)^{\beta -1},
$$

for shape parameters $$\alpha, \beta > 0$$, $$\Gamma$$ the gamma function and $$x \in [0,1]$$. It is also a known fact that the expectation and variance of a beta-distributed variable $$V$$ is given by 
$$
\begin{aligned}
E[V] &= \frac{\alpha}{\alpha + \beta}, \\
\operatorname{Var}(V) &= \frac{\alpha \beta}{(\alpha + \beta)^2 (\alpha + \beta + 1)}.
\end{aligned}
$$ 

Hence, one possible choice of a prior here is to say that we are reasonably sure that the probability of heads $$p$$ on the coin is somewhere around $$0.5$$, and we can pick $$\alpha$$ and $$\beta$$ so that $$E[p] = 0.5$$ with the appropriate variance to represent our uncertainty. Let's choose 

$$
p \sim \text{Beta}(2,2),
$$

so that our posterior $$\pi(p|X)$$ is now 

$$
\pi(p \mid X) \propto p^6 (1-p)^4 \times p(1-p) = p^7(1-p)^5.
$$

The exponents identify the posterior as $$p \mid X \sim \text{Beta}(8,6)$$. We have omitted normalising constants that do not depend on $$p$$. I've plotted the density below using R, with a red line at the expected value of $$p$$ (note: the expected value is _not_ the mode!)

<figure>
  <img src="/images/beta_density.png" alt="Beta distribution density after observing six heads, with its mean marked in red">
  <figcaption>The Beta(8, 6) posterior density; the red line marks its mean.</figcaption>
</figure>

We can see that visually what the effect of updating the prior with the observed data does: $$6$$ heads in $$10$$ coin tosses is most likely to occur when the coin has a $$60\%$$ chance of heads, but this effect is tempered by our prior belief that the probability is closer to $$50\%$$.

<br>

In the next section, I discuss how we can draw statistical inferences from the computations we have done, whether we have used a conjugate prior or obtained samples via computational simulations. 

## Inference

Inference in the Bayesian framework is essentially encapsulated in obtaining the posterior distribution of the parameter $$\theta$$ of interest. Assuming we have computed the posterior distribution $$\pi(\theta | X)$$...

<br>

**Want to get a point estimate of $$\theta$$?** The posterior mean minimises expected squared error, the posterior median minimises expected absolute error, and the posterior mode is the maximum a posteriori (MAP) estimate. These losses are averaged over the posterior distribution $$\pi(\theta \mid X)$$.

<br> 

**Want to understand the uncertainty in $$\theta$$?** Compute a $$\rho \%$$ _credible interval_, the Bayesian equivalent of the Frequentist confidence intervals. No more finnicky interpretations of what a confidence interval means - since $$\theta$$ is a random variable, a 95% credible interval contains 95% of the posterior probability. Easy!

<br> 

**A/B testing: Want to decide whether Website A performed better than Website B in generating sales in your most recent pricing experiment?** Decide on prior distributions for $$p_A$$ and $$p_B$$, the probabilities that customers will convert on Website A and Website B respectively, then compute the posterior distributions $$\pi(p_A | X_A)$$ and $$\pi(p_B | X_B)$$ using Bayes' rule; look at the posterior distribution $$\pi(p_A - p_B | X_A, X_B)$$ to understand the direction and magnitude of the difference between Website $$A$$ and Website $$B$$, and the credible intervals to quantify the uncertainty. This is a simple Bayesian A/B testing framework which has very low computational requirements (with appropriate choices of conjugate prior), and can avoid a lot of the pitfalls of Frequentist A/B testing (p-values and early stopping rules, I'm looking at you!)

<br>

As you can see, inferential statements on parameters convert to natural statements on probability distributions that we are used to making. This flexibility can be very useful when attempting to create a model of a more complicated data generating process, as we will see in the next section when creating a changepoint detection algorithm.


## A Basic Changepoint Detection Algorithm

To put all of the theory given above into practice, let's define a basic offline Bayesian changepoint detection algorithm to a real-world problem. We are given a dataset consisting of the number of mining disasters that occurred in Great Britain between 1851 and 1962, and the problem is to try and identify whether any major changes have occurred in the ongoing rate of accidents per year. If there is strong evidence that a particular time period separates periods of high and low rate of mining accidents, then one can dig further into that time period: was there a piece of legislation enacted, or safety equipment introduced, that significantly improved mining safety?

<br> 

To begin, we model the number of mining accidents per year $$X_t$$ ($${1851 \leq t \leq 1962}$$) as a Poisson random variable with rate $$\lambda_t$$. 

<br>

At some year $$\tau$$ between 1851 and 1962, the rate $$\lambda$$ changes. Since we have no clear opinion of when this change may have occurred, we give $$\tau$$ a discrete uniform prior on those years and write $$\lambda$$ as

$$
\lambda_t = \begin{cases}
\lambda_1 \quad \text{for } 1851 \leq t < \tau, \\
\lambda_2 \quad \text{for } \tau \leq t \leq 1962.
\end{cases}
$$

What priors should we choose for $$\lambda_1$$ and $$\lambda_2$$? Both are positive rates, so an exponential prior is convenient. PyMC parameterises its exponential distribution by a rate $$\alpha$$, which we set to the inverse of the observed mean annual count:

$$
\alpha = \left( \frac{1}{112} \sum_{t=1851}^{1962} X_t \right)^{-1}.
$$

This uses the data to set the prior as well as to fit the model, so the example should be read as illustrative. A prior chosen independently of these observations would avoid that double use of the data.

<br> 

At this point, with our likelihood and priors set in place, we are ready to define a model, compute the posterior and do some inference. I choose to use the probabilistic programming package PyMC to do the heavy lifting for me.

### Code
```python

# Imports 
import arviz as az
import pandas as pd
import numpy as np
import pymc as pm

# Replace with your own data source
accidents_by_year = pd.read_csv("./mining_accidents.csv")["x"] 
accidents_by_year.index = accidents_by_year.index + 1851
first_year = accidents_by_year.index.min()
last_year = accidents_by_year.index.max() 

with pm.Model() as model:

    # Define rate parameter for lambda priors
    alpha = 1 / np.mean(accidents_by_year)

    lambda_1 = pm.Exponential('lambda_1', lam=alpha)
    lambda_2 = pm.Exponential('lambda_2', lam=alpha) 

    # tau is the first year with rate lambda_2
    tau = pm.DiscreteUniform('tau', lower=first_year, upper=last_year)
    

    # Lambdas are the two rates. 
    # The switch function requires an index passed in
    idx = np.arange(first_year, last_year+1)
    lambda_total = pm.math.switch(tau > idx, lambda_1, lambda_2) 

    # Define observations as Poissons and pass in observation data. 
    # The provided rate argument mu is a random variable here!
    obs = pm.Poisson('obs', mu=lambda_total, observed=accidents_by_year)


    # Define simulation algorithm 
    step = pm.Metropolis()
    posterior = pm.sample(step=step, draws=10000, tune=1000, chains=4)
```

<br>

After running this, we can plot and summarise the simulations using the Arviz package. We first look at the diagnostics charts for each chain generated by the Metropolis-Hastings algorithm:

<figure>
  <img src="/images/diagnostic_charts.png" alt="Posterior histograms and MCMC trace plots for the changepoint and two accident rates">
  <figcaption>Posterior distributions and MCMC traces for the changepoint and two accident rates.</figcaption>
</figure>

The graphs on the left display the histograms of the simulated variables $$\tau$$, $$\lambda_1$$ and $$\lambda_2$$. The graphs on the right show the sequence of samples drawn in the simulation. The traces show no obvious drift, although visual inspection alone cannot establish good mixing. There appears to be strong evidence of a changepoint around 1890, with a second mode around 1887–1888.

<figure>
  <img src="/images/MCMC%20diagnostics.png" alt="Posterior summary table for the changepoint year and accident rates">
  <figcaption>Posterior summaries with 94% highest density intervals.</figcaption>
</figure>

We see these observations come through in the diagnostics table. Our model predicts a high probability of a changepoint in the number of yearly mining incidents occurring in the year 1890, with a significantly different rate of incidents before and after this change point with an average of $$3.11$$ incidents occurring before the changepoint and $$0.90$$ incidents occurring afterwards. 

<br>

The table's 3% and 97% HDI columns are the lower and upper bounds of a 94% highest density interval, showing uncertainty around each estimate.

<br>

Just to connect this back to reality, there was indeed a Royal Commision on Mining Royalties (1890-1891), with the Coal Mines Regulation Act (1887), Truck Amendment Act (1887) and Factory and Workshop Act (1891) occurring around this time period. That being said, coal mining and their disasters in Britain have had a long history, with many different acts, restrictions and laws put in place (see \[3\] for details), and I'm not entirely confident that the Bayesian methods used here have completely captured the nuances of that history. But it is a start, and a very interesting application of a simple changepoint algorithm.  

### Extensions
In writing out the above model, it became clear to me that there are some obvious ways of extending the model to be more flexible. Indeed, We assumed that there is only one changepoint, but we could just as easily define multiple $$\lambda$$'s and multiple changepoints. The number of changepoints doesn't even need to be fixed: if $$n$$ is the number of changepoints, we could put a uniform prior on $$n$$ and simulate to find the number of changepoints that match the data the best. 







## References
\[1\] [Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers) - this is a great resource on getting started with implementing Bayesian models in Python with PyMC. The text message example in Chapter 1 is almost an exact replica of the example I give above, with a few differences in the chosen distributions. 

<br>

\[2\] [Dataset1](https://www.cmhrc.co.uk/site/disasters/disasters_list_1850.html) and [Dataset2](https://www.cmhrc.co.uk/site/disasters/disasters_list_1900.html) - the dataset was provided as part of a class on Bayesian Inference (MATH5960, UNSW T3 2021), but some digging gave me a concrete listing of coal mine disasters in the provided years, with the counts summarised into the dataset provided in the code above. 

\[3\] [Government and mining](https://mininginstitute.org.uk/wp-content/uploads/2016/02/Government-and-mining-Jan16.pdf) - a history of coal mining disasters, royal commissions and acts enforced on the coal mining industry.

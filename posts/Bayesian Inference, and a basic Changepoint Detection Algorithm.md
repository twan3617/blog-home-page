---
title:  "Bayesian Inference, and a basic Changepoint Detection Algorithm"
date:   '2024-01-09'
---


# Background
Any Bayesian analysis of data must start with Bayes' rule: 

$$
P(A|B) = \frac{P(B|A) P(A)}{P(B)},
$$

where $$A$$ and $$B$$ are two measurable events with $$P(B) \neq 0$$ . In fact, it has been proven that Bayes' rule continues to hold when $$A$$ and $$B$$ are instead replaced with random variables and their corresponding distributions. Let $$\theta$$ be the parameter of interest (conceptualised as a random variable under the Bayesian framework\*) and $$X$$ be the observed data. Bayes' rule for distributions can be written as 

$$
\pi(\theta | X) = \frac{L(X | \theta) \pi(\theta)}{m(x)},
$$

where $\pi(\theta | X)$ is the _posterior distribution_ of $$\theta$$ after observing the data $$X$$, $$L(\theta | X)$$ is called the _likelihood function_ which is (proportional to) the probability of observing the data $$X$$ given $$\theta$$, $$\pi(\theta)$$ is the _prior_ which describes our initial belief in the distribution of the parameter $$\theta$$, and $$m(x)$$ is a constant which normalises the right-hand expression into a probability distribution that integrates to $$1$$. 

<br> 

We can interpret the above equation as follows: after having observed the data $$X$$, we are taking our prior belief of what $$\theta$$ could be and updating it to more accurately fit the observations. This update is encoded in the posterior $$\pi(\theta | X)$$ using Bayes' rule and can only be calculated once we can evaluate $$\pi(\theta)$$ (i.e. specifiying the prior distribution) and $$L(X|\theta)$$ (i.e. specifying the likelihood).

<br> 

Note that the normalisation constant $$m(x)$$ is usually not of theoretical interest, since the distributional information on $$X$$ and $$\theta$$ is entirely encapsulated within the prior and likelihood terms and the integration required to find $${m(x) = \int_\Theta L(X|\theta)\pi(\theta) \: d\theta}$$ is usually intractable by hand anyway. Hence, you might see Bayes' rule being written and applied in proportional form: 

$$
\pi(\theta | X) \propto L(X | \theta) \pi(\theta) = \text{Likelihood} \times \text{Prior},
$$

and many Bayesian calculations done by hand only do so up to some fixed multiplicative constant, with the final distribution determined by the functional form of the posterior, as we will discuss in the next section.

<br> 

\*: _I've completely skipped a very important point here, a point which is entirely responsible for the difference between the Frequentist and Bayesian statistical treatise. Fitting statistical and probabilistic models onto data requires finding parameters of interest which tell us something about the underlying data generating process. From the Frequentist perspective, these parameters are unknown but fixed, and the goal is usually to optimise some probability function to find these parameters. The Bayesian framework posits that these parameters are not only unknown, but are inherently random in itself. In the absense of data, the probability distribution of $$\theta$$ represents our underlying belief of what values $$\theta$$ can take, which is then updated to take into account observed data $$X$$._

# Obtaining the Posterior Distribution 

To do any sort of inference in the Bayesian framework, we need to compute the posterior $$\pi(\theta | X)$$ (usually only up to some normalisation constant). This requires us to: 

<br> 

** Specify the _prior distribution_ of $$\theta$$, $$\pi(\theta)$$.

** Specify the _distributional form_ relating $$\theta$$ to the data generation process behind $$X$$, and hence the likelihood $$L(X | \theta)$$.

<br> 

There is a lot of nuance in each of these steps which we should elaborate on.

## Picking a Prior
This is the modelling step with the most flexibility, and represents a way for "external information" to make its way into the model. Essentially, here, we need to give our view on what values we _think_ $$\theta$$ could take on. Depending on the strength of evidence that one chooses to encode in the prior, it could either heavily impact the inferential step, or not. At a minimum, the prior should be supported on values that make sense: for example, a prior for the probability $$p$$ of flipping heads on a coin should be in the interval $$[0,1]$$, and nowhere else; a prior on the number of patients at a hospital should, likewise, only have positive probability on the natural numbers. 

<br>

However, how _certain_ we are about parameter $$\theta$$, and what value $$\theta$$ would most likely take, is entirely up to us. A common approach when there is no strong opinion or past experience to incorporate is to take an "objective" prior which has high variance and (ideally) doesn't weight any particular value higher than any other - for example, choosing a uniform $$U[0,1]$$ distribution for a probability $$p$$, or a normal $$N(0, 100)$$ prior for a regression parameter.

<br> 

One might consider whether the subjectivity of choosing a prior means that Bayesian analyses are, in some sense, not rigorous. Certainly, this type of choice is not (explicitly) present in Frequentist analyses! One argument against this is as follows: In the "big data limit" as the number of data points $$n \to \infty$$, the effect of the prior on the posterior distribution vanishes. Intuitively, in the formula given by Bayes' rule, the prior $$\pi(\theta)$$ remains constant in $$n$$ while the likelihood term $$L(X|\theta)$$ scales multiplicatively with the number of data points observed (assuming independence) and hence dominates in the big data limit. Explicit examples can also be worked out to concretely demonstrate this fact (a common exercise is to show the equivalence of Frequentist and Bayesian ordinary least squares regression, one which I may write out in a future post). 

<br> 

Hence, we can be confident that despite the inherently subjective nature of picking a prior in the Bayesian world, both frameworks ultimately converge to a consistent underlying truth, with the added flexibility of being able to encode prior, expert knowledge into the model with the Bayesian framework.


## Specifying the Distributional Form 
Most of the time, the distributional form relating $$\theta$$ and $$X$$ can be inferred by the type of data that is being modelled. For example, data that comes through as counts could be described by a Poisson distribution; waiting times between events would naturally follow an Exponential distribution, and heights could (roughly) be approximated by a Normal distribution. 

<br> 

That being said, there is no requirement that the likelihood $$L(X | \theta)$$ has to be of a known distributional form - as long as the likelihood function can be evaluated given $$X$$ and $$\theta$$, there are still methods to obtaining the posterior distribution; it simply means that a calculation by hand won't be possible, which is perfectly fine with 21st century computational power. Indeed, I describe some methods to obtain the posterior distribution in the sections below. This means that non-parametric methods of modelling the likelihood $$L(X|\theta)$$ is possible, albeit harder to analyse theoretically.

## Computations by Hand, or by Computer

Ultimately, we must be able to obtain the posterior distribution $$\pi(\theta | X)$$ in order to do Bayesian inference. As with all mathematical methods developed before modern computing, there are ways to compute the posterior distribution by hand. This is done via a clever choice of a prior, called the conjugate prior, which is essentially a choice of distribution whose probability density has the same core form as the likelihood function. Conjugate priors exist for exponential family distributions, which encompass a large number of commonly used distributions in statistical modelling. Since priors can be designed to contain as little information as possible (e.g. by increasing variance), this should not theoretically impact any posterior inference. 

<br> 

New algorithms focused on running simulations to obtain _samples_ from the posterior distribution, such as Gibbs sampling, the Metropolis-Hastings algorithm and Hamiltonian Monte Carlo also work well and open up opportunities for many statistical problems to be tractable in the Bayesian framework, albeit requiring more computational power and hence also restricting its potential industrial applications. Below, I will describe an example of a conjugate prior computation using a simple probability modelling example.

### Computations by Hand with Conjugate Priors

Suppose we have a coin that can either be heads or tails, and we wish to find out the probability $$p$$ that it will turn up heads. To do so, we conduct an experiment to flip the coin $$10$$ times and observe $$6$$ heads and $$4$$ tails. The _likelihood_ of observing this result, given that we know $$p$$, is given by the binomial formula: 

$$
P(6H \text{ and } 4T | p) =  {10 \choose 6} p^6 (1-p)^{4}.
$$

Using Bayes' rule, we can obtain an expression for the posterior distribution of $$p$$ (up to normalisation constant):

$$
\pi(p | X) \propto \pi(X | p) \pi(p) \\
= {10 \choose 6} p^6 (1-p)^{4}\pi(p) \\
\propto p^6 (1-p)^{4}\pi(p).
$$

By choosing the prior $$\pi(p)$$ to be of the same _form_ as the binomial, we can ensure that the posterior will be a known distribution that can be worked out by hand. As it turns out, the conjugate distribution for the binomial is the beta distribution, which has form 

$$
f(x; \alpha, \beta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha) \Gamma(\beta)} x^{\alpha-1} (1-x)^{\beta -1},
$$

for shape parameters $$\alpha, \beta > 0$$, $$\Gamma$$ the gamma function and $$x \in [0,1]$$. It is also a known fact that the expectation and variance of a beta-distributed variable $$V$$ is given by 
$$
E[V] = \frac{\alpha}{\alpha + \beta}, \\
\text{Var}[V] = \frac{\alpha \beta}{(\alpha + \beta)^2 (\alpha + \beta + 1)}
$$ 

Hence, one possible choice of a prior here is to say that we are reasonably sure that the probability of heads $$p$$ on the coin is somewhere around $$0.5$$, and we can pick $$\alpha$$ and $$\beta$$ so that $$E[p] = 0.5$$ with the appropriate variance to represent our uncertainty. Let's choose 

$$
p \sim \text{Beta}(2,2),
$$

so that our posterior $$\pi(p|X)$$ is now 

$$
\pi(p | X) \propto p^6 (1-p)^{4} \times p^{1}(1-p)^{1} 
= p^7 (1-p)^{5} 
\sim \text{Beta}(8, 6),
$$

where we have removed all the terms that do not depend on $$p$$ or the observed data $$X$$ (for example, the Gamma coefficients from the Beta distribution). Our posterior distribution comes out to be a $$\text{Beta}(8, 6)$$ distribution. I've plotted the density below using R, with a red line at the expected value of $$p$$ (note: the expected value is _not_ the mode!) 

<br> 

![diagnostics charts](/images/beta_density.png)

<br> 

We can see that visually what the effect of updating the prior with the observed data does: $$6$$ heads in $$10$$ coin tosses is most likely to occur when the coin has a $$60\%$$ chance of heads, but this effect is tempered by our prior belief that the probability is closer to $$50\%$$.

<br>

In the next section, I discuss how we can draw statistical inferences from the computations we have done, whether we have used a conjugate prior or obtained samples via computational simulations. 

# Inference 

Inference in the Bayesian framework is essentially encapsulated in obtaining the posterior distribution of the parameter $$\theta$$ of interest. Assuming we have computed the posterior distribution $$\pi(\theta | X)$$...

<br>

** **Want to get a point estimate of $$\theta$$?** Compute a summary statistic of the posterior $$\pi(\theta | X)$$. The posterior mean ($$\mathbb{E}[\theta|X]$$) and mode (AKA maximum a posteriori estimate or MAP) of the posterior are common statistics to use here, and have nice interpretations in the decision theory framework. For example, the posterior mean minimises the $L^2$-decision error $${\int_\Theta |\theta - \theta_{\text{est}}|^2 \: d\theta}$$ (AKA mean squared error), and the mode minimises the $L^1$-decision error $${\int_\Theta |\theta - \theta_{\text{est}}| \: d \theta}$$.

<br> 

** **Want to understand the uncertainty in $$\theta$$?** Compute a $$\rho \%$$ _credible interval_, the Bayesian equivalent of the Frequentist confidence intervals. No more finnicky interpretations of what a confidence interval means - since $$\theta$$ is a random variable, the $$95%$$ credible interval is any interval which contains $$95%$$ of the posterior density. Easy!

<br> 

** **AB Testing: Want to decide whether Website A performed better than Website B in generating sales in your most recent pricing experiment?** Decide on prior distributions for $$p_A$$ and $$p_B$$, the probabilities that customers will convert on Website A and Website B respectively, then compute the posterior distributions $$\pi(p_A | X_A)$$ and $$\pi(p_B | X_B)$$ using Bayes' rule; Look at the posterior distribution $$\pi(p_A - p_B | X_A, X_B)$$ to understand the direction and magnitude of the difference between Website $$A$$ and Website $$B$$, and the credible intervals to quantify the uncertainty. This is a simple Bayesian AB testing framework which has very low computational requirements (with appropriate choices of conjugate prior), and can avoid a lot of the pitfalls of Frequentist AB testing (p-values and early stopping rules, I'm looking at you!)

<br>

As you can see, inferential statements on parameters convert to natural statements on probability distributions that we are used to making. This flexibility can be very useful when attempting to create a model of a more complicated data generating process, as we will see in the next section when creating a changepoint detection algorithm.


# A Basic Changepoint Detection Algorithm 

To put all of the theory given above into practice, let's define a basic offline Bayesian changepoint detection algorithm to a real-world problem. We are given a dataset consisting of the number of mining disasters that occurred in Great Britain between 1851 and 1962, and the problem is to try and identify whether any major changes have occurred in the ongoing rate of accidents per year. If there is strong evidence that a particular time period separates periods of high and low rate of mining accidents, then one can dig further into that time period: was there a piece of legislation enacted, or safety equipment introduced, that significantly improved mining safety?

<br> 

To begin, we model the number of mining accidents per year $$X_t$$ ($${1851 \leq t \leq 1962}$$) as a Poisson random variable with rate $$\lambda_t$$. 

<br>

At some year $$\tau$$ between 1851 and 1962, the rate $$\lambda$$ changes. Since we have no clear opinion of when this change may have occurred, we give $$\tau$$ a discrete uniform prior $$\tau \sim \text{Unif}(1851, 1962)$$, and write $$\lambda$$ as 

$$
\lambda_t = \begin{cases}
\lambda_1 \quad \text{for } 1851 \leq t \leq \tau, \\
\lambda_2 \quad \text{for } \tau < t \leq 1962.
\end{cases}
$$

What priors should we choose for $$\lambda_1$$ and $$\lambda_2$$? One answer here is to go with a prior which, thus far, agrees with the data that has been provided. Since $$\lambda_1$$ and $$\lambda_2$$ are _rates_ and hence correspond to units of time, we could consider using an Exponential distribution $$\text{Exp}(\alpha)$$ with scale parameter $$\alpha$$ equal to the inverse of the mean of the counts thus far: 

$$
\alpha = \left( \frac{1}{111} \sum_{i=1851}^{1962} X_t \right)^{-1}.
$$

This choice is "morally correct" in the sense that this is the "most likely value" for the scale parameter to be if the Exponential distribution was the true data generating process, but does suffer from double-dipping by defining parts of the model with the observed data then fitting the model onto the data. There is no evidence that this poses an issue, and further simulations using more uninformed priors (e.g. a $$\text{Gamma}(1, 0.5)$$ prior) converge to the same results. So we're happy to move on!

<br> 

At this point, with our likelihood and priors set in place, we are ready to define a model, compute the posterior and do some inference. I choose to use the probabilistic programming package PyMC to do the heavy lifting for me.

## Code
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

    N = last_year - first_year 

    # Define scale parameter for lambda priors
    alpha = 1 / np.mean(accidents_by_year)

    lambda_1 = pm.Exponential('lambda_1', lam=alpha)
    lambda_2 = pm.Exponential('lambda_2', lam=alpha) 

    # tau is the changepoint - discrete between 1851 and 1961
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

<br> 

![diagnostics charts](/images/diagnostic_charts.png)

<br>

The graphs on the left display the histograms of the simulated variables $$\tau$$, $$\lambda_1$$ and $$\lambda_2$$. The graphs on the right show the sequence of samples that are drawn in the simulation. No obvious issues such as non-zero autocorrelation and transient behaviour seem to be occurring in the drawm samples. There appears to be very strong evidence of a change point occurring around 1890, with a second mode occurring around 1887-1888. 

<br> 

![MCMC Diagnostics](/images/MCMC%20diagnostics.png)

<br>

We see these observations come through in the diagnostics table. Our model predicts a high probability of a changepoint in the number of yearly mining incidents occurring in the year 1890, with a significantly different rate of incidents before and after this change point with an average of $$3.11$$ incidents occurring before the changepoint and $$0.90$$ incidents occurring afterwards. 

<br>

Aside from simple posterior summary statistics neatly summarising the results into quantifiable statements, we can also quantify the uncertainty of our estimates with the $$3%$$ and $$97%$$ credible intervals (the acronym HDI used in the table means highest density interval, which is a type of credible interval which takes the top percentage of the density ordered by probability). 

<br>

Just to connect this back to reality, there was indeed a Royal Commision on Mining Royalties (1890-1891), with the Coal Mines Regulation Act (1887), Truck Amendment Act (1887) and Factory and Workshop Act (1891) occurring around this time period. That being said, coal mining and their disasters in Britain have had a long history, with many different acts, restrictions and laws put in place (see \[3\] for details), and I'm not entirely confident that the Bayesian methods used here have completely captured the nuances of that history. But it is a start, and a very interesting application of a simple changepoint algorithm.  

## Extensions
In writing out the above model, it became clear to me that there are some obvious ways of extending the model to be more flexible. Indeed, We assumed that there is only one changepoint, but we could just as easily define multiple $$\lambda$$'s and multiple changepoints. The number of changepoints doesn't even need to be fixed: if $$n$$ is the number of changepoints, we could put a uniform prior on $$n$$ and simulate to find the number of changepoints that match the data the best. 







# References
\[1\] [Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers) - this is a great resource on getting started with implementing Bayesian models in Python with PyMC. The text message example in Chapter 1 is almost an exact replica of the example I give above, with a few differences in the chosen distributions. 

<br>

\[2\] [Dataset1](https://www.cmhrc.co.uk/site/disasters/disasters_list_1850.html) and [Dataset2](https://www.cmhrc.co.uk/site/disasters/disasters_list_1900.html) - the dataset was provided as part of a class on Bayesian Inference (MATH5960, UNSW T3 2021), but some digging gave me a concrete listing of coal mine disasters in the provided years, with the counts summarised into the dataset provided in the code above. 

\[3\] [Government and mining](https://mininginstitute.org.uk/wp-content/uploads/2016/02/Government-and-mining-Jan16.pdf) - a history of coal mining disasters, royal commissions and acts enforced on the coal mining industry.
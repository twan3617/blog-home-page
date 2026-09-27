---
title: "Borel–Cantelli: Finite Probability Sums Rule Out Infinite Recurrence"
date: "2022-03-14"
description: "If event probabilities have a finite sum, only finitely many of those events occur almost surely."
topics: [Probability, Measure Theory]
featured: true
---

The Borel–Cantelli lemma says that if the probabilities of a sequence of events have a finite sum, the probability that infinitely many of them occur is zero. We prove the result and use it to show how small probabilities behave under a dominated finite measure.

<br>

_Notation_: We work in a measure space $$(\Omega, \mathcal{F}, P)$$ with $$\{A_n\}_{n \in \mathbb{N}}$$ a sequence of measurable sets. We call them "events", although the lemma does not require $$P(\Omega) = 1$$. The second measure $$Q$$ used below is finite.

_Definition 1: The limit superior of a sequence of measurable sets _$$\{A_n\}_{n \in \mathbb{N}}$$_ is the set_

$$
\limsup_{n \to \infty} A_n := \cap_{n=1}^\infty \cup_{k \geq n} A_k.
$$

Intuitively speaking, the limit superior contains the set of events $A_n$ which occur infinitely often in the sequence $\{A_k\}_{k \in \mathbb{N}}$ - indeed, any event which occurs finitely often can only be contained within a finite number of the unions $$\cup_{k \geq n} A_k$$, and hence cannot be in the intersection of all of those unions. Going off of this intuition, probability theorists may also write this as $A_n$ i.o, standing for "$$A_n$$ infinitely often".

<br>

_Definition 2: A measure $$P$$ is said to dominate a measure _$$Q$$_ (written _$$P >> Q$$_) if _

$$P(A) = 0 \implies Q(A) = 0.
$$

_Lemma 1 (Borel–Cantelli Lemma): Suppose _$$\sum_{k=1}^{\infty} P(A_k) < \infty$$_. Then _

$$
P(\limsup_{n \to \infty} A_n) = P(\cap_{n=1}^\infty \cup_{k \geq n}A_k) = 0.
$$


_Proof_: Write $$U_n = \cup_{k \geq n} A_k$$. These sets decrease: if $$n \geq m$$, then $$U_n \subseteq U_m$$. Moreover, $$P(U_1) \leq \sum_{k=1}^{\infty} P(A_k) < \infty$$, so continuity from above applies:

$$
P(\limsup_{n \to \infty} A_n) = \lim_{n \to \infty} P(U_n).
$$

For each $$n$$, subadditivity gives

$$
P(U_n) \leq \sum_{k=n}^\infty P(A_k).
$$

The tails of a convergent series tend to zero. We conclude that

$$
P(\limsup_{n \to \infty} A_n) = 0,
$$

as we wanted. 

_Corollary 1 (Small Events Under a Dominated Measure): Suppose _$$Q$$_ is a finite measure on _$$\mathcal{F}$$_ and _$$P$$_ dominates _$Q$_ (i.e., _$P >> Q$_). Then for every _$\varepsilon > 0$_, there exists _$\delta > 0$_ such that, for every measurable set _$A$_,_

$$ P(A) < \delta \implies Q(A) < \varepsilon.
$$

Small events under $$P$$ are also small under the finite dominated measure $$Q$$.

_Proof_: Suppose the claim fails for some $$\varepsilon_0 > 0$$. Then, for each $$k \in \mathbb{N}$$, we can choose a measurable set $$A_k$$ with

$$
P(A_k) < 2^{-k} \quad \text{but} \quad Q(A_k) \geq \varepsilon_0 > 0.
$$

Set $$U_n = \cup_{k \geq n} A_k$$ and $$A = \limsup_{n \to \infty} A_n$$. Since $$Q$$ is finite, continuity from above applies to the decreasing sets $$U_n$$. Each $$U_n$$ contains $$A_n$$, so $$Q(U_n) \geq \varepsilon_0$$ and therefore

$$
Q(A) = \lim_{n \to \infty} Q(U_n) \geq \varepsilon_0 > 0.
$$

But the lemma gives $$P(A) = 0$$, since $$\sum_{k=1}^{\infty} P(A_k) < \sum_{k=1}^{\infty} 2^{-k} = 1$$.

This contradicts $$P >> Q$$, because $$P(A) = 0$$ while $$Q(A) > 0$$.

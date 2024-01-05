---
title: "Infinitely Recurring Events occur with Probability Zero"
date: "2022-03-14"
---

In this post, we give short proofs for the Borel-Cantelli lemma, which, succinctly stated, says that events which reoccur infinitely often in a sequence of events must have probability zero. One of its main consequences is that events unlikely to occur under one probability measure is unlikely in any probability measure "dominated" by it. We make all of these concepts precise.

<br>

_Notation_: We work in an arbitrary measure space $$(\Omega, \mathcal{F}, P)$$ with $$\{A_n\}_{n \in \mathbb{N}}$$ a sequence of measurable sets. While we interpret the measure $$P$$ as a probability measure and measurable sets as "events" in the outcome space $$\Omega$$, there is no explicit requirement that $$P(\Omega) = 1$$ for the theory to work. 

<br>

The expectation $\mathbb{E}$ refers to the integral operator 

$$
\mathbb{E}[f] = \int_\Omega f(\omega) \: dP(\omega).
$$

_Definition 1: The limit superior of a sequence of measurable sets _$$\{A_n\}_{n \in \mathbb{N}}$$_ is the set_

$$
\limsup_{n \to \infty} A_n := \cap_{n=1}^\infty \cup_{k \geq n} A_k.
$$

Intuitively speaking, the limit superior contains the set of events $A_n$ which occur infinitely often in the sequence $\{A_k\}_{k \in \mathbb{N}}$ - indeed, any event which occurs finitely often can only be contained within a finite number of the unions $$\cup_{k \geq n} A_k$$, and hence cannot be in the intersection of all of those unions. Going off of this intuition, probability theorists may also write this as $A_n$ i.o, standing for "$$A_n$$ infinitely often".

<br>

_Definition 2: A measure $$P$$ is said to dominate a measure _$$Q$$_ (written _$$P >> Q$$_) if _

$$P(A) = 0 \implies Q(A) = 0.
$$

_Lemma 1 (Borel-Cantelli Lemma): Suppose _$$\sum_{k=1}^n \mathbb{E}[A_n] < \infty$$_. Then _

$$
P(\limsup_{n \to \infty} A_n) := P(\cap_{n=1}^\infty \cup_{k \geq n}A_k) = 0.
$$


_Proof_: Note that $$\cup_{k \geq n}A_k$$ is a decreasing sequence of sets in $$n$$, with $$\cup_{k \geq n} A_k \supset \cup_{k \geq m} A_k$$ if $$n \geq m$$. Hence, by the continuity of measures, we can pull out the limit:

$$
P(\cap_{n=1}^\infty \cup_{k \geq n} A_k) = \lim_{n \to \infty} P(\cup_{k \geq n} A_k).
$$

Now, each for each $$n$$, we have 

$$
P(\cup_{k \geq n} A_k) \leq \sum_{k=n}^\infty P(A_k)
$$

by the sub-additivity property of measures. Since the total sum $$\sum_{k=1}^\infty P(A_k)$$ is assumed to be finite, the tail sums 

$$
\sum_{k = n}^\infty P(A_k)
$$ 
must converge to zero as $n \to \infty$. We conclude that 

$$
P(\limsup_{n \to \infty} A_n) = 0,
$$

as we wanted. 

_Corollary 1 (Unlikely Events are Equally as Unlikely in Dominated Measures): Suppose _$$Q$$_ is another measure on _$$\mathcal{F}$$_ and _$$P$$_ dominates _$Q$_ (i.e., _$P >> Q$_). Then for all _$\varepsilon > 0$_, there exists _$\delta > 0$_ such that_

$$ P(A) < \delta \implies Q(A) < \varepsilon.
$$

We interpret this condition probabilistically as follows: unlikely events under $$P$$ forces those same events to be unlikely under any dominated measure $$Q$$. 

_Proof_: We argue by contradiction. Suppose there exists some $$\varepsilon_0 > 0$$ such that for all $$k \in \mathbb{N}$$, we have a sequence of measurable sets $$A_k$$ with 

$$
P(A_k) < 2^{-k} \quad \text{but} \quad Q(A_k) \geq \varepsilon_0 > 0.
$$

Define $$A := \limsup_{n \to \infty} A_n = \cap_{n=1}^\infty \cup_{k \geq n} A_k$$. Then 

$$
Q(A) = Q(\limsup_{n \to \infty} A_n) \\
=^{(*)} \lim_{n \to \infty} Q(\cup_{k \geq n}A_k) \\
\geq^{(**)} Q(A_n) \quad \text{for all } n \in \mathbb{N}, \\
\geq \varepsilon_0 > 0,
$$

where in Step $$(*)$$ we used the continuity of the measure $$Q$$ (again, the unions of $$A_k$$ is a decreasing sequeunce), and in Step $$(**)$$ we the fact that every union $$\cup_{k \geq n} A_k$$ contains $$A_n$$. However, we must have $$P(A) = 0$$ by the Borel-Cantelli lemma, since 
$$
\sum_{k=1}^\infty P(A_k) < \sum_{k=1}^\infty 2^{-k} < 1 < \infty.
$$

This contradicts the fact that $$P >> Q$$, since we have a set of measure zero under $$P$$ that does not have measure zero under $$Q$$. This completes the proof.

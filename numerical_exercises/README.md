# Numerical verifications

Implement numerically some algorithms to confirm intuition on problems of combinatorics and discrete probabilities.

1. [Counting positive paths](#positive-paths-count)
2. [Optimising some discrete order execution](#discrete-optimal-execution)
3. [Random walk crossing threshold probability](#negative-price-probability)
4. [Dealing cards evenly without observation](#dealing-indistinguishable-cards)
5. [Counting diagonal paths](#counting-diagonal-paths)

*Also includes :*
- detours on combinatorics and proofs
- detours revolving around the Rust environment
***
## Positive paths count
Given an integer $n$, count the number of positive paths
that start and end at $0$
and moving $\pm 1$ at each step.

**Solution**

<p style="color:green">
The closed-form expression can be written as Catalan numbers, 
which also represent the number of balanced parenthesis.
</p>

$$ C_{2n} = \begin{pmatrix} 2n \\ n \end{pmatrix} - \begin{pmatrix} 2n \\ n + 1\end{pmatrix}  = \frac{1}{n+1} \begin{pmatrix} 2n+1 \\ n \end{pmatrix}$$

But as one does not necessarily remember this formula from the top of it's head, 
dynamic programming is also implemented.

Computing incrementally all $C(j, k)$ for $0 \leq j \leq k $, being the number of paths 
starting at $0$ and ending at $2k$ in $2j$ steps.
Then the solution is given by $C(n, 0)$.

From a direct first-step analysis, the recurrence relation is :
$$ C(j+1, k) = 
\begin{cases}
2 C(j, k) + C(j, k-1) + C(j, k+1) \ \text{if 0 < k < j} \\
2 C(j, j) + C(j, j-1) \ \text{if k = j}\\
C(j, 1) + C(j, 0) \ \text{if k = 0}
\end{cases} $$

> Tested the [benching crate](https://doc.rust-lang.org/cargo/commands/cargo-bench.html) to have an idea of feasible $n$ with direct computations and dynamic programming. At the time of writing it requires to use the *nightly* channel :
```bash
cargo +nightly run
```

<p style="color:orange">
<i>
This solution quickly runs into numerical overflow with u32.
</i>
</p>

***
## Discrete optimal execution

An agent has to select $n$ values sequentially.
He has the choice to either take the value from a uniform $\mathcal{U} [a, b]$ 
random variable, 
or to select a fixed-value alternative.

The alternative value can be selected at most $k$ times.

The objective is to find the optimal strategy and compute the expected value over the $n$ steps.

| Variable | Description |
| --- | --- |
| n | Total number of steps |
| p | Fixed alternative value  |
| k | Maximum number of times not using the uniform draw  |
| a, b | Range of the uniform law |

<p style="color:green">
Modelling the problem :
</p>

Assume  $k \geq 1, p < b $ and $n \geq 1$.
The objective function can be written :

$$
M = \min \left( \mathbb{E} \left[ \sum_{j=1}^{n} x_j \right] \right)
$$

Where the $x_j$ are the draws from the random variables resulting of our strategy, which are not independant as decisions are made sequentially :

$$
M = \min \left( \mathbb{E} \left[ \sum_{j=1}^{n} \mathbb{E} \left[x_j | \mathcal{F}_{j-1} \right] \right] \right)
$$

If we know the uniform draws in advance, the optimal strategy would be to choose the $k$ larger draws and to replace it with $\min (x_{(i)}, p)$
where the $x_{(i)}$ are the ordered variables.

It gives a lower bound to the problem :

$$
M \geq \sum_{j=1}^{n-k} \left( a + \frac{j(b-a)}{n+1} \right) 
+\sum_{j=1}^{k} \min (p, a +\frac{(n-k+j)(b-a)}{n+1})
$$

<p style="color:green">
Strategy : keep a flat threshold at the value of the top-k uniform value of n independant draws. <br>
If we reach the point where there are as many k than n, then apply the min.
</p>

The threshold then is $a + \frac{(n-k)(b-a)}{n+1} $

$$
\begin{align*}
\mathbb{E} \left[ \sum_{j=1}^{n} x_j \right] & = 
\mathbb{E} \left[ \sum_{j=1}^{n} \sum_{m=0}^{j \wedge k} \mathbb{E} \left[ x_j | k_j = m \right] P(k_j = m) \right]
\end{align*}
$$

Where the variable $k_i \in [0, i]$ is the number of $u_1, \dots , u_{i-1} $ being above the threshold 
(the factorisation is possible since $u_i \perp (u_j)_{j < i }$ ).

The sum can be rewritten :

$$
\begin{align*}
\mathbb{E} \left[ \sum_{j=1}^{n} x_j \right] & = 
\mathbb{E} \left[ \sum_{j=1}^{k} \sum_{m=0}^{j} \mathbb{E} \left[ x_j | k_j = m \right] P(k_j = m) + \sum_{j=k+1}^{n} \sum_{m=0}^{k} \mathbb{E} \left[ x_j | k_j = m \right] P(k_j = m)\right] \\
& = \int \dots \int \sum_i x_i \frac{du_1 \dots du_n}{(b-a)^n} \\
& = \sum_{\sigma \in P_n} \int_{u_{\sigma(1)} < \dots < u_{\sigma(n)}} \dots \int \sum_i x_i(u_1, \dots , u_n) \frac{du_1 \dots du_n}{(b-a)^n}
\end{align*}
$$


**Law of top-k drawn from a uniform distribution :**

We used the fact fact that the expectation of the top-k value of n independant draws from a uniform variable is : 

$$ 
x_{(k)}^n = a + \frac{k (b-a)}{n+1}
$$

To simplify the computation, let $(u_i)_{i\in [1, \dots, n]} \sim \mathcal{U}[0, c] $ and independant.

Let $k \in [1, n]$ and $\left( u_{(i)} \right)_{i \in [1, \dots, n]}$ the ordered variables.

$$
\begin{aligned}
\mathbb{E} \left[ u_{(k)} \right] &=
\frac{1}{c^n}\int \cdots \int u_{(k)} du_1\cdots du_n \\
&= \frac{n! }{c^n} \int_{u_1 < \cdots < u_n} \int u_k du_1\cdots du_n \\
&= \frac{n! }{(k-1) ! c^n} \int_{u_k < \cdots < u_n} u_k \times u_k^{k-1}\  du_k \cdots du_n \\
&= \frac{n! }{(k-1) ! (k+1)c^n} \int_{u_{k+1} < \cdots < u_n} u_{k+1}^{k+1} \  du_{k+1} \cdots du_n \\
&= \frac{n! }{(k-1) ! (k+1)\cdots n c^n} \int_{0 < u_n < c} u_{n}^{n} \  du_{n} \\
&= \frac{n! }{(k-1) ! (k+1)\cdots n (n+1) c^n} \times c^{n+1} \\
&= \frac{k c}{n+1} 
\end{aligned}
$$

***
## Negative price probability
Compute the probability of a random walk reaching zero 
before $n$ steps, starting at $s$,
moving by $\pm \delta s$ at each step, 
with probability of moving up being $p \in [0,1]$.

| Variable | Description |
| --- | --- |
| n | Total number of steps |
| p | Probability of going up |
| s | Initial value |
| ds | Step size |

<p style="color:green">
Implemented : <br> 
</p>

- Monte Carlo estimate for verification :
    - simulate a trajectory and check whether is gets to zero
- Incremental solution :
    - Disjoint cases on "how many times the trajectory goes up" :
    - Catalan number being the number of positive paths starting and ending at 0

Looking for paths that do cross zero at $2k+r$ steps for the first time, 
the probability of any of these paths will be :

$$ \tilde{p_k} = p^k (1-p)^{k+r}$$

Let : 

$$r = \lfloor \frac{s}{\delta s} \rfloor $$

(the number of steps needed to reach / cross zero)

The number of paths that will reach zero at step $2k+r$ is : 

$$
c_k = \sum_{G \ \text{integer partition of k}} \left( \prod_{g_i \in G} C_{2g_i}\right) \times \begin{pmatrix} r \\ s \end{pmatrix}
$$

So the solution will be for each $k$ such that $2k + r \leq n $, to incrementally add $\tilde{p_k} c_k $.

**Intuition behind $c_k$ :**

We assume to know that the Catalan numbers represent the number of positive paths of $2n$ steps.

Thinking in terms of *"when does a valid path goes irreversibly down by one"*,
a path that will rise $k$ times and go down $k+r$ times will :
- go down at the last step : remains $r-1$ "right parenthesis" to place,
- "going up $k$ times" can be decomposed in groups of integer partitions of $k$, where each sub-path becomes a positive one (~ Catalan number).
We need at least one right parenthesis between these groups so that they cannot be ambiguous. So if the length of the integer partition is $j$, we need to use $j-1$ right parenthesis at this point to distinguish the "sub-positive-paths" which count we already know.
- The remaining $r-1 - (j-s) = r-j$ right parenthesis to place are equivalent to placing $r-1$ indistinct balls into $j + 1$ distinct buckets, where $j$ is the length of the integer partition of $k$.
This values is :

$$ 
\begin{pmatrix}
(r-j) + (j+1) - 1 \\
(j+1) - 1
\end{pmatrix}= \begin{pmatrix}
r \\
j
\end{pmatrix}
$$

**Efficient computation of integer partitions :**
- [Kelleher, Jerome, and Barry O'Sullivan. "Generating all partitions: A comparison of two encodings." arXiv preprint arXiv:0909.2331 (2009).](https://arxiv.org/pdf/0909.2331)
- https://jeromekelleher.net/generating-integer-partitions.html

**(Weak) Composition of an integer**
The number of ways to decompose an integer $n$ into $k$ non-negative integers is :

$$ 
\begin{pmatrix}
n + k- 1 \\
k - 1
\end{pmatrix}
$$

It is the number of ways one can dispatch $n$ indistinct balls into $k$ distinct buckets, and is also equivalent to the *stars and bars* problem.

The binomial number can be understood as the number of sequences of length $n + k - 1$ ,
containing $n$ unit elements and $k - 1$ separation elements 
(having then $k$ parts, possibly empty to represent zero).


**(Strong) Composition of an integer**

The number of ways to decompose an integer $n$ into $k$ positive integers is :

$$\begin{pmatrix} n - 1 \\ k-1 \end{pmatrix}$$

which [can be seen as](https://en.wikipedia.org/wiki/Composition_(combinatorics)) choosing $k-1$ seperations between $n$ units (therefore having $n-1$ intervals).

One can obtain the result also from point of view of the weak composition, 
seeing, after dispatching one unit to each $k$ parts, that $n-k$ 
indistinguishable units are left to place in $k$ distinct buckets.

**Benching ascending integer partitions algorithms**

Both algorithms from  https://jeromekelleher.net/generating-integer-partitions.html are implemented, 
with the significant difference that they are eager, whereas the Python 
example generates an iterator which in practice is a lot more memory-efficient.

Still, using the `bench` tool from rust, the efficient algorithm has a smaller variance than the simpler one :

```bash
cargo +nightly bench
test tests::bench_exec_time_bf  ... bench:       2,195.70 ns/iter (+/- 959.86)
test tests::bench_exec_time_dyn ... bench:       1,130.59 ns/iter (+/- 684.28)
```

***

## Dealing indistinguishable cards 

With a deck of 52 cards 
and the initial information that it comprises 17 cards facing up and 35 facing down,
find an algorithm so to dealing the deck in two piles,
such that the two piles have the same number of cards facing upwards.
The only allowed operation is to flip an arbitrary amount of cards.
Cards are not visible before or at any time during the algorithm.

**Solution**

<p style="color:green">
Take 17 of those cards, flip all of them. 
This forms one pile and the other cards - untouched - go in the other pile.
</p>


## Other references
- https://laurentmazare.github.io/

## Counting diagonal paths

Starting at $(0,0)$, 
how many paths are there to reach $(4,6)$,
going either UP or RIGHT at each turn,
and avoiding to go three times in the same directlion successively ?

<p style="color:green">
The total number of paths, without the constraint is :

$$\begin{pmatrix} 10 \\\ 4 \end{pmatrix}$$

The objective is then to count the number of paths that are not valid,
that is paths containing 
one or more sequence of three successive UP or RIGHT.
</p>

> [!NOTE]
> To complete

## Probability of fortune before ruin

I currently own $3$ coins and want to reach $5$.
Each turn I bet the maximum amount up to what is necessary to reach $5$.
The probability of winning each turn is $\frac{2}{3}$ - 
the amount won is the number of betted coins.
What is the probability of reaching $5$ before loosing all the coins ?

> [!NOTE]
> To complete

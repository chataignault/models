# Numerical verifications

Implement numerically algorithms to verify problem solutions.

***
## Positive paths count
Given integer $n$, count the number of positive paths
that start and end at $0$
and moving $\pm 1$ at each step.

<p style="color:green">
The closed-form expression can be written as Catalan numbers, 
which also represent the number of balanced parenthesis.
</p>

$$ C_{2n} = \begin{pmatrix} 2n \\ n \end{pmatrix} - \begin{pmatrix} 2n \\ n + 1\end{pmatrix}  = \frac{1}{n+1} \begin{pmatrix} 2n+1 \\ n \end{pmatrix}$$

But as one does not know this from the top of it's head, 
dynamic programming is also implemented.

It is a strong recurrence on $C(j, k)$, being the number of paths 
starting at $0$ and ending at $k$ in $2j$ steps.
Then the solution is $C(n, 0)$.

$$
C(j+1, k) = 
\begin{cases}
2 C(j, k) + C(j, k-1) + C(j, k+1) \ \text{if 0 < k < j} \\
2 C(j, j) + C(j, j-1) \ \text{if k = j}\\
C(j, 1) + C(j, 0) \ \text{if k = 0}
\end{cases}
$$

<p style="color:orange">
<i>
This solution quickly runs into numerical overflow with u32.
</i>
</p>

***
## Discrete optimal execution

An agent has to select $n$ values sequentially.
He has the choice to either take the value from $\mathcal{U} [a, b]$ 
random variable, 
or to select a fixed-value alternative.

The alternative value can be selected at most $k$ times.

What is the optimal strategy and compute the expected value over the $n$ steps.

| Variable | Description |
| --- | --- |
| n | Total number of steps |
| p | Fixed alternative value  |
| k | Maximum number of times not using the uniform draw  |
| a, b | Range of the uniform law |

<p style="color:green">
Problem definition :
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

Where the variable $k_i \in [0, i]$ is the number of $u_1, \dots, u_{i-1} $ being above the threshold 
(the factorisation is possible since $u_i \perp (u_j)_{j < i }$ ).

The sum can be rewritten :

$$
\begin{align*}
\mathbb{E} \left[ \sum_{j=1}^{n} x_j \right] & = 
\mathbb{E} \left[ \sum_{j=1}^{k} \sum_{m=0}^{j} \mathbb{E} \left[ x_j | k_j = m \right] P(k_j = m) + \sum_{j=k+1}^{n} \sum_{m=0}^{k} \mathbb{E} \left[ x_j | k_j = m \right] P(k_j = m)\right] 
\end{align*}
$$


**Law of top-k drawn from a uniform distribution :**

We used the fact fact that the expectation of the top-k value of n independant draws from a uniform variable is : 

$$ 
x_{(k)}^n = a + \frac{k (b-a)}{n+1}
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
\end{pmatrix}
= 
\begin{pmatrix}
r \\
j
\end{pmatrix}
$$

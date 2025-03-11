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

***
## Discrete optimal execution

***
## Negative price probability
Compute the probability of a random walk reaching zero 
before $n$ steps, starting at $s$ and 
moving by $\pm \delta s$ at each step, 
with probability of moving up being $p \in [0,1]$.

<p style="color:green">
Implemented : <br> 
n : total number of steps <br>
p : probability of going up <br>
s : initial value <br>
ds : step size
</p>

- Monte Carlo estimate for verification :
    - simulate a trajectory and check whether is gets to zero
- Incremental solution :
    - Disjoint cases on "how many times the trajectory goes up" :
    - Catalan number being the number of positive paths starting and ending at 0

Looking for paths that do cross zero after $2k+r$ steps, 
the probability of any of these paths will be :
$$ p^k (1-p)^{k+r}$$
The number of possible paths is : 
$$
\sum_{G \ \text{integer partition of k}} \left( \prod_{g_i \in G} C_{2g_i}\right) \times \begin{pmatrix} r \\ s \end{pmatrix}
$$
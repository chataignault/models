
## Combinatorics

Picking $5$ cards out of a deck containing $5$ pairs.

> What is the expected number of pairs picked ?

***

No replacement and indistinguishable pairs.
Combinatorics of parts of a fixed size of a set (binomial number).

For a given number of pairs picked, (how many possible pair combination), then of the remaining cards to pick, 
select cards from distinct remaining pairs.  

With $ \lfloor \frac{n}{2} \rfloor \leq p $,

$$
\mathbb{E} \left[ P \right] = \sum_{k=1}^{\lfloor n / 2 \rfloor }
k \frac{
    \begin{pmatrix} p \\ k \end{pmatrix}
    \begin{pmatrix} p-k \\ n - 2k \end{pmatrix}
}{
    \begin{pmatrix} 2p \\ p \end{pmatrix}
}
$$


## Counting

$ N = 100 $, $x \in [ 1 \dots 10 ]$ 

Taking turns, first to $100$ wins.

> Strategy and how should start first ?

***

Start and add $1$ to the total count. 
Then complete each turn to total $11$ with what the opponent plays.
Reach $89$ at the end of my turn.
Whatever the other player do $100$ can be reached at the next move.

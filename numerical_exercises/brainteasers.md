
## Combinatorics

### Expected number of pairs

Picking $5$ cards out of a deck containing $5$ pairs.

> What is the expected number of pairs picked ?

<p style="color:green">
No replacement and indistinguishable pairs.<br>
Combinatorics of parts of a fixed size of a set (binomial number).<br>
For a given number of pairs picked, (how many possible pair combination), then of the remaining cards to pick, 
select cards from distinct remaining pairs.  
</p>

With $\lfloor \frac{n}{2} \rfloor \leq p$,

$$
\mathbb{E} \left[ P \right] = \sum_{k=1}^{\lfloor n / 2 \rfloor }
k \frac{
    \begin{pmatrix} p \\ k \end{pmatrix}
    \begin{pmatrix} p-k \\ n - 2k \end{pmatrix}
}{
    \begin{pmatrix} 2p \\ p \end{pmatrix}
}
$$

***

### Ordering probability

A deck of 64 cards is well shuffled.

Cards are drawn one by one until the 2 of hearts is drawn and the game stops.

> What is the probability of drawing exactly one king, one queen and one jack before the game stops ?

***

There are 4 cards for each of the three heads and and the 2 of hearts.
The ordering of all other cards does not matter.

Therefore the probability model is considered on the ordering and position
of the 13 distinguishable cards.

Since :

- the number of ordering of 13 elements is $13!$,
- the number of position combinations these cards can be placed at is given by binomial numbers,

The total number of combinations is :

$$13 ! \begin{pmatrix} 64 \\ 13 \end{pmatrix}$$

Then, a combination that is valid satisfies :

- there is exactly one king, one queen and one jack at the start,
- then the 2 of hearts
- then all other cards

The number of combination for the three first cards is $4^3 = 64 $, their order is $3! = 6$,
and the number of ordering for the remaining 9 distinguishable cards 
(therefore excluding the 2 of hearts since it has a fixed position)
is $9!$.

The number of combinations for the indistinguishable cards is the same as before.

Eventually, the probability is :

$$\frac{6 \times 64 \times 9!}{13 !}$$

## Optimal Algorithms

$N = 100$, $x \in [ 1 \dots 10 ]$ 

Taking turns, first to $100$ wins.

> Strategy and how should start first ?

<p style="color:green">
Start and add 1 to the total count. <br> 
Then complete each turn to total $11$ with what the opponent has played
(if opponent adds k to the total, place 11-k). <br>
After the 8th round, the sum reaches 89, being at opponent's turn. <br>
Whatever the other player does, 100 can be reached at the next move. <br>
</p>

***

One player game with two hidden boxes.
For 100 turns, the player chooses either to `place` or `take`.
- `place` adds $1$ to either boxes at random.
- `take` returns the content of either box at random, 
regardless of their content.

The player can only know how much he obtained at the end of the game.

> What is the optimal strategy ? What is the expected sum that the player will get ?

<p style="color:green">
If there was only one box, the optimal strategy would be to `place` until the 99th round and the `take`. 
The expected payoff of doing so is 99 and is an upper bound for the game. <br>
The same strategy but with the two boxes has an expected payoff of 49,5. <br>
The problem is then :
</p>

$$
\left(\mathcal{P}\right):\ \max_m \left[ \frac{100 - m}{2} \sum_{k=0}^{m-1} 2^{-k}  \right] \\
\left(\mathcal{P}\right):\ \max_m \left[ (100 - m) \left( 1 - 2^{-m} \right) \right]
$$

<p style="color:green">
Which is maximum for m=6,
so the optimal strategy is to `place` 94 times and then `take` until the game finishes. <br>
The expected sum is then 2961/32 ~ 92,5.
</p>

## Other Probabilities

Flower $A$ and $B$ blossom in the coming $40$ days uniformly independently.
Flower $A$ lasts $4$ days and flower $B$ blossoms during $16$ days. 

> What is the probability to observe both flowers blossomed in the next 40 days ?

***

$$
[A, (A+4) \wedge 40 ] \cap [B, (B+16) \wedge 40 ] \neq \emptyset \\
\Leftrightarrow \left(B \leq A \leq B+16\right) \cup \left( A \leq B \leq A+4 \right) \\
\Leftrightarrow -4 \leq A - B \leq 16   
$$

While the distribution $A - B$ can be computed, 
it seems more direct to condition the computation on the realisation of either variable.

$$
\begin{align*} 
p & = P (-4 \leq A - B \leq 16) \\
& = \sum_{k=1}^{40} P (-4 + B \leq A \leq 16 + B | B = k) P(B=k) \\
& = \frac{1}{40} \left( \sum_{k=1}^3 p_k + \sum_{k=4}^{24} p_k + \sum_{k=25}^{40} p_k \right)
\end{align*}
$$

With distinct cases depending on the contact with either boundaries of the interval.

$$
\begin{cases}
p_k = \frac{16 + k}{40}, \hspace{.5cm} k \in 1\dots 3 \\
p_k = \frac{1}{2}, \hspace{.95cm} k \in 4 \dots 24 \\
p_k = \frac{44 - k}{40}, \hspace{.5cm} k \in 25 \dots 40
\end{cases}
$$

Eventuallly : 
$$p = \frac{654}{1600}$$

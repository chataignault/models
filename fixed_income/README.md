# FICC

Fixed-Income and Commodities related topics.

## PV01 and DV01 for a coupon-paying bond

While : 
$$ 
\text{PV01} = \frac{\partial \text{PV}}{\partial y} \leq 0
$$
values will be considered in absolute term all along.

With a bond paying some coupon on a regular schedule, 
the cash-flows might look like this :


<img src="img/bond_cf.png" width="500">

Different assumptions can be made on the term structure,
either it goes upwards or downwards.

<img src="img/term_structures.png" width="500">

The highest sensitivity, for either of the PV01 or DV01 
goes at the highest maturity :

<img src="img/dv01.png" width="500">

Due to convexity considerations, 
the DV01 will be relatively more important at the tail 
for a downward term structure compared to an upward term structure.

<img src="img/dv01_diff.png" width="500">

(green is positive and red negative in log scale)

Considering the *flat rate* assumption of PV01,
the former plot also involves that if the term structure is **downward sloping**, then (in absolute terms) :

$$
\text{PV01} \leq \text{DV01}
$$

and vice-versa.

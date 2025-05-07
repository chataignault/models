# Probabilitic PCA on MNIST dataset

A simple idea to generative modelling, as mentionned in 
*Bengio, Yoshua, Ian Goodfellow, and Aaron Courville. Deep learning. Vol. 1. Cambridge, MA, USA: MIT press, 2017.* 
is to consider the PCA on some dataset, and to add noise to the decomposed matrix
to generate samples that were not in the training set, 
but that share the same principal components with their respective amplitude.

![image](img/ppca_mnist.png)

### References :

```bibtex
@article{tipping1999probabilistic,
  title={Probabilistic principal component analysis},
  author={Tipping, Michael E and Bishop, Christopher M},
  journal={Journal of the Royal Statistical Society Series B: Statistical Methodology},
  volume={61},
  number={3},
  pages={611--622},
  year={1999},
  publisher={Oxford University Press}
}
```

which has more applications like dealing with missing data,
maximum likelihood comparison with other probabilistic models for model selection 
and covariance matrix fitting for Gaussian mixture models.

```bibtex
@book{bengio2017deep,
  title={Deep learning},
  author={Bengio, Yoshua and Goodfellow, Ian and Courville, Aaron and others},
  volume={1},
  year={2017},
  publisher={MIT press Cambridge, MA, USA}
}
```
**For the computation of the Singular Value Decomposition :** 

The naive SVD algorithm, pure QR with Householder reflections and bidiagonalisation algorithm are taken from `Trehefen`, 
while the SVD implementation with Givens rotations is from `Golub`.

```bibtex
@book{trefethen2022numerical,
  title={Numerical linear algebra},
  author={Trefethen, Lloyd N and Bau, David},
  year={2022},
  publisher={SIAM}
}
```

```bibtex
@book{golub2013matrix,
  title={Matrix computations},
  author={Golub, Gene H and Van Loan, Charles F},
  year={2013},
  publisher={JHU press}
}
```
The bidiagonalisation is computed with full orthogonal matrices, 
that is to say for $A \in \mathbb{R}^{n \times m}$ : 

$$ A = U B V^T $$

where $U\in \mathbb{R}^{n \times n}$ and $V \in \mathbb{R}^{m \times m}$ .

A reduced form can be found, for instance if $m > n$, the matrices would be of the form :

$$
\begin{cases}
U\in \mathbb{R}^{n \times n} \\
A\in \mathbb{R}^{n \times n} \\
V\in \mathbb{R}^{n \times m}
\end{cases}
$$

But doing so, $V$ is not orthogonal anymore.

Decomposing matrices $A$ and $V$ into sub square matrices $\tilde{A}, \tilde{V}$:

$$ 
\begin{cases} 
A = \begin{pmatrix} \tilde{A} & R \end{pmatrix} \\
V = \begin{pmatrix} \tilde{V} & Z \end{pmatrix}
\end{cases}
$$

while applying the normal Golub bidiagonalisation algorithm to $\tilde{A}$ :

$$ 
A = U \begin{pmatrix} \tilde{B} & R \end{pmatrix} 
\begin{pmatrix} \tilde{V}^T \\ Z^T \end{pmatrix} 
$$

and $Z$ can be left to compute at the end by solving the system :

$$
\tilde{B} Z^T = U^T R
$$

where $\tilde{B}$ is square and upper bi-diagonal.

A similar system is solved if $A$ is thin instead of wide, with the same complexity.

**TODO :**
- [ ] Implement MLE estimation of PPCA with EM algorithm
- [ ] Benchmark execution time between naive SVD and Golub-Kahan algorithm

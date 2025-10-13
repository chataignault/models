# Simulate 2-d gaussian with 2-d uniform

Let $Y_1, Y_2 \sim \mathcal{N}(0,1)$ two independant gaussian random variables.

The characteristic function of the pair is :

$$
\begin{aligned}
\phi(Y_1, Y_2)(t,s) & = \mathbb{E} \left[ \exp \left( i ( tY_1 + sY_2 ) \right) \right] \\
& = \int_{\mathbb{R}^2} e^{i(ty_1 + sy_2)} \frac{1}{2\pi} \exp \left( -\frac{y_1^2 + y_2^2}{2} \right) dy_1 dy_2
\end{aligned}
$$

Then applying cylindrical change of variables, one gets :

$$\phi(Y_1, Y_2)(t,s) = \int_{\mathbb{R}\_+} d\rho \int_{[0, 2 \pi ]} d \theta \frac{\rho}{2\pi} 
\exp \left( i (t\rho \cos \theta + s \rho \sin \theta) - \frac{\rho^2}{2} \right) $$

Eventually, with the change of variable :

$$
\begin{cases}
\theta = 2\pi v \\
\rho = \sqrt{-\log (u^2) }
\end{cases}
$$

One obtains the form :

$$\phi(Y_1, Y_2)(t,s) = \phi\left(\sqrt{-\log (U^2)}\cos (2 \pi V) ,\sqrt{-\log (U^2) } \sin (2 \pi V)\right)(t,s) $$

Where the pair $U, V$ is an independant pair of uniform variables on $[0, 1]$.

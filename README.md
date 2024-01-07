# arbitragerepair
Python modules and jupyter notebook examples for the paper [Detect and Repair
Arbitrage in Price Data of Traded Options](https://arxiv.org/abs/2008.09454).

[![DOI](https://zenodo.org/badge/288789939.svg)](https://zenodo.org/badge/latestdoi/288789939)

## Repair option price data -- remove arbitrage
For a finite collection of call option prices written on the same underlying
asset, there are six types of constraints that ensure their prices to be
statically arbitrage-free. Given violation of the arbitrage constraints, we
repair the price data in a fast, model-independent way to remove all arbitrages.

Some examples of arbitrage repair are shown as below:

![image](https://user-images.githubusercontent.com/32545513/83334755-9666ad80-a2a0-11ea-9910-34137539517b.png)

An arbitrage-free normalised call price surface
<img src="https://render.githubusercontent.com/render/math?math=(T,k) \mapsto c(T,k)">
should satisfy some shape constraints. Assuming smooth surface function
<img src="https://render.githubusercontent.com/render/math?math=c(T,k) \in C^{1,2} (\mathbb{R}_{ %3E 0} \times \mathbb{R}_{\geq 0})">,
then these shape constraints are
- Positivity: <img src="https://render.githubusercontent.com/render/math?math=0 \leq c \leq 1">
- Monotonicity: <img src="https://render.githubusercontent.com/render/math?math=-1\leq\partial c / \partial k \leq 0">, <img src="https://render.githubusercontent.com/render/math?math=\partial c / \partial T \geq 0">
- Convexity: <img src="https://render.githubusercontent.com/render/math?math=\partial^2 c / \partial^2 k \geq 0">

## Code

### Installation

It is recommended to create a new virtual environment and install from pypi.

>```
>pip install arbitragerepair
>```

**Requirements:** ```Python >= 3.8``` and ```CVXOPT >= 1.3.0```.

### Usage

While an example can be found in [this jupyter notebook](notebook/example.ipynb),
the usage of this code consists of the following steps.

**0.** Import packages

>```
>from arbitragerepair import constraints, repair
>```

**1.** Normalise strikes and call prices

The inputs are four 1D ```numpy.array``` objects ```T, K, C, F```, which should be of the same length, representing 
time-to-expiries, strike prices, option prices and forward prices. 

>```
>normaliser = constraints.Normalise()
>normaliser.fit(T, K, C, F)
>T1, K1, C1 = normaliser.transform(T, K, C)
>```

**2.** Construct arbitrage constraints and detect violation
>```
>mat_A, vec_b, _, _ = constraints.detect(T1, K1, C1, verbose=True)
>```
Setting `verbose=True`, an arbitrage detection report will be shown. An example
is shown below:
>```
>Number of violations to non-negative outright price:                   0/13
>Number of violations to non-negative and unit-bounded vertical spread: 0/130
>Number of violations to non-negative butterfly spread:                 0/104
>Number of violations to non-negative calendar (horizontal) spread:     0/0
>Number of violations to non-negative calendar vertical spread:         9/513
>Number of violations to non-negative calendar butterfly spread:        126/3070
>```

**3.** Repair arbitrage

Repair using the
<img src="https://render.githubusercontent.com/render/math?math=\ell^1">-norm
objective:

>```
>epsilon = repair.l1(mat_A, vec_b, C1)
>```

Repair using the
<img src="https://render.githubusercontent.com/render/math?math=\ell^1">-BA
objective (where bid-ask spreads data need to be supplied):

>```
>epsilon = repair.l1ba(mat_A, vec_b, C1, spread)
>```

**4.** De-normalise

>```
>K0, C0 = normaliser.inverse_transform(K1, C1 + epsilon)
>```

## Why is repairing arbitrage in price data important?

### Frequent presence of arbitrage in historical price data

We collect daily close (bid, ask and mid) prices from 1st November, 2007 to 31st May, 
2018 for OTC FX options from Bloomberg. We count violations of arbitrage constraints 
in raw daily close mid-prices over time for some major currencies and emerging market 
(EM) currencies.

![Screenshot 2020-05-31 at 17 26 36](https://user-images.githubusercontent.com/32545513/83357422-186bda80-a364-11ea-8293-fc1ea9b6faf5.png)

### Repairing arbitrage can improve robustness of model calibration

We verify that our repair method improves model calibration with more robust parameter 
estimates and smaller calibration error.

#### Experiment design
![Screenshot 2020-07-03 at 16 49 52](https://user-images.githubusercontent.com/32545513/86484098-54bc9d00-bd4d-11ea-8fdb-f01ec9c06b76.png)

#### Results
![Screenshot 2020-07-03 at 16 44 28](https://user-images.githubusercontent.com/32545513/86483770-a7498980-bd4c-11ea-88b6-137e6d0c4855.png)

## How do our algorithm and code perform in practice?
All of the following studies were carried out on a quadcore Intel Core i7-8650U CPU 
with 32GB RAM. All LPs are solved using the GLPK (GNU Linear Programming Kit) solver 
wrapped by the [CVXOPT](https://cvxopt.org/) Python package.

![Screenshot 2020-05-31 at 17 27 11](https://user-images.githubusercontent.com/32545513/83357427-1c97f800-a364-11ea-9f38-bf034ab40952.png)

## The formation and disappearance of executable arbitrage

We can use the our method on order book data for identifying executable arbitrage. An 
example is given below. We collect the order book data for all E-mini S&P 500 monthly 
European call options from 12:00 ET to 16:10 ET on 12th June, 2020. There is a large 
spike of number of arbitrages at around 15:52 ET, a few minutes before the close of 
the S&P 500 index market at 16:00 ET. This spike coincided with rallies in the futures 
market, while the IV maintained its relatively low level.


![Screenshot 2020-08-19 at 14 56 54](https://user-images.githubusercontent.com/32545513/90644137-4c9cbc00-e22c-11ea-9f01-e5525c1a6575.png)
The formation and disappearance of intra-day executable arbitrage opportunities in the
E-mini S\&P 500 monthly European call option market on 12th June, 2020.
## Citation

>```
>@misc{arbitragerepair2020,
>    author = {Samuel N. Cohen and Christoph Reisinger and Sheng Wang},  
>    title = {arbitragerepair},
>    publisher = {GitHub},
>    journal = {GitHub repository},
>    year = {2020},
>    howpublished = {\url{https://github.com/vicaws/arbitragerepair}},
>    note = {DOI: 10.5281/zenodo.5338299}
>}
>```

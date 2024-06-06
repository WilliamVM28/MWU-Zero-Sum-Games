# MWU-Zero-Sum-Games

## How to Use the Code

### Running the Code
1. Execute the script `MWU.py`.
2. Open the Jupyter notebook `MWU-Implementation-Matching Pennies.ipynb` to explore how the Multiplicative Weights Update (MWU) algorithm performs in the game of Matching Pennies under various learning rates.
3. Open `MWU-Increasing-Delta.ipynb` to examine the effects of increasing asymmetry with consistent learning rates on the MWU algorithm.
4. Review `Regret.ipynb` to observe how the MWU minimizes regret over time. Note that a general form is not essential for this study.

### Setting Up the Payoff Matrix
Define a payoff matrix called `cost_list` for analysis. For the standard Matching Pennies game, specify the matrix as follows:
```python
cost_list = [[[-1, 1], [1, -1]], [[1, -1], [-1, 1]]]

### Configuring the Simulation
Choose the number of iterations and the learning rate for the MWU algorithm. For example, to run 2500 iterations with a constant learning rate of 0.5, use:
Matching.MWU(2500, 0.5)


### Using Time-Decaying Learning Rates
If using a time-decaying learning rate as mentioned in the study, offset the denominator slightly to prevent the MWU from stagnating at the Nash Equilibrium. For instance, for a learning rate \epsilon=\frac{1}{t^{1/3}}
configure it as: history = Matching.MWU(2500, new_ep=lambda ep, t: 0.8 / t**(1/3))


### Note on Payoff Matrix Configuration
The implementation aligns Player 2's payoff with Player 1's position to replicate findings from Bailey and Piliouras. Altering the cost_list to reflect Player 1’s perspective does not impact the results.

### Transitioning to Asymmetric Zero-Sum Games
When moving from a symmetrical Nash equilibrium (represented by a single point) to an asymmetric setting where each player has different strategies (yet still represented by a single point), ensure that the strategies
p and q are correctly adjusted for Player 1.

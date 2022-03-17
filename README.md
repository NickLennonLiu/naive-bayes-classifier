
##  How to analyze your results
- What is the issue that you encounter?
- How do you address the issue?
  - how do you design the experiment?
  - how do you modify the algorithm?
- Does your solution work or not?
  - Does the classification performance improve?
- And finally try to explain why your solution works (or why it does not)

## Issue 1: The size of training size
How does the size of training set influence the classification performance?
> Suggestion: Sample 5%, 50%, 100% from the training set to train.

## Issue 2: Zero-probabilities
Suppose on training set, no records with xi = k, y = c

Then P_hat(y = c | x1, ..., xi = k, xn) = 0

- Why is this an issue?
- When does it likely to happen?

> Possible solution: P_hat(xi = k | y = c) = (#{y=c,xi=k} + alpha) / (#{y=c} + M*alpha)
> 
> Where M = # unique class label (?)

## Issue 3: Specific Features
Are there any specific features except for bag-of-words?
> Hints:
> - Received from ...
> - Time
> - Priority/Mailer
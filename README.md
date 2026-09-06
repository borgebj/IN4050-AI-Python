# IN4050 — Artificial Intelligence

This repository contains two programming assignments for IN4050.

## Assignment 1 — Travelling Salesperson Problem

`obliger/1` explores ways to find short round trips between European cities using a supplied distance matrix. It implements and compares:

- exhaustive search over every route permutation;
- hill climbing with routes formed by swapping pairs of cities; and
- a genetic algorithm with tournament selection, ordered crossover, mutation, and elitism.

The assignment also measures runtime, extrapolates the factorial cost of exhaustive search, runs repeated experiments, and produces route and performance plots. Start with `Assignment1.ipynb`, or run `exhaustive_search.py`, `hill_climb.py`, or `genetic.py` with `--cities <number>`.

## Assignment 2 — Classification with NumPy

`obliger/2` implements classification models from scratch on a generated two-dimensional, five-class dataset. The data is split into training, validation, and test sets, with both binary and multi-class versions of the task.

Implemented models include:

- linear regression used for binary classification;
- logistic regression;
- one-vs-rest logistic regression;
- softmax regression; and
- a one-hidden-layer multilayer perceptron (MLP) for binary and multi-class classification.

The code includes gradient-descent training, feature normalisation, loss and accuracy tracking, early stopping, evaluation metrics, and visualisations of learning curves and decision regions. Start with `in4050_assignment2.ipynb`, or run the individual model scripts such as `logreg.py`, `softmax.py`, or `mlp.py`.

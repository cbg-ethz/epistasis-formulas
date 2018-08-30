# Description:

This Python code computes higher-order interactions such as 2-way, 3-way,…, n-way interaction coordinates and some circuits in the n-locus case taking as input 2^n experimental measurements.
This code can also be used to compute the intervals in which the above interactions are contained if the starting measurements are given as intervals. 

For instance, if w_{00} belongs to the interval [x_1,x_2],
w_{11} belongs to [x_3,x_4], w_{10} to [x_5,x_6]$ and
w_{01} to[x_7,x_8], for some real numbers, then our code can be used to deduce the values of a=x_1+x_4-x_6-x_8 and b=x_2+x_4-x_5-x_7 such that the 2-way interaction

w_{00}+w_{11}-w_{01}-w_{10}

is contained in [a,b].


# Instructions:

To compute all interactions insert manually experimental data in the epistasis.py file and run. Currently, the program is set to work in the 5-locus case, but it can be altered easily to analyze interactions in the n-locus case.


# Observation: 

The code builds on:
* Python sequences. In particular tuple/list/dict comprehensions, generators and 'yield   from' idioms have been extensively applied to delegate a lot of data manipulation to Python.
* The package NumPy. Tensors and slicing determine how the fitness projections are created. Only simple mathematical operators are used (inner product, matrix product).
* Elementary interval arithmetics.

# References:

* A.L Gould et al, [High-dimensional microbiome interactions shape host fitness
](https://www.biorxiv.org/content/early/2018/06/01/232959.1)
* N. Beerenkinkel et al, [Epistasis and shapes of fitness landscapes](http://www3.stat.sinica.edu.tw/statistica/oldpdf/A17n43.pdf)

# Authors:

Korasidis Nikolaos <nkorasid@student.ethz.ch>, Lamberti Lisa <lmlamberti@bsse.ethz.ch>

## Initial release‎: August 2018
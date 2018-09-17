# Description:

This Python code computes higher-order interactions such as 2-way, 3-way,…, n-way interaction coordinates and some circuits in the n-species (or loci in genetics) case taking as input 2^n experimental measurements.
These interactions are described in A.L Gould et al's work on [High-dimensional microbiome interactions shape host fitness                                                                                                                        ](https://www.biorxiv.org/content/early/2018/06/01/232959.1).



# Instructions:

* To compute the above mentioned interactions insert manually experimental data and standard error measurements in the epistasis.py file and run. Currently, the program is set to work in the 5-species case, but it can be altered easily to analyze interactions in the n-species case. 

* This code can also compute the intervals in which the results of the above interactions are contained by knowing the intervals in which the starting measurments are contained: for instance, average measurement +/- SE. 

    * Example: if each measurment w_{00},w_{11}, w_{10} and w_{01} belongs to the intervals [w_{00}-SE,w_{00}+SE], [w_{11}-SE,w_{11}+SE], [w_{10}-SE,w_{10}+SE] and resp. [w_{01}-SE,w_{01}+SE], then our code coutputs the bounds of the interval [a,b] containing for instance the 2-way interaction

        w_{00}+w_{11}-w_{01}-w_{10} . 

Similarly, the code computes all other intervals for the various higher-order interactions described above.



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

_Initial release: August 2018_

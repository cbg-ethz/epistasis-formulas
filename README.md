# Description:

This Python code computes higher-order interactions such as 2-way, 3-way,…, n-way interaction coordinates and some circuits in the n-species (or loci in genetics) case taking as input 2^n experimental measurements.
These interactions are described in A.L Gould et al's work on [High-dimensional microbiome interactions shape host fitness                                                                                                                        ](https://www.biorxiv.org/content/early/2018/06/01/232959.1).



# Instructions:

* To compute the above mentioned interactions insert manually experimental data and standard error measurements in the epistasis.py file and run. Currently, the program is set to work in the 5-species case, but it can be altered easily to analyze interactions in the n-species case. 

* This code can also compute the intervals in which the results of the above interactions are contained by knowing the intervals in which the starting measurments are contained: for instance, average measurement +/- SE. 

    * Example: if each measurment w_X belongs to the interval [w_X-SE,w_X+SE], then our code coutputs the bounds of the interval [a,b] containing for instance the 2-way interaction

        w_00000+w_00011-w_00001-w_00010 . 

        Similarly, the code computes all other intervals for the various higher-order interactions described above.


# Dependencies:

* [NumPy](http://www.numpy.org/): Python interface to store and operate on numerical arrays
* [Itertools](https://docs.python.org/2/library/itertools.html): Python module used to creat iterators

The epistasis.py file uses the code contained in the following files:

* Circuits.py: generates all 2 and 3 dimensional circuit interactions inside n-dimensional hypercubes.
* Fourier.py: module written to compute interactions coordinates as described in equation 8 in N. Beerenkinkel et al, [Epistasis and shapes of fitness landscapes](http://www3.stat.sinica.edu.tw/statistica/oldpdf/A17n43.pdf)
* Slicing.py: generates all possible contexts for circuit and interaction coordinates.
* Utlis.py: minor module with additional codes used in the above files.

# Observation: 

The code builds on:  Python sequences. In particular tuple/list/dict comprehensions, generators and 'yield   from' idioms have been extensively applied to delegate a lot of data manipulation to Python. Only simple mathematical operators are used (inner product, matrix product). Elementary interval arithmetics.



# References:

* A.L Gould et al, [High-dimensional microbiome interactions shape host fitness
](https://www.biorxiv.org/content/early/2018/06/01/232959.1)
* N. Beerenkinkel et al, [Epistasis and shapes of fitness landscapes](http://www3.stat.sinica.edu.tw/statistica/oldpdf/A17n43.pdf)

# Authors:

Korasidis Nikolaos <nkorasid@student.ethz.ch>, Lamberti Lisa <lmlamberti@bsse.ethz.ch>

_Initial release: August 2018_

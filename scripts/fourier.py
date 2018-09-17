"""
We are studying the interaction of n species, for each of which there are 2
options: absent from a setup (case 0) or present in a setup (case 1). There are
thus 2^n possible setups, which we can think of as the vertices of a n-cube.

Suppose W is the fitness matrix which aggregates the fitness of each setup:

    W = [w_0, w_1, ..., w_{2^n - 1}]^T
    
For an i such that 0 <= i < 2^n, we can expand i as a binary string,
i = b_0 b_1 ... b_{n - 1}, where b_k can be either 0 or 1. We thus obtain a
mapping between {0,1}^n and the numbers [0...2^{n-1}]. We will assume that the
indices of w_i in W respect this mapping.

----

The equation in page 8 of [1] defines the interaction coordinates U that
arise from the n-cube geometry of interactions. For the construction of the
mathematical framework, we will consider all 2^n such coordinates, so that:

    U = [u_0, u_1, ..., u_{2^n - 1}]^T

similar to W. We note that the indices of u_j can be interpreted as n-bit
strings, j = c_0 c_1 ... c_{n - 1}. Only the u_j where the expansion of j
contains at least two 1-bits are meaningful, as they denote epistatic
interactions between ---at least two--- species. Thus, out of 2^n interaction
coordinates, there are n + 1 meaningless points, denoting interaction of no
species (u_0) or "interaction" of a single species (u_{2^k} for k in [0..n-1]).

The equation from which we obtain U from W is the discrete Fourier transform
over {0, 1}^n:

    2^{n - 1} u_{c_0 c_1 ... c_{n-1}}
        = sum_{b_0 = 0}^{1} sum_{b_0 = 0}^{1} ... sum_{b_{n-1} = 0}^{1}
            (-1)^{b_0 c_0 + b_1 c_1 + ... + b_{n - 1} c_{n - 1}}
            w_{b_0 b_1 ... b_{n - 1}}

The construction has the following properties:
    1. Each u_i is a linear combination of ---all--- w_j.
    2. Each w_j appears with either (+1) or (-1) sign in any such combination.
       The sign is specified by calculating 
       (-1)^{b_0 c_0 + b_1 c_1 + ... + b_{n - 1} c_{n - 1}}

Because of property 1, we can express the relation between U and W by a linear
mapping, using an auxiliary matrix F:

    2^{n - 1} U = F W

F is a dense 2^n by 2^n matrix, which contains only (+1) and (-1) elements. A
naive way to construct F is to calculate each element separately, which costs
O(n 2^{2n}) in total. We have implemented that way for testing purposes.

A faster and more elegant way to compute F takes advantage of symmetries
inherent in the Fourier transform. Let us split F in 4 square matrices of the
same geometry:

        |---|
        |A|B| 
    F = |---|
        |C|D| 
        |---|
    
Suppose that we have already computed the 2^{n-1} by 2^{n-1} upper left part
that is, A. This involves all cases where b_{n - 1} = c_{n - 1} = 0. Now, the
parts B and C should be indentical to A because they involve calculations where
exactly one of b_{n - 1} and c_{n - 1} are 1, and thus the element computed is
the same as the respective element of A. To the contrary, when
b_{n - 1} = c{n - 1} = 1, then the element computed is the opposite of the
corresponding one in A, thus D = -A.

This can be summarized in the following recursive construction: Let F_{n,n} be
the (unscaled) Fourier matrix that maps U_{n,0} to W_{n,0}, where the subscript
n denotes that there exist 2^n elements in each relevant dimension. Then

    For n = 0: F_{0, 0} = [1]
    For n > 0:
                   |----------------------------|
                   |F_{n-1, n-1} |  F_{n-1, n-1}| 
        F_{n, n} = |----------------------------|
                   |F_{n-1, n-1} |- F_{n-1, n-1}| 
                   |----------------------------|

In this way, F can be computed very efficiently, in time linear to the number
of elements. If many cores are available and n is sufficiently large, the
calculation can be parallelized.

---

Once F has been computed, one can remove calculations corresponding to
meaningless interaction coordinates, by deleting rows 0 and 2^{0, 1, ... n - 1}
from F and U.

[1] Epistasis and Shapes of Fitness Landscapes
"""

import numpy as np


def _inner_product(i, j):
    return sum(1 for s in np.binary_repr(i & j) if s == '1')


def generate_full_fourier_matrix_iter(n):
    size = 2**n
    f = np.empty((size, size), dtype=np.int16)
    for i in range(size):
        for j in range(size):
            if _inner_product(i, j) % 2 == 0:
                f[i, j] = 1
            else:
                f[i, j] = -1
    return f


def generate_full_fourier_matrix_rec(n):
    full_size = 2**n
    f = np.empty((full_size, full_size), dtype=np.int16)
    f[0, 0] = 1
    q_size = 1
    while q_size < full_size:
        next_q_size = 2 * q_size
        f[0:q_size, q_size:next_q_size] = f[:q_size, :q_size] # up right
        f[q_size:next_q_size, 0:q_size] = f[:q_size, :q_size] # low left
        f[q_size:next_q_size, q_size:next_q_size] = - f[:q_size, :q_size] # low right
        q_size = next_q_size
    return f


generate_full_fourier_matrix = generate_full_fourier_matrix_rec


def generate_singleton_indices(n):
    return [0] + [2**i for i in range(n)]


def generate_fourier_matrix(n):
    f = generate_full_fourier_matrix(n)
    del_indices = generate_singleton_indices(n)
    return np.delete(f, del_indices, axis=0)


def print_fourier(n):
    print('Full Fourier matrix on', 4, 'species')
    fn = generate_full_fourier_matrix(4)
    print(fn)
    print()

    print('Singleton-free Fourier matrix on', n, 'species')
    fsn = generate_fourier_matrix(n)
    print(fsn)
    print()


if __name__ == '__main__':
    print_fourier(2)
    print_fourier(3)

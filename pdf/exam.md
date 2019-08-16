# Basic Tools for CIL
## Matrix-vector basis
* Symmetric matrix: $A = A^T$
* Orthogonal matrix: $A^{-1} = A^T$ i.e. $A^T A = A^{-1}A = I$ and $det(I) = 1$
* Transposed matrix: $(A^T)^{-1} = (A^{-1})^{T}$
* Inner Prod: $\langle x,y \rangle =\| x\|_2 \cdot \| y\| \cdot cos(\theta) $, if $y$ is a unit vector then the inner product projects $x$ onto $y$.
    * $\langle x,y \rangle = x^T y = \sum_i^N x_i y_i$
    * $\langle x+y, x+y \rangle = \langle x, x \rangle + \langle y, y \rangle + 2 \langle x, y \rangle$
    * $\langle x-y, x-y \rangle = \langle x, x \rangle + \langle y, y \rangle - 2 \langle x, y \rangle$
    * $\langle x, y+z \rangle = \langle x,y \rangle + \langle x,z \rangle$
    * $\langle x+z, y \rangle = \langle x,y \rangle + \langle z,y \rangle$
* Outer product: $X = u v^T$ and $X_{i,j} = u_i v_j$
* Orthonormal basis: Set of vectors in an $N$ dimensional space for which the basis vectors fulfill:
    * Unit vectors (length = 1)
    * Together the vectors have an inner product of zero, i.e. the vectors are orthogonal
    * Ex for basis for $R^3$: $\{e_1, e_2, e_3\} = \{(0,0,1),(0,1,0), (1,0,0)\}$
        * Being a basis for $R^3$ means that every vector $v \in R^3$ can be written as a sum of the 3 vectors scaled: $v = e_1 \cdot x +  e_2 \cdot y +  e_3 \cdot z$
* Gram-Schmidt orthonormal basis algorithm: Finds an orthonormal basis $u=u_1 ... u_k$ given linearly independent set $v = v_1 ... v_k$ where:
    * $u_1 = v_1$
    * $u_2 = v_2 - \frac{\langle v_2, u_1 \rangle}{\langle u_1, u_1 \rangle}$
    * $u_3 = v_3 - \frac{\langle v_3, u_1 \rangle}{\langle u_1, u_1 \rangle} - \frac{\langle v_3, u_2 \rangle}{\langle u_2, u_2 \rangle} $
    * ...
    * $u_k = v_k - \sum_i^{k-1} \frac{\langle v_k, u_i \rangle}{\langle u_i, u_i \rangle}$
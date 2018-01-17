cimport cython
cimport numpy as np
import numpy as np
import scipy.sparse

from libc.stdlib cimport malloc, free
from cpython.string cimport PyString_AsString
cimport random_walk


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def build_random_walk_corpus(adj_matrix,
                             np.ndarray[uint8_t, ndim=1, mode='c'] vertex_types,
                             uint8_t n_types,
                             list vertex_names,
                             char *output_file, int path_length,
                             int num_per_vertex, float alpha,
                             int use_meta_path,
                             int n_jobs):
    cdef uint32_t n
    cdef uint32_t i
    cdef uint32_t j
    cdef uint32_t vcount
    cdef uint32_t degree
    cdef StaticGraph g
    cdef char **names
    cdef uint32_t *neighbors

    cdef char **path
    
    vcount = adj_matrix.shape[0]
    names = <char **>malloc(vcount * cython.sizeof(char_pointer))
    if names is NULL:
        raise MemoryError()
    for i in range(vcount):
        names[i] = vertex_names[i]
    types = <uint8_t *>malloc(vcount * cython.sizeof(uint8_t))
    if types is NULL:
        raise MemoryError()
    for i in range(vcount):
        types[i] = vertex_types[i]
    GraphInit(&g, vcount, names, types, n_types)
    free(names)
    free(types)

    indices = adj_matrix.indices
    indptr = adj_matrix.indptr
    n = 0
    for i in range(vcount):
        degree = indptr[i + 1] - indptr[i]
        neighbors = <uint32_t *>malloc(degree * cython.sizeof(uint32_t))
        if neighbors is NULL:
            raise MemoryError()
        for j in range(degree):
            neighbors[j] = indices[n]
            n += 1
        SetNeighbors(&g, i, neighbors, degree)
        free(neighbors)
    SetOutputFile(output_file)
    with nogil:
        GenerateRandomWalk(&g, path_length, num_per_vertex,
                           alpha, use_meta_path, n_jobs)
    GraphDestroy(&g)


    

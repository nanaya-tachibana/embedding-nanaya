cimport cython
from libc.stdlib cimport malloc, free
from cpython.string cimport PyString_AsString
cimport random_walk


def build_random_walk_corpus(list adjlist,
                             list vertex_types, int n_types,
                             list vertex_names,
                             char *output_file, int path_length,
                             int num_per_vertex, float alpha,
                             int use_meta_path,
                             int n_jobs):
    cdef long i
    cdef long j
    cdef long vcount
    cdef long degree
    cdef StaticGraph g
    cdef char **names
    cdef long *neighbors

    cdef char **path
    

    vcount = len(adjlist)
    names = <char **>malloc(vcount * cython.sizeof(char_pointer))
    if names is NULL:
        raise MemoryError()
    for i in range(vcount):
        names[i] = vertex_names[i]
    types = <int *>malloc(vcount * cython.sizeof(int))
    GraphInit(&g, vcount, names, types, n_types)

    for i in range(vcount):
        degree = len(adjlist[i])
        neighbors = <long *>malloc(degree * cython.sizeof(long))
        if neighbors is NULL:
            raise MemoryError()
        for j in range(degree):
            neighbors[j] = adjlist[i][j]
        SetNeighbors(&g, i, neighbors, degree)
        free(neighbors)
    SetOutputFile(output_file)
    with nogil:
        GenerateRandomWalk(&g, path_length, num_per_vertex,
                           alpha, use_meta_path, n_jobs)
    GraphDestroy(&g)
    free(names)


    

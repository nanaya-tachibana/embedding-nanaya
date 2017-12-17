ctypedef char * char_pointer

cdef extern from 'static_graph.h':
    cdef struct Vertex:
        long *neighbors
        long degree
        char *name
    cdef struct StaticGraph:
        Vertex *vertices
        long vcount

    void GraphInit(StaticGraph *g, long vcount, char **names)
    void GraphDestroy(StaticGraph *g)
    void SetNeighbors(StaticGraph *g, long vertex, long *neighbors, long degree)
    void SetOutputFile(char *filename);
    void GenerateRandomWalk(StaticGraph *g,
			    int path_length,
			    int num_per_vertex,
			    float alpha,
			    int n_jobs) nogil

        

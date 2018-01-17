from libc.stdint cimport uint32_t, uint8_t

ctypedef char * char_pointer

cdef extern from 'static_graph.h':
    cdef struct Vertex:
        uint32_t *neighbors
        uint32_t *type_begin_idx
        uint32_t *type_end_idx
        uint32_t degree
        uint8_t type
        char *name
    cdef struct StaticGraph:
        Vertex *vertices
        uint32_t vcount
        uint8_t n_types

    void GraphInit(StaticGraph *g, uint32_t vcount, char **names,
                   uint8_t *types, uint8_t n_types)
    void GraphDestroy(StaticGraph *g)
    void SetNeighbors(StaticGraph *g, uint32_t vertex, uint32_t *neighbors, uint32_t degree)
    void SetOutputFile(char *filename);
    void GenerateRandomWalk(StaticGraph *g,
			    int path_length,
			    int num_per_vertex,
			    float alpha,
                            int use_meta_path,
			    int n_jobs) nogil

        

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include "random.h"

#define MAX_STRING 100

char output_file[MAX_STRING];
clock_t start;
long actual_node_count;
long total_node_count;


typedef struct Vertex {
  long *neighbors;  // neighbors are sorted by type, e.g. (t0, t0, t1, t1, t2, t3)
  long *type_begin_idx;  // the indice of the first neighbor of one particular type
  long *type_end_idx;
  long degree;
  int type;
  char *name;
} Vertex;


typedef struct StaticGraph {
  Vertex *vertices;
  int n_types;
  long vcount;
} StaticGraph;


void GraphInit(StaticGraph *g, long vcount, char **names,
	       int *types, int n_types);
void GraphDestroy(StaticGraph *g);
void SetNeighbors(StaticGraph *g, long vertex, long *neighbors, long degree);
void GraphPrint(StaticGraph *g);
int RandomPath(StaticGraph *g,
	       char **path,
	       long start,
	       int max_length,
	       float alpha,
	       int use_meta_path);
void GenerateRandomWalkThread(void *_g, long idx, int tid);
void GenerateRandomWalk(StaticGraph *g,
			int path_length,
			int num_per_vertex,
			float alpha,
			int use_meta_path,
			int n_jobs);
void SetOutputFile(char *filename);
long RandomInteger(long low, long high);
double RandomFloat(double low, double high);


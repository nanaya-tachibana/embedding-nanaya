#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include "random.h"

#define MAX_STRING 100

char output_file[MAX_STRING];
clock_t start;
uint64_t actual_node_count;
uint64_t total_node_count;


typedef struct Vertex {
  uint32_t *neighbors;  // neighbors are sorted by type, e.g. (t0, t0, t1, t1, t2, t3)
  uint32_t *type_begin_idx;  // the indice of the first neighbor of one particular type
  uint32_t *type_end_idx;
  uint32_t degree;
  uint8_t type;
  char *name;
} Vertex;


typedef struct StaticGraph {
  Vertex *vertices;
  uint8_t n_types;
  uint32_t vcount;
} StaticGraph;


void GraphInit(StaticGraph *g, uint32_t vcount, char **names,
	       uint8_t *types, uint8_t n_types);
void GraphDestroy(StaticGraph *g);
void SetNeighbors(StaticGraph *g, uint32_t vertex, uint32_t *neighbors, uint32_t degree);
void GraphPrint(StaticGraph *g);
int RandomPath(StaticGraph *g,
	       char **path,
	       uint32_t start,
	       int max_length,
	       float alpha,
	       int use_meta_path);
void GenerateRandomWalkThread(void *_g, uint64_t idx, int tid);
void GenerateRandomWalk(StaticGraph *g,
			int path_length,
			int num_per_vertex,
			float alpha,
			int use_meta_path,
			int n_jobs);
void SetOutputFile(char *filename);
uint32_t RandomInteger(uint32_t low, uint32_t high);
double RandomFloat(double low, double high);


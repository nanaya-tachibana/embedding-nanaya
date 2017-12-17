#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include "random.h"

#define MAX_STRING 100

char output_file[MAX_STRING];


typedef struct Vertex {
  long *neighbors;
  long degree;
  char *name;
} Vertex;


typedef struct StaticGraph {
  Vertex *vertices;
  long vcount;
} StaticGraph;


void GraphInit(StaticGraph *g, long vcount, char **names);
void GraphDestroy(StaticGraph *g);
void SetNeighbors(StaticGraph *g, long vertex, long *neighbors, long degree);
void GraphPrint(StaticGraph *g);
int RandomPath(StaticGraph *g,
	       char **path,
	       long start,
	       int max_length,
	       float alpha);
void GenerateRandomWalkThread(void *_g, long i, int tid);
void GenerateRandomWalk(StaticGraph *g,
			int path_length,
			int num_per_vertex,
			float alpha,
			int n_jobs);
void SetOutputFile(char *filename);
long RandomInteger(long low, long high);
double RandomFloat(double low, double high);


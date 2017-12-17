#include "static_graph.h"


typedef struct {
  FILE **files;
  StaticGraph *graph;
  int length;
  float alpha;
} global_t;


void GraphInit(StaticGraph *g, long vcount, char **names) {
  long name_len;
  long i;
  g->vcount = vcount;
  g->vertices = (Vertex *)malloc(vcount * sizeof(Vertex));
  if (g->vertices == NULL) {
    perror("(ERROR) Memory allocation failed\n");
    exit(-1);
  }

  for (i = 0; i < vcount; i++) {
    g->vertices[i].neighbors = NULL;

    name_len = strlen(names[i]) + 1;
    if (name_len > MAX_STRING)
      name_len = MAX_STRING;
    g->vertices[i].name = (char *)malloc(name_len * sizeof(char));
    if (g->vertices[i].name == NULL) {
      perror("(ERROR) Memory allocation failed\n");
      exit(-1);
    }
    strcpy(g->vertices[i].name, names[i]);

    g->vertices[i].degree = 0;
  }
  strcpy(output_file, "./random_walk");
}


void GraphDestroy(StaticGraph *g) {
  long i;
  for (i = 0; i < g->vcount; i++) {
    if (g->vertices[i].neighbors != NULL)
      free(g->vertices[i].neighbors);
  }
  if (g->vertices != NULL)
    free(g->vertices);
}


void SetNeighbors(StaticGraph *g, long vertex, long *neighbors, long degree) {
  long i;
  if (vertex >= g->vcount) {
    fprintf(stderr, "(ERROR) Vertex %ld is out of index\n", vertex);
    exit(-1);
  }

  Vertex *v = &g->vertices[vertex];
  v->neighbors = (long *)malloc(degree * sizeof(long));
  if (v->neighbors == NULL) {
    perror("(ERROR) Memory allocation failed\n");
    exit(-1);
  }

  for (i = 0; i < degree; i++) {
    if (neighbors[i] >= g->vcount) {
      fprintf(stderr, "(WARNING) Vertex %ld is out of index(Removed)\n", neighbors[i]);
      continue;
    }
    v->neighbors[v->degree] = neighbors[i];
    v->degree++;
  }
}


void GraphPrint(StaticGraph *g) {
  Vertex v;
  long i, j;
  printf("\n");
  for (i = 0; i < g->vcount; i++) {
    v = g->vertices[i];
    printf("vertex %ld: [", i);
    if (v.neighbors != NULL)
      for (j = 0; j < v.degree; j++)
	printf("%ld, ", v.neighbors[j]);
    printf("]\n");
  }
}


int RandomPath(StaticGraph *g, char **path, long start, int max_length, float alpha) {
  int length;
  long i;
  long neighbor;
  Vertex current;

  if (start >= g->vcount) {
    fprintf(stderr, "(ERROR) Vertex %ld is out of index\n", start);
    exit(-1);
  }

  length = 0;
  strcpy(path[length++], g->vertices[start].name);
  
  current = g->vertices[start];
  while (length < max_length) {
    if (RandomFloat(0, 1) >= alpha) {  // include alpha = 0
      if (current.degree == 0)
	return length;
      else {
	i = RandomInteger(0, current.degree);
	neighbor = current.neighbors[i];
	strcpy(path[length++], g->vertices[neighbor].name);
      }
    } else  // restart from the starting point
      strcpy(path[length++], g->vertices[start].name);
    current = g->vertices[neighbor];
  }
  return length;
}


void SetOutputFile(char *filename) {
  strcpy(output_file, filename);
}


void GenerateRandomWalkThread(void *_g, long index, int tid) {
  char **path;
  int actual_length;
  int i;

  global_t *g = (global_t *)_g;
  index = index % g->graph->vcount;  // start point index

  path = (char **)malloc(g->length * sizeof(char *));
  if (path == NULL) {
    perror("(ERROR) Memory allocation failed\n");
    exit(-1);
  }
  for (i = 0; i < g->length; i++) {
    path[i] = (char *)malloc(sizeof(char) * MAX_STRING);
    if (path[i] == NULL) {
      perror("(ERROR) Memory allocation failed\n");
      exit(-1);
    }
  }

  actual_length = RandomPath(g->graph, path, index, g->length, g->alpha);
  for (i = 0; i < actual_length; i++) {
    if (i != 0)
      fprintf(g->files[tid], " ");
    fprintf(g->files[tid], "%s", path[i]);
  }
  fprintf(g->files[tid], "\n");

  for (i = 0; i < g->length; i++)
    free(path[i]);
  free(path);
}


void GenerateRandomWalk(StaticGraph *g, int path_length, int num_per_vertex,
			float alpha, int n_jobs) {
  global_t global;
  char temp[MAX_STRING];
  FILE **files;
  int i;

  global.graph = g;
  global.length = path_length;
  global.alpha = alpha;

  files = (FILE **)malloc(n_jobs * sizeof(FILE *));
  if (files == NULL) {
    perror("(ERROR) Memory allocation failed\n");
    exit(-1);
  }

  for (i = 0; i < n_jobs; i++) {
    sprintf(temp, "%s_%d.txt", output_file, i);
    files[i] = fopen(temp , "w");
    if (files[i] == NULL) {
      fprintf(stderr, "(ERROR) Cannot open file %s: %s\n", temp, strerror(errno));
      exit(-1);
    }
  }
  global.files = files;
    
  kt_for(n_jobs, &GenerateRandomWalkThread, &global, g->vcount * num_per_vertex);

  for (i = 0; i < n_jobs; i++)
    fclose(files[i]);
  free(files);
}

// Generate a random integer in [low, high)
// 
long RandomInteger(long low, long high) {
  return (long)(ran_uniform() * (high - low)) + low;
}

// Generate a random float in [low, high]
// 
double RandomFloat(double low, double high) {
  return ran_uniform() * (high - low) + low;
}


int main() {
  StaticGraph g;
  char *names[10] = {"a\0", "b\0", "c\0",
		   "e\0", "f\0", "g\0",
		   "h\0", "i\0", "j\0",
		  "k\0"};
  GraphInit(&g, 10, names);

  long neighbors1[] = {11, 7, 8, 2, 1, 4};
  SetNeighbors(&g, 1, neighbors1, 6);
  GraphPrint(&g);

  long neighbors2[] = {1, 8, 0};
  SetNeighbors(&g, 2, neighbors2, 3);
  GraphPrint(&g);

  long neighbors4[] = {0, 1, 3, 8};
  SetNeighbors(&g, 4, neighbors4, 4);
  GraphPrint(&g);

  long neighbors8[] = {4, 1, 7, 2, 1};
  SetNeighbors(&g, 8, neighbors8, 5);
  GraphPrint(&g);

  GenerateRandomWalk(&g, 5, 10, 0, 4);
  GraphDestroy(&g);

  long x[10];
  long i;
  for (i = 0; i < 10; i++)
    x[i] = RandomInteger(0, 10);
  for (i = 0; i < 10; i++)
    printf("%ld\n", x[i]);
}


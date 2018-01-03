#include "static_graph.h"


typedef struct {
  FILE **files;
  StaticGraph *graph;
  int length;
  float alpha;
  int meta;
} global_t;


void GraphInit(StaticGraph *g, long vcount,
	       char **names, int *types, int n_types) {
  long name_len;
  long i;
  g->vcount = vcount;
  g->n_types = n_types;
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

    g->vertices[i].type = types[i];
    g->vertices[i].degree = 0;
  }
  strcpy(output_file, "./random_walk");
}


void GraphDestroy(StaticGraph *g) {
  long i;
  for (i = 0; i < g->vcount; i++) {
    if (g->vertices[i].neighbors != NULL)
      free(g->vertices[i].neighbors);
    if (g->vertices[i].name != NULL)
      free(g->vertices[i].name);
    if (g->vertices[i].type_begin_idx != NULL)
      free(g->vertices[i].type_begin_idx);
    if (g->vertices[i].type_end_idx != NULL)
      free(g->vertices[i].type_end_idx);
  }
  if (g->vertices != NULL)
    free(g->vertices);
}


void SetNeighbors(StaticGraph *g, long vertex, long *neighbors, long degree) {
  long i;
  int type;
  int is_first;
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
  v->type_begin_idx = (long *)malloc(g->n_types * sizeof(long));
  if (v->type_begin_idx == NULL) {
    perror("(ERROR) Memory allocation failed\n");
    exit(-1);
  }
  v->type_end_idx = (long *)malloc(g->n_types * sizeof(long));
  if (v->type_end_idx == NULL) {
    perror("(ERROR) Memory allocation failed\n");
    exit(-1);
  }

  for (type = 0; type < g->n_types; type++) {
    is_first = 1;
    for (i = 0; i < degree; i++) {
      if (neighbors[i] >= g->vcount) {
	fprintf(stderr, "(WARNING) Vertex %ld is out of index(Removed)\n", neighbors[i]);
	continue;
      }
      
      if (g->vertices[neighbors[i]].type == type) {
	v->neighbors[v->degree++] = neighbors[i];
	
	if (is_first) {
	  is_first = 0;
	  v->type_begin_idx[type] = v->degree - 1;
	}
      }
    }
    v->type_end_idx[type] = v->degree;
    if (is_first)
      v->type_begin_idx[type] = v->degree;
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


int RandomPath(StaticGraph *g, char **path, long start,
	       int max_length, float alpha, int use_meta_path) {
  int length;
  long i;
  long neighbor;
  Vertex current;
  int next_type;
  int sign;
  long bidx, eidx;
  
  if (start >= g->vcount) {
    fprintf(stderr, "(ERROR) Vertex %ld is out of index\n", start);
    exit(-1);
  }

  if (g->n_types == 1)
    use_meta_path = 0;
  
  length = 0;
  strcpy(path[length++], g->vertices[start].name);

  current = g->vertices[start];
  sign = 1;
  if (use_meta_path && current.type == g->n_types - 1)
    sign = -1;

  while (length < max_length) {
    if (RandomFloat(0, 1) >= alpha) {  // include alpha = 0
      if (current.degree == 0)
	break;
      else {
	if (use_meta_path) { // meta path
          next_type = current.type + sign;
	  bidx = current.type_begin_idx[next_type];
	  eidx = current.type_end_idx[next_type];
	  if (eidx == bidx)
	    break;
	  i = RandomInteger(bidx, eidx);
          if (next_type == 0 || next_type == g->n_types - 1)
            sign = -sign;
	}
	else
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


void GenerateRandomWalkThread(void *_g, long idx, int tid) {
  char **path;
  int actual_length;
  int i;
  clock_t now;

  global_t *g = (global_t *)_g;
  idx = idx % g->graph->vcount;  // start point index

  now = clock();
  printf("%cProgress: %.2f%%  Words/thread/sec: %.2fk  ", 13,
         actual_node_count / (float)(total_node_count) * 100,
         actual_node_count / ((float)(now - start + 1) / (float)CLOCKS_PER_SEC * 1000));
  fflush(stdout);
  actual_node_count++;

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

  actual_length = RandomPath(g->graph, path, idx, g->length, g->alpha, g->meta);
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
			float alpha, int use_meta_path, int n_jobs) {
  global_t global;
  char temp[MAX_STRING];
  FILE **files;
  int i;

  global.graph = g;
  global.length = path_length;
  global.alpha = alpha;
  global.meta = use_meta_path;

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

  total_node_count = g->vcount * num_per_vertex;
  start = clock();
  kt_for(n_jobs, &GenerateRandomWalkThread, &global, total_node_count);

  for (i = 0; i < n_jobs; i++)
    fclose(files[i]);
  free(files);
}

// Generate a random integer in [low, high)
// 
long RandomInteger(long low, long high) {
  return (long)(ran_uniform() * (high - low)) + low;
}

// Generate a random float in [low, high)
// 
double RandomFloat(double low, double high) {
  return ran_uniform() * (high - low) + low;
}


int main() {
  /* StaticGraph g; */
  /* char *names[10] = {"a\0", "b\0", "c\0", */
  /* 		   "e\0", "f\0", "g\0", */
  /* 		   "h\0", "i\0", "j\0", */
  /* 		  "k\0"}; */
  /* GraphInit(&g, 10, names); */

  /* long neighbors1[] = {11, 7, 8, 2, 1, 4}; */
  /* SetNeighbors(&g, 1, neighbors1, 6); */
  /* GraphPrint(&g); */

  /* long neighbors2[] = {1, 8, 0}; */
  /* SetNeighbors(&g, 2, neighbors2, 3); */
  /* GraphPrint(&g); */

  /* long neighbors4[] = {0, 1, 3, 8}; */
  /* SetNeighbors(&g, 4, neighbors4, 4); */
  /* GraphPrint(&g); */

  /* long neighbors8[] = {4, 1, 7, 2, 1}; */
  /* SetNeighbors(&g, 8, neighbors8, 5); */
  /* GraphPrint(&g); */

  /* GenerateRandomWalk(&g, 5, 10, 0, 4); */
  /* GraphDestroy(&g); */

  /* long x[10]; */
  /* long i; */
  /* for (i = 0; i < 10; i++) */
  /*   x[i] = RandomInteger(0, 10); */
  /* for (i = 0; i < 10; i++) */
  /*   printf("%ld\n", x[i]); */
}


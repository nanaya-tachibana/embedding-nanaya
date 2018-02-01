#include "static_graph.h"


typedef struct {
  int tid;
  int num_threads;
  StaticGraph *graph;
  int length;
  int num_per_vertex;
  float alpha;
  uint8_t *meta_path;
  uint8_t meta_path_length;
} Params;


void GraphInit(StaticGraph *g, uint32_t vcount,
	       char **names, uint8_t *types, uint8_t n_types) {
  int name_len;
  uint32_t i;
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
  uint32_t i;
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


void SetNeighbors(StaticGraph *g, uint32_t vertex,
		  uint32_t *neighbors, uint32_t degree) {
  uint32_t i;
  int type;
  int is_first;
  if (vertex >= g->vcount) {
    fprintf(stderr, "(ERROR) Vertex %u is out of index\n", vertex);
    exit(-1);
  }

  Vertex *v = &g->vertices[vertex];
  v->neighbors = (uint32_t *)malloc(degree * sizeof(uint32_t));
  if (v->neighbors == NULL) {
    perror("(ERROR) Memory allocation failed\n");
    exit(-1);
  }
  v->type_begin_idx = (uint32_t *)malloc(g->n_types * sizeof(uint32_t));
  if (v->type_begin_idx == NULL) {
    perror("(ERROR) Memory allocation failed\n");
    exit(-1);
  }
  v->type_end_idx = (uint32_t *)malloc(g->n_types * sizeof(uint32_t));
  if (v->type_end_idx == NULL) {
    perror("(ERROR) Memory allocation failed\n");
    exit(-1);
  }

  for (type = 0; type < g->n_types; type++) {
    is_first = 1;
    for (i = 0; i < degree; i++) {
      if (neighbors[i] >= g->vcount) {
	fprintf(stderr, "(WARNING) Vertex %u is out of index(Removed)\n", neighbors[i]);
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
  uint32_t i, j;
  printf("\n");
  for (i = 0; i < g->vcount; i++) {
    v = g->vertices[i];
    printf("vertex %u: [", i);
    if (v.neighbors != NULL)
      for (j = 0; j < v.degree; j++)
	printf("%u, ", v.neighbors[j]);
    printf("]\n");
  }
}


int RandomPath(StaticGraph *g, char **path, uint32_t start,
	       int max_length, float alpha, uint8_t meta_path_length, uint8_t *meta_path) {
  int length;
  uint32_t i;
  uint32_t neighbor;
  Vertex current;
  uint8_t next_type;
  uint8_t current_type_idx;
  long bidx, eidx;
  
  if (start >= g->vcount) {
    fprintf(stderr, "(ERROR) Vertex %u is out of index\n", start);
    exit(-1);
  }

  if (g->n_types == 1)
    meta_path = NULL;
  
  length = 0;
  strcpy(path[length++], g->vertices[start].name);

  current = g->vertices[start];
  if (meta_path != NULL)
    for (current_type_idx = 0; current_type_idx < meta_path_length; current_type_idx++)
      if (meta_path[current_type_idx] == current.type)
	break;

  while (length < max_length) {
    if (RandomFloat(0, 1) >= alpha) {  // include alpha = 0
      if (current.degree == 0)
	break;
      else {
	if (meta_path != NULL) { // meta path
	  current_type_idx = (current_type_idx + 1) % meta_path_length;
	  next_type = meta_path[current_type_idx];
	  bidx = current.type_begin_idx[next_type];
	  eidx = current.type_end_idx[next_type];
	  if (eidx == bidx)
	    break;
	  i = RandomInteger(bidx, eidx);
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


void *GenerateRandomWalkThread(void *params) {
  char **path;
  int actual_length;
  int i, iter;
  clock_t now;
  Params *p;
  FILE *f;
  uint32_t idx, start, end;
  char temp[MAX_STRING];

  p = (Params *)params;
  sprintf(temp, "%s_%d.txt", output_file, p->tid);
  f = fopen(temp , "w");
  if (f == NULL) {
    fprintf(stderr, "(ERROR) Cannot open file %s: %s\n", temp, strerror(errno));
    exit(-1);
  }
  path = (char **)malloc(p->length * sizeof(char *));
  if (path == NULL) {
    perror("(ERROR) Memory allocation failed\n");
    exit(-1);
  }
  for (i = 0; i < p->length; i++) {
    path[i] = (char *)malloc(sizeof(char) * MAX_STRING);
    if (path[i] == NULL) {
      perror("(ERROR) Memory allocation failed\n");
      exit(-1);
    }
  }

  start = p->graph->vcount / p->num_threads * p->tid;
  if (p->tid == p->num_threads - 1)
    end = p->graph->vcount;
  else
    end = p->graph->vcount / p->num_threads * (p->tid + 1);
  for (iter = 0; iter < p->num_per_vertex; iter++)
    for (idx = start; idx < end; idx++) {
      now = clock();
      printf("%cProgress: %.2f%%  Words/thread/sec: %.2fk  ", 13,
	     actual_node_count / (float)(total_node_count) * 100,
	     actual_node_count / ((float)(now - start + 1) / (float)CLOCKS_PER_SEC * 1000));
      fflush(stdout);
      actual_node_count++;
      actual_length = RandomPath(p->graph, path, idx, p->length, p->alpha,
				 p->meta_path_length, p->meta_path);
      for (i = 0; i < actual_length; i++) {
	if (i != 0)
	  fprintf(f, " ");
	fprintf(f, "%s", path[i]);
      }
      fprintf(f, "\n");
    }

  fclose(f);
  for (i = 0; i < p->length; i++)
    free(path[i]);
  free(path);
  pthread_exit(NULL);
}


void GenerateRandomWalk(StaticGraph *g, int path_length, int num_per_vertex,
			float alpha, uint8_t meta_path_length,
			uint8_t *meta_path, int n_jobs) {
  Params *param_list;
  int i;
  pthread_t *pt;

  param_list = (Params *)malloc(n_jobs * sizeof(Params));
  if (param_list == NULL) {
    perror("(ERROR) Memory allocation failed\n");
    exit(-1);
  }
  pt = (pthread_t *)malloc(n_jobs * sizeof(pthread_t));
  if (pt == NULL) {
    perror("(ERROR) Memory allocation failed\n");
    exit(-1);
  }

  start = clock();
  actual_node_count = 0;
  total_node_count = g->vcount * num_per_vertex;
  for (i = 0; i < n_jobs; i++) {
    param_list[i].tid = i;
    param_list[i].num_threads = n_jobs;
    param_list[i].graph = g;
    param_list[i].length = path_length;
    param_list[i].num_per_vertex = num_per_vertex;
    param_list[i].alpha = alpha;
    param_list[i].meta_path = meta_path;
    param_list[i].meta_path_length = meta_path_length;
    pthread_create(&pt[i], NULL, &GenerateRandomWalkThread, (void *)&param_list[i]);
  }
  for (i = 0; i < n_jobs; i++)
    pthread_join(pt[i], NULL);
  free(pt);
  free(param_list);
}

// Generate a random integer in [low, high)
// 
uint32_t RandomInteger(uint32_t low, uint32_t high) {
  return (uint32_t)(ran_uniform() * (high - low)) + low;
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
  /*   printf("%u\n", x[i]); */
}


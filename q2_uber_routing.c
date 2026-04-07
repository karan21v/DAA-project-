/*
 * Q2 - Uber Routing System
 *
 * Problem:
 * A city road network is a directed weighted graph.
 * Nodes are intersections (1 to r_nodes), edges are roads with travel times.
 *
 * For each node we need:
 *   forward_latency[v]  = shortest time to reach v from node 1
 *   reverse_latency[v]  = shortest time to reach r_nodes from v
 *
 * A road (u -> v, weight w) is "consistent" if it lies on at least one
 * shortest path from node 1 to node r_nodes. The condition is:
 *   forward_latency[u] + w + reverse_latency[v] == total_shortest_distance
 *
 * Approach: Run Dijkstra twice.
 *   1. Forward  - from node 1 on original graph
 *   2. Reverse  - from node r_nodes on graph with all edges flipped
 *      (flipping edges lets us find "shortest distance TO r_nodes" for all nodes at once)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_NODES  100002
#define MAX_EDGES  200002
#define INF        0x3f3f3f3f3f3f3f3fLL

// edge in adjacency list
typedef struct {
    int  to;
    int  weight;
    int  next; // index of next edge from same source, -1 if none
} Edge;

static Edge edges[MAX_EDGES];
static int  head[MAX_NODES]; // head[u] = index of first edge from node u
static int  edge_cnt;

static void graph_init(void) {
    memset(head, -1, sizeof(head));
    edge_cnt = 0;
}

static void graph_add(int from, int to, int weight) {
    edges[edge_cnt].to     = to;
    edges[edge_cnt].weight = weight;
    edges[edge_cnt].next   = head[from];
    head[from]             = edge_cnt++;
}

// min-heap node for Dijkstra
typedef struct {
    long long dist;
    int       node;
} HeapNode;

static HeapNode heap[MAX_EDGES];
static int      heap_sz;

static void heap_init(void) { heap_sz = 0; }

static void heap_push(long long dist, int node) {
    int i = heap_sz++;
    heap[i].dist = dist;
    heap[i].node = node;
    // sift up to maintain min-heap property
    while (i > 0) {
        int p = (i - 1) / 2;
        if (heap[p].dist <= heap[i].dist) break;
        HeapNode tmp = heap[p]; heap[p] = heap[i]; heap[i] = tmp;
        i = p;
    }
}

static HeapNode heap_pop(void) {
    HeapNode top = heap[0];
    heap[0] = heap[--heap_sz];
    // sift down to maintain min-heap property
    int i = 0;
    while (1) {
        int L = 2*i+1, R = 2*i+2, sm = i;
        if (L < heap_sz && heap[L].dist < heap[sm].dist) sm = L;
        if (R < heap_sz && heap[R].dist < heap[sm].dist) sm = R;
        if (sm == i) break;
        HeapNode tmp = heap[sm]; heap[sm] = heap[i]; heap[i] = tmp;
        i = sm;
    }
    return top;
}

/*
 * Dijkstra's shortest path algorithm using a min-heap.
 *
 * Always processes the closest unvisited node first (greedy).
 * Lazy deletion: if we pop a node with outdated distance, we skip it.
 * This avoids the need for a decrease-key operation.
 *
 * Time: O((V + E) log V)
 */
static void dijkstra(int source, int n, long long *dist) {
    for (int i = 0; i <= n; i++) dist[i] = INF;
    dist[source] = 0;

    heap_init();
    heap_push(0, source);

    while (heap_sz > 0) {
        HeapNode cur = heap_pop();
        long long d = cur.dist;
        int       u = cur.node;

        if (d > dist[u]) continue; // stale entry, skip

        for (int e = head[u]; e != -1; e = edges[e].next) {
            int       v  = edges[e].to;
            long long nd = dist[u] + edges[e].weight;
            if (nd < dist[v]) {
                dist[v] = nd;
                heap_push(nd, v);
            }
        }
    }
}

static int   g_r_from  [MAX_EDGES/2];
static int   g_r_to    [MAX_EDGES/2];
static int   g_r_weight[MAX_EDGES/2];
static long long dist_fwd[MAX_NODES];
static long long dist_rev[MAX_NODES];

/*
 * markRoads - main function
 *
 * Returns result[] where result[i] = 1 if road i is consistent, 0 otherwise.
 * A road is consistent if it lies on any shortest path from node 1 to r_nodes.
 */
void markRoads(int r_nodes, int r_edges,
               int *r_from, int *r_to, int *r_weight,
               int *result)
{
    // step 1: run Dijkstra on original graph from node 1
    graph_init();
    for (int i = 0; i < r_edges; i++)
        graph_add(r_from[i], r_to[i], r_weight[i]);
    dijkstra(1, r_nodes, dist_fwd);

    // step 2: flip all edges, run Dijkstra from r_nodes
    // this gives us the shortest distance TO r_nodes for every node
    graph_init();
    for (int i = 0; i < r_edges; i++)
        graph_add(r_to[i], r_from[i], r_weight[i]); // direction flipped
    dijkstra(r_nodes, r_nodes, dist_rev);

    long long total = dist_fwd[r_nodes];

    // step 3: check each road
    for (int i = 0; i < r_edges; i++) {
        int u = r_from[i], v = r_to[i], w = r_weight[i];
        // if using this road uses up the budget exactly, it's on a shortest path
        result[i] = (dist_fwd[u] + w + dist_rev[v] == total) ? 1 : 0;
    }
}

// brute force using Floyd-Warshall, only for testing small graphs
#define SMALL 20
static void markRoads_brute(int r_nodes, int r_edges,
                             int *r_from, int *r_to, int *r_weight,
                             int *result)
{
    long long d[SMALL+1][SMALL+1];
    for (int i = 0; i <= r_nodes; i++)
        for (int j = 0; j <= r_nodes; j++)
            d[i][j] = (i == j) ? 0 : INF;

    for (int i = 0; i < r_edges; i++) {
        int u=r_from[i], v=r_to[i], w=r_weight[i];
        if (w < d[u][v]) d[u][v] = w;
    }
    for (int mid=1; mid<=r_nodes; mid++)
        for (int u=1; u<=r_nodes; u++)
            for (int v=1; v<=r_nodes; v++)
                if (d[u][mid] < INF && d[mid][v] < INF)
                    if (d[u][mid]+d[mid][v] < d[u][v])
                        d[u][v] = d[u][mid]+d[mid][v];

    long long tot = d[1][r_nodes];
    for (int i = 0; i < r_edges; i++) {
        int u=r_from[i], v=r_to[i], w=r_weight[i];
        result[i] = (d[1][u]+w+d[v][r_nodes] == tot) ? 1 : 0;
    }
}

static void print_result(int *r, int n) {
    printf("[");
    for (int i = 0; i < n; i++) { if (i) printf(", "); printf("%d", r[i]); }
    printf("]");
}
static int arr_eq(int *a, int *b, int n) {
    for (int i = 0; i < n; i++) if (a[i] != b[i]) return 0;
    return 1;
}

int main(void) {
    printf("Q2 - Uber Routing System\n");
    printf("=========================\n\n");

    int res_dijk[100], res_brut[100];

    // Test 1: 4 nodes, two equal shortest paths both costing 5
    // 1->2(2)->4 and 1->3(4)->4(1), all roads should be consistent
    {
        int n=4, e=4;
        int fr[]={1,2,1,3}, to[]={2,4,3,4}, w[]={2,3,4,1};
        markRoads      (n,e,fr,to,w, res_dijk);
        markRoads_brute(n,e,fr,to,w, res_brut);
        printf("Test 1 - 4 nodes, two paths both cost 5\n");
        printf("  Dijkstra: "); print_result(res_dijk,e); printf("\n");
        printf("  Brute   : "); print_result(res_brut,e); printf("\n");
        printf("  Match   : %s\n\n", arr_eq(res_dijk,res_brut,e)?"PASS":"FAIL");
    }

    // Test 2: edge 1->3 with weight 10 is not on any shortest path
    // shortest is 1->2->3 = 2, so the direct 1->3(10) is a detour
    {
        int n=3, e=3;
        int fr[]={1,2,1}, to[]={2,3,3}, w[]={1,1,10};
        int exp[]={1,1,0};
        markRoads      (n,e,fr,to,w, res_dijk);
        markRoads_brute(n,e,fr,to,w, res_brut);
        printf("Test 2 - edge 1->3 (weight 10) not on shortest path\n");
        printf("  Expected: [1, 1, 0]\n");
        printf("  Dijkstra: "); print_result(res_dijk,e); printf("  %s\n", arr_eq(res_dijk,exp,e)?"PASS":"FAIL");
        printf("  Brute   : "); print_result(res_brut,e); printf("  %s\n\n", arr_eq(res_brut,exp,e)?"PASS":"FAIL");
    }

    // Test 3: chain 1->2->3->4->5, shortcut 2->5 with weight 10 is not optimal
    {
        int n=5, e=5;
        int fr[]={1,2,3,4,2}, to[]={2,3,4,5,5}, w[]={1,1,1,1,10};
        int exp[]={1,1,1,1,0};
        markRoads      (n,e,fr,to,w, res_dijk);
        markRoads_brute(n,e,fr,to,w, res_brut);
        printf("Test 3 - shortcut 2->5 (weight 10) is not consistent\n");
        printf("  Expected: [1, 1, 1, 1, 0]\n");
        printf("  Dijkstra: "); print_result(res_dijk,e); printf("  %s\n", arr_eq(res_dijk,exp,e)?"PASS":"FAIL");
        printf("  Brute   : "); print_result(res_brut,e); printf("  %s\n\n", arr_eq(res_brut,exp,e)?"PASS":"FAIL");
    }

    return 0;
}

/*
 * PSEUDOCODE
 *
 * Dijkstra's Algorithm (min-heap) ──
 *
 * FUNCTION DIJKSTRA(graph, source, n):
 *   dist[0..n] = INFINITY
 *   dist[source] = 0
 *   minHeap = empty priority queue
 *   INSERT (distance=0, node=source) into minHeap
 *
 *   WHILE minHeap is not empty:
 *     (d, u) = EXTRACT_MIN(minHeap)
 *
 *     IF d > dist[u]:
 *       CONTINUE                          // stale heap entry, skip it
 *
 *     FOR each edge (u -> v) with weight w in graph:
 *       new_dist = dist[u] + w
 *       IF new_dist < dist[v]:
 *         dist[v] = new_dist
 *         INSERT (new_dist, v) into minHeap
 *
 *   RETURN dist[]
 *
 *   Time:  O((V + E) log V)   -- each edge may produce one heap insert
 *   Space: O(V + E)
 *
 *   Note on lazy deletion:
 *   When we find a shorter path to v, we insert a new (dist, v) entry
 *   without removing the old one. The old entry becomes stale. We detect
 *   this when popping: if d > dist[u], the entry is outdated so we skip.
 *   This is simpler than decrease-key and works correctly.
 *
 *
 *  markRoads ──
 *
 * FUNCTION markRoads(r_nodes, r_edges, r_from[], r_to[], r_weight[]):
 *
 *   // Step 1: forward shortest paths from node 1
 *   Build fwd_graph using r_from[], r_to[], r_weight[]
 *   dist_fwd[] = DIJKSTRA(fwd_graph, source = 1, r_nodes)
 *
 *   // Step 2: reverse shortest paths TO node r_nodes
 *   Build rev_graph by flipping every edge
 *     (each edge u -> v becomes v -> u, same weight)
 *   dist_rev[] = DIJKSTRA(rev_graph, source = r_nodes, r_nodes)
 *
 *   shortest_total = dist_fwd[r_nodes]
 *
 *   // Step 3: consistency check for each road
 *   FOR i = 0 TO r_edges - 1:
 *     u = r_from[i],  v = r_to[i],  w = r_weight[i]
 *
 *     IF dist_fwd[u] + w + dist_rev[v] == shortest_total:
 *       result[i] = 1      // road lies on a shortest path
 *     ELSE:
 *       result[i] = 0      // road is a detour
 *
 *   RETURN result[]
 *
 *   Time:  O((V + E) log V)   -- two Dijkstra runs dominate
 *   Space: O(V + E)           -- two adjacency lists + two dist arrays
 *
 *
 *  Why the consistency condition works ──
 *   Think of the shortest path as a fixed budget = shortest_total.
 *   To use road u->v (cost w), we spend:
 *     dist_fwd[u]  to reach u from node 1
 *     w            to cross the road
 *     dist_rev[v]  to reach r_nodes from v
 *   If these three exactly add up to the budget, no cost is wasted.
 *   The road is part of some shortest path.
 *   If the sum exceeds the budget, using this road is suboptimal.
 *
 *
 * Why flip edges for reverse Dijkstra ──
 *   We need dist_rev[v] = shortest distance FROM v TO r_nodes.
 *   This is hard to compute directly (would need Dijkstra from every node).
 *   Instead: flip all edges, then run Dijkstra FROM r_nodes.
 *   "Shortest from r_nodes to v in reversed graph"
 *   = "Shortest from v to r_nodes in original graph"
 *   One Dijkstra pass gives us the answer for all nodes at once.
 */

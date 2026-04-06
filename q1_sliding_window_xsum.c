/*
 * Q1 - Sliding Window X-Sum
 *
 * Problem:
 * Given an array nums[], window size k, and integer x:
 * For every subarray of size k, find the top-x most frequent elements
 * (if two elements have the same frequency, pick the larger value),
 * then return the sum of all copies of those elements.
 * If the window has fewer than x distinct elements, just sum the whole window.
 *
 * Compile (Mac/Linux): gcc -O2 -std=c99 q1_sliding_window_xsum.c -o q1
 * Compile (Windows):   gcc -O2 -std=c99 q1_sliding_window_xsum.c -o q1.exe
 * Run (Mac/Linux): ./q1
 * Run (Windows):   q1.exe
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// stores one element and how many times it appears in the current window
typedef struct {
    int value;
    int count;
} FreqEntry;

// used by qsort to sort an integer array in ascending order
static int cmp_int(const void *a, const void *b) {
    return (*(int *)a - *(int *)b);
}

// used by qsort to sort freq table: highest frequency first, larger value breaks ties
static int cmp_freq(const void *a, const void *b) {
    const FreqEntry *x = (const FreqEntry *)a;
    const FreqEntry *y = (const FreqEntry *)b;
    if (y->count != x->count)
        return y->count - x->count;
    return y->value - x->value;
}

/*
 * BRUTE FORCE - O(n * k log k)
 *
 * For every window, sort it, build the frequency table from scratch,
 * then sort the table to pick top-x elements.
 *
 * Simple but slow because we redo everything for each window.
 */
long long *xsum_brute(int *nums, int n, int k, int x, int *out_sz) {
    *out_sz = n - k + 1;
    long long  *result = (long long *)malloc(*out_sz * sizeof(long long));
    int        *win    = (int *)malloc(k * sizeof(int));
    FreqEntry  *fe     = (FreqEntry *)malloc(k * sizeof(FreqEntry));

    for (int i = 0; i <= n - k; i++) {
        // copy current window and sort it so equal values sit together
        memcpy(win, nums + i, k * sizeof(int));
        qsort(win, k, sizeof(int), cmp_int);

        // one pass over sorted window builds the freq table
        int d = 0;
        for (int j = 0; j < k; j++) {
            if (j == 0 || win[j] != win[j - 1]) {
                fe[d].value = win[j];
                fe[d].count = 1;
                d++;
            } else {
                fe[d - 1].count++;
            }
        }

        // fewer distinct elements than x -> sum whole window
        if (d <= x) {
            long long s = 0;
            for (int j = i; j < i + k; j++) s += nums[j];
            result[i] = s;
            continue;
        }

        // sort by (freq desc, value desc) and sum top-x
        qsort(fe, d, sizeof(FreqEntry), cmp_freq);
        long long s = 0;
        for (int t = 0; t < x; t++)
            s += (long long)fe[t].value * fe[t].count;
        result[i] = s;
    }

    free(win);
    free(fe);
    return result;
}

/*
 * OPTIMISED - O(n * k log x)
 *
 * Instead of rebuilding the freq table from scratch each time,
 * we keep it alive and just make two updates when the window moves:
 *   - add the new element coming in on the right
 *   - remove the old element leaving from the left
 *
 * This avoids the O(k) rebuild cost per step.
 */

// find index of val in freq table, returns -1 if not found
static int fe_find(FreqEntry *fe, int d, int val) {
    for (int i = 0; i < d; i++)
        if (fe[i].value == val) return i;
    return -1;
}

// compute x-sum from the current freq table
static long long compute_xsum(FreqEntry *fe, int d, int x) {
    if (d <= x) {
        // fewer distinct than x, sum everything
        long long s = 0;
        for (int i = 0; i < d; i++)
            s += (long long)fe[i].value * fe[i].count;
        return s;
    }
    // sort a copy so we don't mess up the live table
    FreqEntry *tmp = (FreqEntry *)malloc(d * sizeof(FreqEntry));
    memcpy(tmp, fe, d * sizeof(FreqEntry));
    qsort(tmp, d, sizeof(FreqEntry), cmp_freq);
    long long s = 0;
    for (int i = 0; i < x; i++)
        s += (long long)tmp[i].value * tmp[i].count;
    free(tmp);
    return s;
}

long long *xsum_optimised(int *nums, int n, int k, int x, int *out_sz) {
    *out_sz = n - k + 1;
    long long *result = (long long *)malloc(*out_sz * sizeof(long long));
    FreqEntry *fe = (FreqEntry *)malloc(k * sizeof(FreqEntry));
    int d = 0;

    // build freq table for the first window
    for (int i = 0; i < k; i++) {
        int idx = fe_find(fe, d, nums[i]);
        if (idx < 0) { fe[d].value = nums[i]; fe[d].count = 1; d++; }
        else           fe[idx].count++;
    }
    result[0] = compute_xsum(fe, d, x);

    // slide the window one step at a time
    for (int right = k; right < n; right++) {
        int left_val  = nums[right - k]; // element leaving
        int right_val = nums[right];     // element entering

        // add new element
        int idx = fe_find(fe, d, right_val);
        if (idx < 0) { fe[d].value = right_val; fe[d].count = 1; d++; }
        else           fe[idx].count++;

        // remove old element
        idx = fe_find(fe, d, left_val);
        fe[idx].count--;
        if (fe[idx].count == 0) {
            fe[idx] = fe[--d]; // swap with last to avoid shifting
        }

        result[right - k + 1] = compute_xsum(fe, d, x);
    }

    free(fe);
    return result;
}

// helpers for printing and comparing results
static void print_ll(const long long *a, int n) {
    printf("[");
    for (int i = 0; i < n; i++) { if (i) printf(", "); printf("%lld", a[i]); }
    printf("]");
}

static int ll_eq(const long long *a, const long long *b, int n) {
    for (int i = 0; i < n; i++) if (a[i] != b[i]) return 0;
    return 1;
}

int main(void) {
    printf("Q1 - Sliding Window X-Sum\n");
    printf("==========================\n\n");

    int sz;
    long long *brute, *opt;

    // Test 1: basic case
    int A1[] = {1,1,2,2,3,3};
    long long E1[] = {4,5,7,8};
    brute = xsum_brute    (A1, 6, 3, 2, &sz);
    opt   = xsum_optimised(A1, 6, 3, 2, &sz);
    printf("Test 1 - nums=[1,1,2,2,3,3], k=3, x=2\n");
    printf("  Expected : "); print_ll(E1, 4);     printf("\n");
    printf("  Brute    : "); print_ll(brute, sz);  printf("  %s\n", ll_eq(brute,E1,sz)?"PASS":"FAIL");
    printf("  Optimised: "); print_ll(opt, sz);    printf("  %s\n\n", ll_eq(opt,E1,sz)?"PASS":"FAIL");
    free(brute); free(opt);

    // Test 2: fewer distinct elements than x, should sum whole window
    int A2[] = {1,2,3};
    long long E2[] = {6};
    brute = xsum_brute    (A2, 3, 3, 5, &sz);
    opt   = xsum_optimised(A2, 3, 3, 5, &sz);
    printf("Test 2 - nums=[1,2,3], k=3, x=5 (only 3 distinct, less than x)\n");
    printf("  Expected : "); print_ll(E2, 1);     printf("\n");
    printf("  Brute    : "); print_ll(brute, sz);  printf("  %s\n", ll_eq(brute,E2,sz)?"PASS":"FAIL");
    printf("  Optimised: "); print_ll(opt, sz);    printf("  %s\n\n", ll_eq(opt,E2,sz)?"PASS":"FAIL");
    free(brute); free(opt);

    // Test 3: tie-break test - same frequency, larger value should win
    int A3[] = {3,3,2,2,1};
    long long E3[] = {6,4};
    brute = xsum_brute    (A3, 5, 4, 1, &sz);
    opt   = xsum_optimised(A3, 5, 4, 1, &sz);
    printf("Test 3 - nums=[3,3,2,2,1], k=4, x=1 (tie-break: larger value wins)\n");
    printf("  Expected : "); print_ll(E3, 2);     printf("\n");
    printf("  Brute    : "); print_ll(brute, sz);  printf("  %s\n", ll_eq(brute,E3,sz)?"PASS":"FAIL");
    printf("  Optimised: "); print_ll(opt, sz);    printf("  %s\n\n", ll_eq(opt,E3,sz)?"PASS":"FAIL");
    free(brute); free(opt);

    // Performance test
    int N = 3000, K = 100, X = 5;
    int *big = (int *)malloc(N * sizeof(int));
    srand(42);
    for (int i = 0; i < N; i++) big[i] = rand() % 20 + 1;

    clock_t t0, t1;
    t0 = clock(); brute = xsum_brute    (big,N,K,X,&sz); t1 = clock(); free(brute);
    double tb = (double)(t1-t0)/CLOCKS_PER_SEC*1000.0;
    t0 = clock(); opt   = xsum_optimised(big,N,K,X,&sz); t1 = clock(); free(opt);
    double to = (double)(t1-t0)/CLOCKS_PER_SEC*1000.0;

    printf("Performance (n=%d, k=%d, x=%d)\n", N, K, X);
    printf("  Brute    : %.2f ms\n", tb);
    printf("  Optimised: %.2f ms\n", to);
    printf("  Speedup  : %.2fx\n", tb/to);

    free(big);
    return 0;
}

/*
 * PSEUDOCODE
 * ==========
 *
 * ── APPROACH 1: Brute Force ──────────────────────────────────────
 *
 * FUNCTION xsum_brute(nums[0..n-1], k, x):
 *   result = array of size (n - k + 1)
 *
 *   FOR i = 0 TO n - k:
 *     window = nums[i .. i+k-1]
 *
 *     SORT window in ascending order          // so equal values are adjacent
 *
 *     // build frequency table in one linear scan
 *     freq = []
 *     FOR each element v in window:
 *       IF v == previous element:
 *         freq.last.count += 1
 *       ELSE:
 *         freq.append( {value: v, count: 1} )
 *
 *     IF len(freq) <= x:
 *       result[i] = sum of all elements in window   // special case
 *     ELSE:
 *       SORT freq by (count DESC, value DESC)        // top-x first
 *       result[i] = SUM of freq[0..x-1] (value * count)
 *
 *   RETURN result
 *
 *   Time:  O(n * k log k)    -- sort dominates; done once per window
 *   Space: O(k)              -- freq table holds at most k entries
 *
 *
 * ── APPROACH 2: Optimised Sliding Window ─────────────────────────
 *
 * FUNCTION xsum_optimised(nums[0..n-1], k, x):
 *   result = array of size (n - k + 1)
 *   freq   = {}                                // live frequency table
 *
 *   // seed the first window
 *   FOR i = 0 TO k - 1:
 *     freq[nums[i]].count += 1
 *   result[0] = compute_xsum(freq, x)
 *
 *   // slide one step at a time
 *   FOR right = k TO n - 1:
 *     left = right - k
 *
 *     freq[nums[right]].count += 1             // new element enters
 *     freq[nums[left]].count  -= 1             // old element leaves
 *     IF freq[nums[left]].count == 0:
 *       DELETE freq[nums[left]]                // clean up zero entries
 *
 *     result[right - k + 1] = compute_xsum(freq, x)
 *
 *   RETURN result
 *
 *
 * FUNCTION compute_xsum(freq, x):
 *   IF number of distinct elements <= x:
 *     RETURN sum of (value * count) for all entries   // sum whole window
 *   SORT copy of freq by (count DESC, value DESC)
 *   RETURN sum of top-x entries (value * count)
 *
 *   Time:  O(n * k log x)    -- 2 updates per slide, sort only x elements
 *   Space: O(k)              -- freq table has at most k distinct entries
 *
 *
 * ── KEY INSIGHT ──────────────────────────────────────────────────
 *   Brute force throws away and rebuilds the frequency table for
 *   every single window -- O(k) rebuild work each step.
 *   Sliding window keeps the table alive and makes exactly 2 updates
 *   per slide (one ADD, one REMOVE), saving O(k) work per step.
 */

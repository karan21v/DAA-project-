/*
 * Q3 - Strassen's Matrix Multiplication
 *
 * Problem:
 * Standard n x n matrix multiplication takes O(n^3) operations.
 * Strassen's algorithm uses divide and conquer to reduce this.
 *
 * Key idea: split each matrix into 4 quadrants of size n/2.
 * Normal divide and conquer would need 8 recursive multiplications.
 * Strassen reduces this to 7 by computing clever combinations (M1-M7),
 * which means all extra terms cancel out algebraically.
 *
 * Recurrence: T(n) = 7*T(n/2) + O(n^2)
 * By Master Theorem: T(n) = O(n^log2(7)) = O(n^2.807)
 *
 * Matrices are stored as flat 1D arrays in row-major order:
 *   M[i][j] = M_flat[i * n + j]
 * This is cache-friendly.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// access element (i,j) of an n-wide flat matrix
#define AT(M, n, i, j)  ((M)[(i)*(n)+(j)])

// allocate an n x n zero matrix
static double *mat_alloc(int n) {
    double *M = (double *)calloc((size_t)n * n, sizeof(double));
    if (!M) { fprintf(stderr, "Out of memory\n"); exit(1); }
    return M;
}

static void mat_free(double *M) { free(M); }

// fill matrix with random integers in [lo, hi]
static void mat_rand(double *M, int n, int lo, int hi) {
    for (int i = 0; i < n * n; i++)
        M[i] = lo + rand() % (hi - lo + 1);
}

static void mat_copy(double *dst, const double *src, int n) {
    memcpy(dst, src, (size_t)n * n * sizeof(double));
}

// C = A + B
static void mat_add(const double *A, const double *B, double *C, int n) {
    for (int i = 0; i < n * n; i++) C[i] = A[i] + B[i];
}

// C = A - B
static void mat_sub(const double *A, const double *B, double *C, int n) {
    for (int i = 0; i < n * n; i++) C[i] = A[i] - B[i];
}

// split n x n matrix M into four n/2 x n/2 quadrants
static void mat_split(const double *M, int n,
                      double *A11, double *A12,
                      double *A21, double *A22)
{
    int h = n / 2;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < h; j++) {
            AT(A11, h, i, j) = AT(M, n, i,     j    );
            AT(A12, h, i, j) = AT(M, n, i,     j + h);
            AT(A21, h, i, j) = AT(M, n, i + h, j    );
            AT(A22, h, i, j) = AT(M, n, i + h, j + h);
        }
    }
}

// join four n/2 x n/2 quadrants back into one n x n matrix
static void mat_join(double *C,
                     const double *C11, const double *C12,
                     const double *C21, const double *C22, int n)
{
    int h = n / 2;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < h; j++) {
            AT(C, n, i,     j    ) = AT(C11, h, i, j);
            AT(C, n, i,     j + h) = AT(C12, h, i, j);
            AT(C, n, i + h, j    ) = AT(C21, h, i, j);
            AT(C, n, i + h, j + h) = AT(C22, h, i, j);
        }
    }
}

// round n up to next power of 2
static int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

// pad n x n matrix to p x p with zeros
static double *mat_pad(const double *M, int n, int p) {
    double *P = mat_alloc(p);
    for (int i = 0; i < n; i++)
        memcpy(P + i * p, M + i * n, n * sizeof(double));
    return P;
}

// trim p x p matrix back to n x n
static double *mat_trim(const double *M, int p, int n) {
    double *T = mat_alloc(n);
    for (int i = 0; i < n; i++)
        memcpy(T + i * n, M + i * p, n * sizeof(double));
    return T;
}

/*
 * Naive matrix multiplication - O(n^3)
 *
 * Three nested loops. Loop order i-k-j is better than i-j-k
 * for cache performance (A[i][k] is accessed sequentially).
 */
double *naive_multiply(const double *A, const double *B, int n) {
    double *C = mat_alloc(n);
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            if (A[i*n+k] == 0.0) continue;
            for (int j = 0; j < n; j++)
                AT(C, n, i, j) += AT(A, n, i, k) * AT(B, n, k, j);
        }
    return C;
}

/*
 * Strassen's algorithm - O(n^2.807)
 *
 * For A = [[a,b],[c,d]] and B = [[e,f],[g,h]]:
 *
 * 7 products:
 *   M1 = (a+d)(e+h)
 *   M2 = (c+d)*e
 *   M3 = a*(f-h)
 *   M4 = d*(g-e)
 *   M5 = (a+b)*h
 *   M6 = (c-a)(e+f)
 *   M7 = (b-d)(g+h)
 *
 * Result:
 *   C11 = M1 + M4 - M5 + M7
 *   C12 = M3 + M5
 *   C21 = M2 + M4
 *   C22 = M1 - M2 + M3 + M6
 *
 * Below the threshold, we just use naive because the recursion overhead
 * (18 additions + malloc) isn't worth it for small matrices.
 */

#define STRASSEN_THRESHOLD 64

static double *strassen_core(const double *A, const double *B, int n) {
    if (n <= STRASSEN_THRESHOLD)
        return naive_multiply(A, B, n);

    int h = n / 2;

    // split both matrices into quadrants
    double *a = mat_alloc(h), *b = mat_alloc(h);
    double *c = mat_alloc(h), *d = mat_alloc(h);
    double *e = mat_alloc(h), *f = mat_alloc(h);
    double *g = mat_alloc(h), *hh= mat_alloc(h);
    mat_split(A, n, a, b, c, d);
    mat_split(B, n, e, f, g, hh);

    double *t1 = mat_alloc(h), *t2 = mat_alloc(h);

    // compute the 7 products
    mat_add(a, d, t1, h);  mat_add(e, hh, t2, h);
    double *M1 = strassen_core(t1, t2, h);

    mat_add(c, d, t1, h);
    double *M2 = strassen_core(t1, e, h);

    mat_sub(f, hh, t2, h);
    double *M3 = strassen_core(a, t2, h);

    mat_sub(g, e, t2, h);
    double *M4 = strassen_core(d, t2, h);

    mat_add(a, b, t1, h);
    double *M5 = strassen_core(t1, hh, h);

    mat_sub(c, a, t1, h);  mat_add(e, f, t2, h);
    double *M6 = strassen_core(t1, t2, h);

    mat_sub(b, d, t1, h);  mat_add(g, hh, t2, h);
    double *M7 = strassen_core(t1, t2, h);

    mat_free(t1); mat_free(t2);
    mat_free(a); mat_free(b); mat_free(c); mat_free(d);
    mat_free(e); mat_free(f); mat_free(g); mat_free(hh);

    // assemble the result quadrants
    double *C11 = mat_alloc(h), *C12 = mat_alloc(h);
    double *C21 = mat_alloc(h), *C22 = mat_alloc(h);
    double *tmp = mat_alloc(h);

    // C11 = M1 + M4 - M5 + M7
    mat_add(M1, M4, tmp, h);  mat_sub(tmp, M5, C11, h);
    mat_add(C11, M7, tmp, h); mat_copy(C11, tmp, h);

    mat_add(M3, M5, C12, h);  // C12 = M3 + M5

    mat_add(M2, M4, C21, h);  // C21 = M2 + M4

    // C22 = M1 - M2 + M3 + M6
    mat_sub(M1, M2, tmp, h);  mat_add(tmp, M3, C22, h);
    mat_add(C22, M6, tmp, h); mat_copy(C22, tmp, h);

    mat_free(tmp);
    mat_free(M1); mat_free(M2); mat_free(M3); mat_free(M4);
    mat_free(M5); mat_free(M6); mat_free(M7);

    double *C = mat_alloc(n);
    mat_join(C, C11, C12, C21, C22, n);

    mat_free(C11); mat_free(C12); mat_free(C21); mat_free(C22);
    return C;
}

// public function - handles any n by padding to next power of 2
double *strassen_multiply(const double *A, const double *B, int n) {
    int p = next_pow2(n);
    if (p == n) return strassen_core(A, B, n);

    double *Ap = mat_pad(A, n, p);
    double *Bp = mat_pad(B, n, p);
    double *Cp = strassen_core(Ap, Bp, p);
    mat_free(Ap); mat_free(Bp);

    double *C = mat_trim(Cp, p, n);
    mat_free(Cp);
    return C;
}

static int mat_eq(const double *A, const double *B, int n, double tol) {
    for (int i = 0; i < n * n; i++)
        if (fabs(A[i] - B[i]) > tol) return 0;
    return 1;
}

static void mat_print(const double *M, int n) {
    for (int i = 0; i < n; i++) {
        printf("  [");
        for (int j = 0; j < n; j++) {
            if (j) printf(", ");
            printf("%6.1f", AT(M, n, i, j));
        }
        printf("]\n");
    }
}

int main(void) {
    printf("Q3 - Strassen's Matrix Multiplication\n");
    printf("=======================================\n\n");

    // Test 1: known 2x2 result
    {
        double A[] = {1,2, 3,4};
        double B[] = {5,6, 7,8};
        double E[] = {19,22, 43,50};
        double *C = strassen_multiply(A, B, 2);
        printf("Test 1 - 2x2 known result\n");
        printf("  Expected: [[19,22],[43,50]]\n");
        printf("  Got     : [[%.0f,%.0f],[%.0f,%.0f]]  %s\n\n",
               C[0],C[1],C[2],C[3], mat_eq(C,E,2,1e-6)?"PASS":"FAIL");
        mat_free(C);
    }

    // Test 2: multiplying by identity matrix should return original
    {
        int n = 4;
        double I[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
        srand(99);
        double *A = mat_alloc(n);
        mat_rand(A, n, 1, 9);
        double *C = strassen_multiply(A, I, n);
        printf("Test 2 - A x Identity = A (4x4)\n");
        printf("  Match: %s\n\n", mat_eq(A, C, n, 1e-6) ? "PASS" : "FAIL");
        mat_free(A); mat_free(C);
    }

    // Test 3: non-power-of-2 size, checks that padding works correctly
    {
        int n = 5;
        srand(1);
        double *A = mat_alloc(n), *B = mat_alloc(n);
        mat_rand(A, n, -5, 5);
        mat_rand(B, n, -5, 5);
        double *Cn = naive_multiply(A, B, n);
        double *Cs = strassen_multiply(A, B, n);
        printf("Test 3 - 5x5 (non-power-of-2, tests padding)\n");
        printf("  Match: %s\n\n", mat_eq(Cn, Cs, n, 1e-6) ? "PASS" : "FAIL");
        mat_free(A); mat_free(B); mat_free(Cn); mat_free(Cs);
    }

    // Test 4: print 3x3 result to visually verify
    {
        int n = 3;
        double A[] = {1,2,3, 4,5,6, 7,8,9};
        double B[] = {9,8,7, 6,5,4, 3,2,1};
        double *Cn = naive_multiply(A, B, n);
        double *Cs = strassen_multiply(A, B, n);
        printf("Test 4 - 3x3 printed output\n");
        printf("  Naive:\n"); mat_print(Cn, n);
        printf("  Strassen:\n"); mat_print(Cs, n);
        printf("  Match: %s\n\n", mat_eq(Cn, Cs, n, 1e-6) ? "PASS" : "FAIL");
        mat_free(Cn); mat_free(Cs);
    }

    printf("Complexity summary:\n");
    printf("  Naive    : O(n^3)       at n=256 -> %lld ops\n", (long long)256*256*256);
    printf("  Strassen : O(n^2.807)   at n=256 -> ~%.0f ops\n", pow(256.0, log(7)/log(2)));
    printf("  Crossover: naive is faster for n <= %d\n\n", STRASSEN_THRESHOLD);

    // Benchmark
    printf("Benchmark:\n");
    printf("%6s | %12s | %14s | %8s | %s\n", "n", "Naive (ms)", "Strassen (ms)", "Speedup", "Match");
    printf("-----------------------------------------------------------\n");

    int sizes[] = {64, 128, 256, 512};
    int nsizes  = sizeof(sizes) / sizeof(sizes[0]);
    srand(7);

    for (int s = 0; s < nsizes; s++) {
        int n = sizes[s];
        double *A = mat_alloc(n), *B = mat_alloc(n);
        mat_rand(A, n, -10, 10);
        mat_rand(B, n, -10, 10);

        clock_t t0, t1;
        t0 = clock();
        double *Cn = naive_multiply(A, B, n);
        t1 = clock();
        double t_naive = (double)(t1 - t0) / CLOCKS_PER_SEC * 1000.0;

        t0 = clock();
        double *Cs = strassen_multiply(A, B, n);
        t1 = clock();
        double t_str = (double)(t1 - t0) / CLOCKS_PER_SEC * 1000.0;

        int ok = mat_eq(Cn, Cs, n, 1e-4);
        printf("%6d | %12.1f | %14.1f | %7.2fx  | %s\n",
               n, t_naive, t_str, t_naive / t_str, ok ? "PASS" : "FAIL");

        mat_free(A); mat_free(B); mat_free(Cn); mat_free(Cs);
    }

    printf("\nNote on numerical stability:\n");
    printf("Strassen does 18 matrix additions per recursion level.\n");
    printf("These accumulate floating-point rounding errors across log2(n) levels.\n");
    printf("That's why BLAS/LAPACK still use the naive algorithm for precision-critical work.\n");

    return 0;
}

/*
 * PSEUDOCODE
 *
 *  APPROACH 1: Naive Multiply ──
 *
 * FUNCTION naive_multiply(A[n x n], B[n x n]):
 *   C = zero matrix of size n x n
 *   FOR i = 0 TO n-1:
 *     FOR k = 0 TO n-1:           // i-k-j order is cache-friendly
 *       FOR j = 0 TO n-1:
 *         C[i][j] += A[i][k] * B[k][j]
 *   RETURN C
 *
 *   Time:  O(n^3)   Space: O(n^2)
 *
 *
 *  APPROACH 2: Strassen's Algorithm ───
 *
 * FUNCTION strassen(A[n x n], B[n x n]):
 *
 *   IF n <= THRESHOLD (64):
 *     RETURN naive_multiply(A, B)    // base case: recursion overhead too high
 *
 *   // pad to next power of 2 if n is not even (handles arbitrary n)
 *
 *   // split each matrix into four n/2 x n/2 quadrants
 *   A = | a  b |       B = | e  f |
 *       | c  d |           | g  h |
 *
 *   // compute 7 products instead of 8
 *   M1 = strassen( a+d,  e+h )
 *   M2 = strassen( c+d,  e   )
 *   M3 = strassen( a,    f-h )
 *   M4 = strassen( d,    g-e )
 *   M5 = strassen( a+b,  h   )
 *   M6 = strassen( c-a,  e+f )
 *   M7 = strassen( b-d,  g+h )
 *
 *   // assemble result using only additions and subtractions
 *   C11 = M1 + M4 - M5 + M7
 *   C12 = M3 + M5
 *   C21 = M2 + M4
 *   C22 = M1 - M2 + M3 + M6
 *
 *   RETURN join(C11, C12, C21, C22)
 *
 *   Recurrence:  T(n) = 7 * T(n/2) + Theta(n^2)
 *   Master Theorem: a=7, b=2, f(n)=n^2
 *     log_b(a) = log2(7) = 2.807 > 2
 *     Case 1 applies => T(n) = Theta(n^2.807)
 *
 *   Time:  O(n^2.807)       Space: O(n^2 * log n)
 *
 *
 * Algebraic Proof that C11 = ae + bg ──
 *
 *   Standard result for top-left quadrant: C11 = ae + bg
 *
 *   Strassen computes C11 = M1 + M4 - M5 + M7. Expanding:
 *
 *     M1 = (a+d)(e+h) = ae + ah + de + dh
 *     M4 = d(g-e)     = dg - de
 *     M5 = (a+b)h     = ah + bh
 *     M7 = (b-d)(g+h) = bg + bh - dg - dh
 *
 *   Sum: M1 + M4 - M5 + M7
 *      = ae + ah + de + dh
 *      + dg - de
 *      - ah - bh
 *      + bg + bh - dg - dh
 *
 *   Cancellations:
 *      ah - ah = 0
 *      de - de = 0
 *      dh - dh = 0
 *      dg - dg = 0
 *     -bh + bh = 0
 *
 *   Remaining: ae + bg  -- which is exactly C11.
 *
 *   The same algebraic cancellation principle applies to C12, C21, C22.
 *   This is Strassen's key insight: trading one multiplication for
 *   several additions saves work since additions cost O(n^2) each.
 *
 *
 *  Crossover Point ──
 *   For small n, Strassen's 7 recursive calls + 18 matrix additions
 *   + memory allocation overhead outweigh the savings.
 *   Empirically, naive is faster for n <= 64.
 *   At n = 512 Strassen is ~1.7x faster (see benchmark output).
 *
 *
 *  Numerical Stability ───
 *   Strassen does 18 additions per recursion level x log2(n) levels.
 *   Each addition introduces tiny floating-point rounding errors.
 *   These errors compound across all levels of the recursion.
 *   For n = 1024 that is 10 levels of accumulated error.
 *   BLAS/LAPACK still use naive multiply for this reason.
 */

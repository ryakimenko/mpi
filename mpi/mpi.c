#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>
#include <inttypes.h>

void get_chunk(int a, int b, int commsize, int rank, int *lb, int *ub)
{
    /* OpenMP 4.0 spec (Sec. 2.7.1, default schedule for loops)
     * For a team of commsize processes and a sequence of n items, let ceil(n ? commsize) be the integer q
     * that satisfies n = commsize * q - r, with 0 <= r < commsize.
     * Assign q iterations to the first commsize - r procs, and q - 1 iterations to the remaining r processes */
    int n = b - a + 1;
    int q = n / commsize;
    if (n % commsize)
        q++;
    int r = commsize * q - n;
    /* Compute chunk size for the process */
    int chunk = q;
    if (rank >= commsize - r)
        chunk = q - 1;
    *lb = a; /* Determine start item for the process */
    if (rank > 0)
    { /* Count sum of previous chunks */
        if (rank <= commsize - r)
            *lb += q * rank;
        else
            *lb += q * (commsize - r) + (q - 1) * (rank - (commsize - r));
    }
    *ub = *lb + chunk - 1;
}



const double eps = 0.001;

int main(int argc, char* argv[])
{
    int n = atoi(argv[1]);
    int commsize, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int lb, ub;
    get_chunk(0, n - 1, commsize, rank, &lb, &ub); 
    int nrows = ub - lb + 1;

    double *a = malloc(sizeof(*a) * nrows * n);
    double *b = malloc(sizeof(*b) * nrows);
    double *x = malloc(sizeof(*x) * n);
    double *TempX = malloc(sizeof(*x) * n);

    
    for (int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < n; j++)
            a[i * n + j] = lb + i + 1;
    }
    for (int j = 0; j < nrows; j++){
        b[j] = j + 1;
        TempX[lb + j] = 0; 
    }
    for(int i = 0; i < n; i++)
    {
        x[i] = 0;
    }

    double maxdiff = 0.0;
    double wtime = MPI_Wtime();
    do {

        for(int i = 0; i < nrows; i++)
        {
            TempX[i] =- b[i];
            for(int j = 0; j < n; j++)
            {
                if(i != j)
                TempX[lb + i] -= a[i * n + j] * x[j];
            }
            TempX[lb + i] /= a[i * n + i];
        }
        maxdiff = fabs(x[lb] - TempX[lb]);
        for (int i = 0; i < nrows; i++) {
			if (fabs(x[lb + i] - TempX[lb + i]) > maxdiff)
				maxdiff = fabs(x[lb + i] - TempX[lb + i]);
            x[lb + i] = TempX[lb + i];
		}
        MPI_Allreduce(MPI_IN_PLACE, &maxdiff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if(maxdiff > eps)
        {
            if(rank == 0)
            {
                int *displs = malloc(sizeof(*displs) * commsize);
                int *rcounts = malloc(sizeof(*rcounts) * commsize);
                for (int i = 0; i < commsize; i++)
                {
                    int l, u;
                    get_chunk(0, n - 1, commsize, i, &l, &u);
                    rcounts[i] = u - l + 1;
                    displs[i] = (i > 0) ? displs[i - 1] + rcounts[i - 1] : 0;
                }
                MPI_Gatherv(MPI_IN_PLACE, ub - lb + 1, MPI_DOUBLE, x, rcounts, displs, MPI_DOUBLE, 0,
                            MPI_COMM_WORLD);
                MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            }
            else
            {
                MPI_Gatherv(&x[lb], ub - lb + 1, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            }
        }
        else
        {
           if(rank == 0)
            {
                int *displs = malloc(sizeof(*displs) * commsize);
                int *rcounts = malloc(sizeof(*rcounts) * commsize);
                for (int i = 0; i < commsize; i++)
                {
                    int l, u;
                    get_chunk(0, n - 1, commsize, i, &l, &u);
                    rcounts[i] = u - l + 1;
                    displs[i] = (i > 0) ? displs[i - 1] + rcounts[i - 1] : 0;
                }
                    MPI_Gatherv(MPI_IN_PLACE, ub - lb + 1, MPI_FLOAT, x, rcounts, displs, MPI_DOUBLE, 0,
                        MPI_COMM_WORLD);
                free(displs);
                free(rcounts);
            }
            else
            {
                MPI_Gatherv(&x[lb], ub - lb + 1, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            }
        }
    } while (maxdiff > eps);

    wtime = MPI_Wtime() - wtime;

    double Tmax;
    MPI_Reduce(&wtime, &Tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
            printf("Time: %lf ", Tmax);
    }
    free(a);
    free(b);
    free(x);
    free(TempX);

    MPI_Finalize();

    return 0;
}

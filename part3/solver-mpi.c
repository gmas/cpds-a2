#include "heat.h"
#include "mpi.h"

#define min(a,b) ( ((a) < (b)) ? (a) : (b) )
#define NB 8
/*
 * Blocked Jacobi solver: one iteration step
 */
double relax_jacobi (double *u, double *utmp, unsigned sizex, unsigned sizey, int prev_rank, int next_rank)
{
    double diff, sum=0.0;
    int nbx, bx, nby, by;
  
    nbx = NB;
    bx = sizex/nbx;
    nby = NB;
    by = sizey/nby;
    MPI_Status status;

    for (int ii=0; ii<nbx; ii++)
        for (int jj=0; jj<nby; jj++) {
            for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
                for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
	            utmp[i*sizey+j]= 0.25 * (u[ i*sizey     + (j-1) ]+  // left
					     u[ i*sizey     + (j+1) ]+  // right
				             u[ (i-1)*sizey + j     ]+  // top
				             u[ (i+1)*sizey + j     ]); // bottom
	            diff = utmp[i*sizey+j] - u[i*sizey + j];
	            sum += diff * diff; 
	        }
        }
    size_t last_non_halo =  (sizex -2) * sizey;
    MPI_Send(&utmp[last_non_halo], sizey, MPI_DOUBLE, next_rank, 0, MPI_COMM_WORLD);
    MPI_Recv(utmp, sizey, MPI_DOUBLE, prev_rank, 0, MPI_COMM_WORLD, &status);

    size_t first_non_halo =  sizey;
    size_t last_halo =  (sizex -1) * sizey;
    MPI_Send(&utmp[first_non_halo], sizey, MPI_DOUBLE, prev_rank, 0, MPI_COMM_WORLD);
    MPI_Recv(&utmp[last_halo], sizey, MPI_DOUBLE, next_rank, 0, MPI_COMM_WORLD, &status);


    // printf("%ld, 0, %ld, %ld\n", last_non_halo, first_non_halo, last_halo);
    return sum;
}

/*
 * Blocked Red-Black solver: one iteration step
 */
double relax_redblack (double *u, unsigned sizex, unsigned sizey)
{
    double unew, diff, sum=0.0;
    int nbx, bx, nby, by;
    int lsw;

    nbx = NB;
    bx = sizex/nbx;
    nby = NB;
    by = sizey/nby;
    // Computing "Red" blocks
    for (int ii=0; ii<nbx; ii++) {
        lsw = ii%2;
        for (int jj=lsw; jj<nby; jj=jj+2) 
            for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
                for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
	            unew= 0.25 * (    u[ i*sizey	+ (j-1) ]+  // left
				      u[ i*sizey	+ (j+1) ]+  // right
				      u[ (i-1)*sizey	+ j     ]+  // top
				      u[ (i+1)*sizey	+ j     ]); // bottom
	            diff = unew - u[i*sizey+ j];
	            sum += diff * diff; 
	            u[i*sizey+j]=unew;
	        }
    }

    // Computing "Black" blocks
    for (int ii=0; ii<nbx; ii++) {
        lsw = (ii+1)%2;
        for (int jj=lsw; jj<nby; jj=jj+2) 
            for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
                for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
	            unew= 0.25 * (    u[ i*sizey	+ (j-1) ]+  // left
				      u[ i*sizey	+ (j+1) ]+  // right
				      u[ (i-1)*sizey	+ j     ]+  // top
				      u[ (i+1)*sizey	+ j     ]); // bottom
	            diff = unew - u[i*sizey+ j];
	            sum += diff * diff; 
	            u[i*sizey+j]=unew;
	        }
    }

    return sum;
}

/*
 * Blocked Gauss-Seidel solver: one iteration step
 */
double relax_gauss (double *u, unsigned sizex, unsigned sizey, int prev_rank, int next_rank)
{
    size_t last_non_halo =  (sizex -2) * sizey;
    MPI_Status status;

    MPI_Recv(u, sizey, MPI_DOUBLE, prev_rank, 0, MPI_COMM_WORLD, &status);
    double unew, diff, sum=0.0;
    int nbx, bx, nby, by;

    nbx = NB;
    bx = sizex/nbx;
    nby = NB;
    by = sizey/nby;
    for (int ii=0; ii<nbx; ii++)
        for (int jj=0; jj<nby; jj++) 
            for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
                for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
	            unew= 0.25 * (    u[ i*sizey	+ (j-1) ]+  // left
				      u[ i*sizey	+ (j+1) ]+  // right
				      u[ (i-1)*sizey	+ j     ]+  // top
				      u[ (i+1)*sizey	+ j     ]); // bottom
	            diff = unew - u[i*sizey+ j];
	            sum += diff * diff; 
	            u[i*sizey+j]=unew;
                }
    
    MPI_Send(&u[last_non_halo], sizey, MPI_DOUBLE, next_rank, 0, MPI_COMM_WORLD);

    size_t first_non_halo =  sizey;
    size_t last_halo =  (sizex -1) * sizey;
    MPI_Send(&u[first_non_halo], sizey, MPI_DOUBLE, prev_rank, 0, MPI_COMM_WORLD);
    MPI_Recv(&u[last_halo], sizey, MPI_DOUBLE, next_rank, 0, MPI_COMM_WORLD, &status);

    return sum;
}
#include "heat.h"

#define NB 8

#define min(a,b) ( ((a) < (b)) ? (a) : (b) )

/*
 * Blocked Jacobi solver: one iteration step
 */
double relax_jacobi (double *u, double *utmp, unsigned sizex, unsigned sizey)
{
    double diff, sum=0.0;
    int nbx, bx, nby, by;
  
    nbx = NB; // 8
    bx = sizex/nbx; // (resolution + 2 = 258) / 8 = 32
    nby = NB; // 8
    by = sizey/nby; // 258 / 8 = 32
		    //
    //#pragma omp parallel for collapse(2) reduction(+:sum)
    #pragma omp parallel for private(diff) reduction(+:sum)
    for (int ii=0; ii<nbx; ii++) // 0 .. 7
        //#pragma omp parallel for
        for (int jj=0; jj<nby; jj++) // 0 .. 7
            for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) // 1 ..
		//#pragma omp parallel for private(diff) reduction(+:sum)
                for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
		    //#pragma omp parallel private(diff) reduction(+:sum) 
	            utmp[i*sizey+j]= 0.25 * (u[ i*sizey     + (j-1) ]+  // left
					     u[ i*sizey     + (j+1) ]+  // right
				             u[ (i-1)*sizey + j     ]+  // top
				             u[ (i+1)*sizey + j     ]); // bottom
	            diff = utmp[i*sizey+j] - u[i*sizey + j];
	            sum += diff * diff; 
	        }

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
double relax_gauss (double *u, unsigned sizex, unsigned sizey)
{
    double unew, diff, sum=0.0;
    int nbx, bx, nby, by;

    nbx = NB;
    bx = sizex/nbx;
    nby = NB;
    by = sizey/nby;
    int ii,jj;
    #pragma omp parallel for ordered(2) collapse(2) private(unew, diff)
    for (ii=0; ii<nbx; ii++)
        for (jj=0; jj<nby; jj++) 
        {
            #pragma omp ordered depend(sink: ii-1, jj) depend(sink: ii, jj-1)
	    {
            double partial_sum = 0;
            for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
                for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
                    unew= 0.25 * (    u[ i*sizey        + (j-1) ]+  // left
                                      u[ i*sizey        + (j+1) ]+  // right
                                      u[ (i-1)*sizey    + j     ]+  // top
                                      u[ (i+1)*sizey    + j     ]); // bottom
                    diff = unew - u[i*sizey+ j];
                    partial_sum += diff * diff; 
                    u[i*sizey+j]=unew;
                }
	    #pragma omp atomic
	    sum += partial_sum;
            #pragma omp ordered depend(source)
	    }
	}
    printf("Current sum: %3.5f\n", sum);
    return sum;
}


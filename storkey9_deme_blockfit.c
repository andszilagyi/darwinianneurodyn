// recombination


#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>

/******* RANDOM NUMBER GENERATOR **********/
#define AA 471
#define BB 1586
#define CC 6988
#define DD 9689
#define M 16383
#define RIMAX 2147483648.0        /* = 2^31 */
#define RandomInteger (++nd, ra[nd & M] = ra[(nd-AA) & M] ^ ra[(nd-BB) & M] ^ ra[(nd-CC) & M] ^ ra[(nd-DD) & M])
void seed(long seed);
static long ra[M+1], nd;


void seed(long seed)
{
 int  i;

 if(seed<0) { puts("SEED error."); exit(1); }
 ra[0]= (long) fmod(16807.0*(double)seed, 2147483647.0);
 for(i=1; i<=M; i++)
 {
  ra[i] = (long)fmod( 16807.0 * (double) ra[i-1], 2147483647.0);
 }
}

long randl(long num)      /* random number between 0 and num-1 */
{
 return(RandomInteger % num);
}

double randd(void)
{
 return((double) RandomInteger / RIMAX);
}
/********* END OF RANDOM NUMBER GENERATOR ********/

// mode parameters

#define N (80)  // number of neurons
#define NN (10) // number of networks (was 25)
#define D (100) // number of demes
#define TRAINNUM (5)
#define Prec (0.1) // probability of recombination
#define Pflow (0.004) // probability of between-deme flow
#define PN (5) // depth of memory (initial dummy training)
#define Th (0.0) // firing threshold
#define NOUP (5) // number of update steps


// fitness landscape paramters

#define P (10)  // partition size
#define T (2)   // number of optima per block
#define B ((int)(N/P))

// global variables

float output[D][NN][N];
float weight[D][NN][N][N];
float recom1[N],recom2[N];
int perm[NN];

unsigned long int zeed = 0;
float blockOptimaSequence[B][T][P];
float blockOptimaFitness [B][T];
float blockOptimaMax;


int hammingDistanceN(float *v, float *u, int n) { // standard HD up to length `n`
  int i, d = 0;
  for(i = 0; i < n; i++) if(v[i] != u[i]) d++;
  return(d);
}

void setBlockOptimaDefault( ) { // sets the same T optima (sequence and fitness) for the B blocks: first is [1111...], rest is [0101...]
	int i, j, k;
	for(i = 0; i < B; i++) {
		for(j = 0; j < T; j++) {
			if(j == 0) {
				for(k = 0; k < P; k++) blockOptimaSequence[i][j][k] = 1.0; // [1111...]
				blockOptimaFitness[i][j] = 10.0;
			} else {
				for(k = 0; k < P; k++) blockOptimaSequence[i][j][k] = (k % 2)?(1.0):(-1.0); // [0101...]
				blockOptimaFitness[i][j] = 1.0;
			}
		}
	}
}

float blockGlobalOptimumFitness( ) { // calculates global optimum using `blockOptimaSequence` and `blockOptimaFitness`
	int i, j, bestP;
	float bestW, sum = 0.0;
	for(i = 0; i < B; i++) {
		bestW = 0.0; // best fitness
		bestP = -99; // position of best fitness
		for(j = 0; j < T; j++) { // find best fitnessed target for given block; assuming there are no to identical best fitnesses
			if(blockOptimaFitness[i][j] > bestW) {
				bestP = j;
				bestW = blockOptimaFitness[i][j];
			}
		}
		for(j = 0; j < T; j++) if(j == bestP) sum += bestW; else sum += 1.0/(1.0 + hammingDistanceN(blockOptimaSequence[i][bestP], blockOptimaSequence[i][j], P));
	}
	return(sum);
}

void blockGlobalOptimumSequence(float *v) { // calculates global optimum using `blockOptimaSequence` and `blockOptimaFitness`
	int i, j, bestP;
	float bestW;
	for(i = 0; i < B; i++) {
		bestW = 0.0; // best fitness
		bestP = -99; // position of best fitness
		for(j = 0; j < T; j++) { // find best fitnessed target for given block; assuming there are no to identical best fitnesses
			if(blockOptimaFitness[i][j] > bestW) {
				bestP = j;
				bestW = blockOptimaFitness[i][j];
			}
		}
		for(j = 0; j < P; j++) v[i*P + j] = blockOptimaSequence[i][bestP][j];
	}
}

float blockFitness(float *v) { // building block fitness
	int i, j, d;
	float c, block[P], sum = 0.0;
	for(i = 0; i < B; i++) {
		for(j = 0; j < P; j++) block[j] = v[i*P + j];
		for(j = 0; j < T; j++) {
			d = hammingDistanceN(block, blockOptimaSequence[i][j], P);
			if(d == 0) c = blockOptimaFitness[i][j]; else c = 1.0/(1.0+(float)d);
			sum += c;
			//printf("\t%d %d %d %f %f\n", i, j, d, c, sum);
		}
	}
	return(sum);
}

float relativeBlockFitness(float *v) { //rescales {min, max} to {min/max, 1.0}
	return(blockFitness(v)/blockOptimaMax);
}



void init(void)
{
  int i,j,k,l;
  
  for(l=0;l<D;l++)
    for(k=0;k<NN;k++)
      for(i=0;i<N;i++)
        for(j=0;j<N;j++)
          weight[l][k][i][j]=0.0;
	
   for(k=0;k<NN;k++)
     perm[k]=k;
   
   setBlockOptimaDefault();
   blockOptimaMax=blockGlobalOptimumFitness();   
}

void piksr2(int n,float *arr, float *brr)
{
	int i,j;
	float a,b;

	for (j=2;j<=n;j++) {
		a=arr[j];
		b=brr[j];
		i=j-1;
		while (i > 0 && arr[i] > a) {
			arr[i+1]=arr[i];
			brr[i+1]=brr[i];
			i--;
		}
		arr[i+1]=a;
		brr[i+1]=b;
	}
}


// training network 'm' from deme 'd' with vector vec -- palimpsest
void train_network_pali(int d, int m, float *vec)
{
  int i,j;
  float h[N];
  
  for(i=0;i<N;i++)
  {
    h[i]=0.0;
    for(j=0;j<N;j++)    
      h[i]+=weight[d][m][i][j]*vec[j];
  }

  for(i=0;i<N;i++)
  {
    for(j=0;j<N;j++)
    {
      if(i==j)
	weight[d][m][i][j]=0.0;
      else
        weight[d][m][i][j]+=1.0/((float)N)*vec[i]*vec[j]-1.0/((float)N)*vec[i]*h[j]-1.0/((float)N)*vec[j]*h[i];
    }
  }
}


void init_nets_random_memory(void)
{
  int l,m,k,n;
  float vect[N];
  
  for(l=0;l<D;l++)
  {
    for(m=0;m<NN;m++)
    {
      for(k=0;k<PN;k++)
      {     
        for(n=0;n<N;n++)
        {
          vect[n]=2.0*randl(2)-1.0;
        }
        train_network_pali(l,m,vect);
      }
    }
  } 
  
}

void set_outputs_same_random(void)
{
  int n,m,l;
  float vect[N];

  for(n=0;n<N;n++)
    vect[n]=2.0*randl(2)-1.0;;

  for(l=0;l<D;l++)
    for(m=0;m<NN;m++)
      for(n=0;n<N;n++)
        output[l][m][n]=vect[n];

}

// update of neuron 'n' in network 'm' in deme 'd' with threshold neuron model
void update_output(int d, int m, int n)
{
   int t;
   float h=0.0;
   
   for(t = 0; t < N; t++)
     if(t != n )
       h += weight[d][m][n][t] * output[d][m][t];

   if(h>=Th)
      output[d][m][n] =  1.0; // linear threshold
   else
      output[d][m][n] = -1.0;

}

// updating network 'm' in deme 'd'
void update_net(int d, int m) 
{
   int t;

   for(t=0;t<N;t++)
     update_output(d, m, randl(N));
}

void tp_recombine(float *rec1, float *rec2)
{
  int k,c1,c2;
  
  c1=-1;
  c2=-1;
  
  while(c1==c2)
  {
    c1=randl(N);
    c2=randl(N);
  }
  
  if(c1>c2)
  {
    k=c1;
    c1=c2;
    c2=k;
  }
  
  for(k=0;k<c1;k++)
  {
      recom1[k]=rec1[k];
      recom2[k]=rec2[k];
  }
  for(k=c1;k<c2;k++)
  {
      recom1[k]=rec2[k];
      recom2[k]=rec1[k];
  }
  for(k=c2;k<N;k++)
  {
      recom1[k]=rec1[k];
      recom2[k]=rec2[k];
  }  
  
  
}


int main(int argc, char** argv)
{
 
   int l,m,n,up,rou,rr1,rr2,rr3;
   float vect[D][N],vect1[N],vect2[N],fit[NN+1],ref[NN+1],fitness[D][NN];
   float outpool[D][NN][N],fn1,fn2;
   gsl_rng *r;

   seed(505); //50500
   zeed=1051;
   r=gsl_rng_alloc(gsl_rng_mt19937);
   r->type->set(r->state, zeed);
   
   init();
   init_nets_random_memory();
   set_outputs_same_random();

  for(rou=1;rou<10000;rou++)
  {

//update all nets in all demes
    for(l=0;l<D;l++)
      for(m=0;m<NN;m++)
        for(up=0;up<NOUP;up++)
          update_net(l,m);
    
//sort and find the best solution for each deme

    for(l=0;l<D;l++)
    {

      for(m=0;m<NN;m++)
      {
        ref[m+1]=m;
        fit[m+1]=relativeBlockFitness(output[l][m]);
      }

      piksr2(NN,fit,ref);
      
// collect sorted outputs in outpool (outpool[l][0][.] has the best fitnes)
      for(m=0;m<NN;m++)
      {
        rr1=rint(ref[NN-m]);
        for(n=0;n<N;n++)  
          outpool[l][m][n]=output[l][rr1][n];
	fitness[l][m]=fit[NN-m];
      }

// recombination or muation
      rr1=randl(NN);
      if(randd() < Prec)  // recombination
      {
        rr2=randl(NN);
        while(rr1==rr2)
	  rr2=randl(NN);
        for(n=0;n<N;n++)
        {
          vect1[n]=outpool[l][rr1][n];
          vect2[n]=outpool[l][rr2][n];
        }
        
        // flow with low probability
        if(randd()<Pflow)
	{
	  rr3=randl(D);
	  while(rr3==l)
	    rr3=randl(D);
	  for(n=0;n<N;n++)
	    vect2[n]=outpool[rr3][rr2][n];
	}
        
        tp_recombine(vect1,vect2);
      
        fn1=relativeBlockFitness(recom1);
	fn2=relativeBlockFitness(recom2);
        if(fn1>=fn2)
        {
          for(n=0;n<N;n++)
            vect[l][n]=recom1[n];
        }
        else
        {
          for(n=0;n<N;n++)
            vect[l][n]=recom2[n];
          fn1=fn2;
        }	
      }
      else  // mutation
      {

        for(n=0;n<N;n++)
        {
          vect[l][n]=outpool[l][rr1][n];
          if(randd()<1.0/((float)N))
	    vect[l][n]=-1.0*vect[l][n];
        }
        fn1=relativeBlockFitness(vect[l]);  
	
      }
      
// replace the worst if the current is better and train a random network with this
      if(fn1>fitness[l][NN-1])
      {
        for(n=0;n<N;n++)
	  outpool[l][NN-1][n]=vect[l][n];
        fitness[l][NN-1]=fn1;
        
        gsl_ran_shuffle(r,perm,NN,sizeof(int));	
	for(m=0;m<TRAINNUM;m++)
	{
	  rr1=perm[m];
          train_network_pali(l,rr1,vect[l]); 
	}

      }

      
// set one pattern randomly from the pool this will be the input for next round
      gsl_ran_shuffle(r,perm,NN,sizeof(int));
      for(m=0;m<NN;m++)
      {
        rr1=perm[m];
        for(n=0;n<N;n++)
        {
          output[l][m][n]=outpool[l][rr1][n];
        }
      }
    
    }// deme    

    
// find the global best
    fn1=0.0;
    rr1=0;
    for(l=0;l<D;l++)
    {
      if(fitness[l][0]>fn1)
      {
        fn1=fitness[l][0];
	rr1=l;
      }
    }
        
    if(rou%2==0)
    {
      //printf("# Round: %d\t global best fitness:%f\n",rou,fn1);
      printf("%d\t%f\n",rou,fn1);
/*      
      	  for(n=0;n<N;n++)
	  {
	    if(outpool[rr1][0][n]==1)
	      printf("+");
	    if(outpool[rr1][0][n]==-1)
	      printf("-");
	  }
	  printf("\n");
*/
      fflush(stdout);
    }
    
  }//round

  return(0);
}

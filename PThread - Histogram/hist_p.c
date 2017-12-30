#include <stdio.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>

int *vet,size;

struct my_args{
	int rank;
	int nval;
	int nbins;
	double min;
	double max; 
	double h;
	double * val;
};


/* funcao que calcula o minimo valor em um vetor */
double min_val(double * vet,int nval) {
	int i;
	double min;

	min = FLT_MAX;

	for(i=0;i<nval;i++) {
		if(vet[i] < min)
			min =  vet[i];
	}
	
	return min;
}

/* funcao que calcula o maximo valor em um vetor */
double max_val(double * vet, int nval) {
	int i;
	double max;

	max = FLT_MIN;

	for(i=0;i<nval;i++) {
		if(vet[i] > max)
			max =  vet[i];
	}
	
	return max;
}

/* conta quantos valores no vetor estao entre o minimo e o maximo passados como parametros */
void * count(void *args) {
	int i, j, count;
	double min_t, max_t;

	struct my_args *info = args;

	int my_rank = info->rank;
	int local_nbins = (info->nbins)/size;
	int my_first_row = my_rank * local_nbins;
	int my_last_row = (my_rank + 1) * local_nbins - 1;

	for(j = my_first_row ; j <= my_last_row ;j++) {
		count = 0;
		min_t = info->min + j*info->h;
		max_t = info->min + (j+1)*info->h;

		for( i = 0 ; i < info->nval ; i++) {
			if(info->val[i] <= max_t && info->val[i] > min_t) {
				count++;
			}
		}

		vet[j] = count;
	}
	free (info);
	return NULL;
}

int main(int argc, char * argv[]) {
	double h, *val, max, min;
	int n, nval, i;
	long unsigned int duracao;
	struct timeval start, end;

	//atribuição do valor da thread para size
	scanf("%d",&size);

	/* entrada do numero de dados */
	scanf("%d",&nval);
	/* numero de barras do histograma a serem calculadas */
	scanf("%d",&n);

	/* vetor com os dados */
	val = (double *)malloc(nval*sizeof(double));
	vet = (int *)malloc(n*sizeof(int));

	/* entrada dos dados */
	for(i=0;i<nval;i++) {
		scanf("%lf",&val[i]);
	}

	/* calcula o minimo e o maximo valores inteiros */
	min = floor(min_val(val,nval));
	max = ceil(max_val(val,nval));

	/* calcula o tamanho de cada barra */
	h = (max - min)/n;

	

	//ponteiro para a estrutura de argumentos
	pthread_t pthid[size];
	struct my_args *info;
	gettimeofday(&start, NULL);
	for (i = 1; i <= size; ++i) {
		//locação de memória ara o struct
		info = malloc(sizeof(struct my_args));

		//atribuindo valores
		info->min = min;
		info->max = max;
		info->nval = nval;
		info->nbins = n;
		info->h = h;
		info->val = val;
		info->rank = i;
		//criação das threads
		pthread_create(&pthid[i],NULL,count,info);
	}

	//aguarda a finalzação das threads disparadas através do pthid[i]
	for (i = 1; i <= size; ++i) {	
		pthread_join(pthid[i], NULL);
	}

	gettimeofday(&end, NULL);

	duracao = ((end.tv_sec * 1000000 + end.tv_usec) - \
	(start.tv_sec * 1000000 + start.tv_usec));

	printf("%.2lf",min);	
	for(i=1;i<=n;i++) {
		printf(" %.2lf",min + h*i);
	}
	printf("\n");

	/* imprime o histograma calculado */	
	printf("%d",vet[0]);
	for(i=1;i<n;i++) {
		printf(" %d",vet[i]);
	}
	printf("\n");

	/* imprime o tempo de duracao do calculo */
	printf("%lu\n",duracao);

	free(vet);
	free(val);

	return 0;
}



/*
Tarefa Complementar


Tabela 1    
			  Threads      1   2    3     4
	________|___________|____|____|____|_____|
	 		| SpeedUp   |  1 |    |    |     |
	arq1.in |___________|____|____|____|_____|
			|Eficiência |  1 |    |    |     |
	________|___________|____|____|____|_____|
			| SpeedUp   |  1 |    |    |     |
	arq2.in |___________|____|____|____|_____|
			|Eficiência |  1 |    |    |     |
	________|___________|____|____|____|_____|
		    | SpeedUp   |  1 |    |    |     |
	arq3.in |___________|____|____|____|_____|
			|Eficiência |  1 |    |    |     |
	________|___________|____|____|____|_____|
*/
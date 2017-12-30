#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
#include<pthread.h>

#define _GNU_SOURCE

//Has the value of the number of TOSS
unsigned int n = 0;

//Has the value of approximation.
long long unsigned int in = 0;

//Number of threads
int size;

//inicializa um mutex.
pthread_mutex_t mutex_p = PTHREAD_MUTEX_INITIALIZER;

void* monte_carlo_pi(void* thread_id) {
	
    long long unsigned int i;
	double x, y, d;
	long long unsigned int in_local = 0;

    //Indentificador único da thread
	long my_id = (long) thread_id;
	unsigned int seed = my_id;


    //calcula o número de lançamentos para cada thread.
    int toss_number = n/size;


    // Calcula o range inicial e final para uma determinada thread "thread_id".
    int beg_interval = my_id * toss_number;
    int end_interval = ((my_id + 1) * toss_number)-1;

	for (i = beg_interval; i < end_interval; i++) {

		x = ((rand_r(&seed) % 1000000)/500000.0)-1;
		y = ((rand_r(&seed) % 1000000)/500000.0)-1;
		d = ((x*x) + (y*y));
		
		if (d <= 1) {	
			in_local+=1;
		}
	}

	//Região Crítica. Soma Global
	pthread_mutex_lock (&mutex_p);
	in += in_local;
	pthread_mutex_unlock (&mutex_p);

	return NULL;
}

int main(void) {
    
	double pi;
	long unsigned int duracao;
	struct timeval start, end;
    long thread;

	scanf("%d %u",&size, &n);

    // cria thread pool.
    pthread_t thread_handles[size];

	//Armazena o horário de ínicio da execução do programa
	gettimeofday(&start, NULL);
	
    // Cria as threads e executa a funação Count. O último parâmetro da função representa o id da thread que irá executá-la
    for (thread = 0; thread < size; thread++) {
        pthread_create(&thread_handles[thread], NULL, monte_carlo_pi, (void *) thread);
    }
	
    // Finaliza as threads. A thread principal irá continuar quando todas as outras threads tiverem sido finalizadas.
    for (thread = 0; thread < size; thread++) {
        pthread_join(thread_handles[thread], NULL);
    }

	//Armazena o horário de Finalização da execução do programa
	gettimeofday(&end, NULL);

	//Subtraí o horário incial do horário final de execução do programa para obter a duração em Microsegundos.
	duracao = ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec));

	pi = 4*in/((double)n);
	printf("%lf\n%lu\n",pi,duracao);

	//Destroi todos os mutex antes de finalizar o programa, liberando assim qualquer thread que esteja adormecida.
	pthread_mutex_destroy(&mutex_p);
	return 0;
}


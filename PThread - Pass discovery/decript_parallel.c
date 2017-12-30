#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <pthread.h>

//Number of threads
int nt;
short pass_found = 0;
char filename[100];
FILE *popen(const char *command, const char *type);

double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

void * find_password_zip (void * thread_num) {

  FILE * fp;
  char finalcmd[300] = "unzip -P%d -t %s 2>&1 &";
  long my_thread = (long) thread_num;
  char ret[200];
  char cmd[400];
  int i;

  if(my_thread == 0)
     return NULL;

  // Entende-se que o melhor tempo para encontrar a senha dependerá do tipo de política que as thread terão em sua execução. Nesse algoritmo, a execução das threads é uniforme, cada thread inicia executando seu número e depois os números referentes a uma progressão aritimética com fator NT (Número de Threads).
  for(i = my_thread; i < 500000; i = i + nt - 1){
    
    //Caso qualquer uma das threads tenha encontrado a senha, essa atualiza a variável global pass_found com valor 1 (true), o que faz com que as threads retornem da função em suas próximas iterações.
    if(pass_found) {
      return NULL;
    }

     sprintf((char*)&cmd, finalcmd, i, filename);
     //printf("Comando a ser executado: %s -- thread: %d\n", cmd, my_thread);
     fp = popen(cmd, "r");
      
     while (!feof(fp)) {
        fgets((char*)&ret, 200, fp);
        if (strcasestr(ret, "ok") != NULL) {
           printf("Senha:%d\n", i);
           pass_found = 1;
        }
	    }
	    pclose(fp);
  } 
  return NULL;
}

int main ()
{
  double t_start, t_end;
  long thread;

  scanf("%d", &nt);
  scanf("%s", filename);
 
  nt++;
  pthread_t thread_handles[nt];
  t_start = rtclock();

   // Cria as threads para a execução da função "find_password".
  for (thread = 0; thread < nt; thread ++ ) {
    pthread_create(&thread_handles[thread], NULL, find_password_zip, (void *) thread);
  }

   // Após as threads retornarem da função, são finalizadas através da função pthread_join.
  for (thread = 0; thread < nt; thread ++ ) {   
     pthread_join(thread_handles[thread], NULL);
  }

  t_end = rtclock();
 
  fprintf(stdout, "%0.6lf\n", t_end - t_start);  
}


/**
Andrius Henrique Sperque
RA:189918
 
Poderia também considerar o meu programa do desafio que se encontra no final deste arquivo comentado. Infelizmente não consegui testá-lo a tempo e pegar todos os resultados. 
 
Resultados:

===============================
Arq1.in
Serial:
	Senha:10000
	76.364680
Paralelo:
	Senha:10000
	18.703530

===============================
Arq2.in
Serial:
	Senha:100000
	772.975867
Paralelo:
	Senha:100000
	293.361068

===============================
 Arq3.in
 Serial:
    Senha:450000
    2124.293743
 Paralelo:
	Senha:450000
    900.326297
 
===============================
 Arq4.in
 Serial:
    Senha:310000
    1650.359274
 
 Paralelo:
	Senha:310000
    615.007147

===============================
 Arq5.in
 Serial:
	Senha:65000
	344.003924
	
 Paralelo:
	Senha:65000
    128.853789

===============================
 Arq6.in
 Serial:
	Senha:245999
	1339.064735
	
 Paralelo:
	Senha:245999
	705.166302
	
*/
/**
 
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <pthread.h>

//Number of threads
int nt;
short pass_found = 0;
char filename[100];
FILE *popen(const char *command, const char *type);

//int buffer;

#define BUFFER_SIZE 10000
#define TRUE 1

//Estruta de dados para controle de sincronismo e regisão critica
typedef struct {
    int buf[BUFFER_SIZE];
    size_t len;
    pthread_cond_t can_consume;
    pthread_mutex_t mutex; // Mutex para adicionar e remover do buffer
    pthread_cond_t can_produce;
} buffer_t;

buffer_t buffer2 = {
    .len = 0,
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .can_produce = PTHREAD_COND_INITIALIZER,
    .can_consume = PTHREAD_COND_INITIALIZER
};

buffer_t * buffer = (buffer_t *)&buffer2;

double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

void * control_other_threads (void *arg) {
    
    buffer_t *buffer = (buffer_t*)arg;
    int i;
    
    for (i = 9000; i <= 500000; i++) {
        
        if(pass_found) {
            //Envia sinal broadcast para todas as threads acordarem e retornarem da função.
            pthread_cond_broadcast(&buffer->can_consume);
            return NULL;
        }
        
        pthread_mutex_lock(&buffer->mutex);
        if(buffer->len == BUFFER_SIZE) {
            
            // Com o buffer cheio, deve-se esperar até alguns números sejam consumidos.
            pthread_cond_wait(&buffer->can_produce, &buffer->mutex);
        }
        
        // printf("Produced: %d\n", i);
        buffer->buf[buffer->len] = i;
        ++buffer->len;
        
        // signal the fact that new items may be consumed
        pthread_cond_signal(&buffer->can_consume);
        pthread_mutex_unlock(&buffer->mutex);
    }
    return NULL;
}

// Consume the numbers that are generated by the producer thread.
void* consumer(void *arg) {
    long threadId  = (long) arg;
    
    FILE * fp;
    char finalcmd[300] = "unzip -P%d -t %s 2>&1 &";
    char ret[200];
    char cmd[400];
    
    while(1) {
        
        pthread_mutex_lock(&buffer->mutex);
        
        while(buffer->len == 0) {
            
            if(pass_found) {
                printf("finish %d", threadId);
                return NULL;
            }
            // wait for new items to be appended to the buffer
            pthread_cond_wait(&buffer->can_consume, &buffer->mutex);
        }
        
        if(pass_found) {
            printf("finish %d", threadId);
            return NULL;
        }
        --buffer->len;
        int value = buffer->buf[buffer->len];
        
        sprintf((char*)&cmd, finalcmd, value, filename);
        //printf("Comando a ser executado: %s -- thread: \n", cmd);
        
        fp = popen(cmd, "r");
        while (!feof(fp)) {
            fgets((char*)&ret, 200, fp);
            if (strcasestr(ret, "ok") != NULL) {
                
                // Se a senha foi encontrada, então atualiza variável global.
                pass_found = TRUE;
                printf("Senha:%d\n", value);
                
                // Desbloqueia a thread que está no controler para se finalizar e enviar sinal broadcast para todas as outras se finalizarem.
                pthread_mutex_unlock(&buffer->mutex);
                pthread_cond_signal(&buffer->can_produce);
                return NULL;
            }
        }
        pclose(fp);
        pthread_cond_signal(&buffer->can_produce);
        pthread_mutex_unlock(&buffer->mutex);
    }
    
    return NULL;
}


int main ()
{
    
    double t_start, t_end;
    long thread;
    
    scanf("%d", &nt);
    scanf("%s", filename);
    
    
    // Cria uma thread a mais para controlar as outras threads.
    nt++;
    pthread_t thread_handles[nt];
    
    t_start = rtclock();
    
    // Inicializa a thread de controle.
    pthread_create(&thread_handles[0], NULL, control_other_threads, (void *) &buffer2);
    
    // Inicializa a thread para consumir os dados produzidos.
    for (thread = 1; thread < nt; thread ++ ) {
        pthread_create(&thread_handles[thread], NULL, consumer, (void *) thread);
    }
    
    // End threads, but just in case the password can't be found.
    for (thread = 0; thread < nt; thread ++ ) {
        pthread_join(thread_handles[thread], NULL);
    }
    
    t_end = rtclock();
    
    fprintf(stdout, "%0.6lf\n", t_end - t_start);
}
*/

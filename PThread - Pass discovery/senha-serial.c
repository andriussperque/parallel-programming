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

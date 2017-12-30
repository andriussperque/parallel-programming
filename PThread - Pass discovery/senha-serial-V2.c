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

int buffer;

double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

void * control_other_threads () {

    int i;
    for (i = 0; i < 500000; i++) {
      
      if (thread) {

      

    }

    return NULL;

}

void * find_password_zip (void * thread_num) {

  FILE * fp;
  char finalcmd[300] = "unzip -P%d -t %s 2>&1";
  long my_thread = (long) thread_num;
  char ret[200];
  char cmd[400];
  int i;

  for(i = my_thread; i < 500000; i += (int)my_thread){
    
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

  // add another thread in order to control the others.
  nt+=1;
  queue = (int *) malloc (nt * sizeof(int));
  pthread_t thread_handles[nt];  
  
  t_start = rtclock();

  // Initialize Producer (manager) Thread
  pthread_create(&thread_handles[0], NULL, control_other_threads, (void *) 0);

  // Initialize Consumer Threads 
  for (thread = 1; thread < nt; thread ++ ) {
      pthread_create(&thread_handles[thread], NULL, find_password_zip, (void *) thread);
  }

   // End threads, but just in case the password can't be found.
  for (thread = 0; thread < nt; thread ++ ) {   
     pthread_join(thread_handles[thread], NULL);
  }

  t_end = rtclock();
 
  fprintf(stdout, "%0.6lf\n", t_end - t_start);
}

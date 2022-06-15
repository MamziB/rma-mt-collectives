/*
 *  RMA-Based MPI All to All
 *
 *  Receive buffer is realized by the window, incoming data
 *  is stored in window only
 *
 *  Threads are utilized to to break down the messages and have
 *  higher parallelism
 *
 * */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>

/* USERS  OPTIONS */
#define MSG_SIZE    1966080
#define THREAD_NUM  16
#define ITER        200
#define WARMUP      20
#define USE_FETCH   1 /* use atmoc fetch-ops to realize the  put completion */
#define FORCE_FENCE 1 /* call win fence even if fetch-ops is used for put completion */
#define MAX_NPROCS  64
/* use device memory for send buffers and window buffers.
                         Comment out if not needed  */
#define CUDA_ENABLED

//#define MESSAGE_PIPELINING
/* do not chunk the message instead, give subset
of targets to each thread */

//#define DEBUG_TIMERS

#define VALIDATION
/* data validation */

/* END OF USERS  OPTIONS */

#define FETCH_SZ    4 * MAX_NPROCS

#ifdef CUDA_ENABLED
#include "cuda.h"
#include "cuda_runtime.h"
#endif


typedef struct arg_struct {
    int tid;
    MPI_Win *window;
    void *sendbuf;
    void *winbuf;
    int msg_size;
    int my_rank;
    int comm_size;
#ifdef CUDA_ENABLED
    int winbuf_size;
    int sendbuf_size;
#endif
} struct_args;


#ifdef DEBUG_TIMERS
double t_barrier_lat=0;
double wallclock=0, fence_time=0, fetch_time=0, put_time=0, start_puts=0,
       start_fetch=0, start_fence=0, start_validation=0, validation_time=0;
#endif
double total=0;
pthread_barrier_t barrier;
volatile int thread_wait_counter1=0, thread_wait_counter2=0;
pthread_mutex_t count_mutex;

static inline void thread_barrier(int tid, int enable_timer) {

#ifdef DEBUG_TIMERS
    double start;
    if (!tid && enable_timer) {
        start = MPI_Wtime();
    }
#endif
    pthread_barrier_wait(&barrier); 
#if 0
    pthread_mutex_lock(&count_mutex);
    thread_wait_counter1++;
    pthread_mutex_unlock(&count_mutex);

    if (!tid) {
        while(thread_wait_counter1 != THREAD_NUM) {};
        thread_wait_counter1 = 0;
    } else {
        while(thread_wait_counter1 != 0) {}; 
    }

    pthread_mutex_lock(&count_mutex);
    thread_wait_counter2++;
    pthread_mutex_unlock(&count_mutex);

    if (!tid) {
        while(thread_wait_counter2 != THREAD_NUM) {};
        thread_wait_counter2 = 0;
    } else {
        while(thread_wait_counter2 != 0) {}; 
    }

    //printf("got out tid=%d\n", tid);
#endif

#ifdef DEBUG_TIMERS
    if (!tid && enable_timer) {
        t_barrier_lat += MPI_Wtime() - start;
    }
#endif
}

void * do_alltoall(void * args ) {
    struct_args * input = (struct_args *)args;
    int target =0, send_disp = 0, recv_disp = 0;
    int tid = input->tid;
    char val;
    double start=0;
    int fetch=1, result=0;
#ifdef CUDA_ENABLED
    char *tmpbuf;
    int ret = 0;
    if (!tid) {
        tmpbuf = malloc(input->winbuf_size);
    }
#endif

    thread_barrier(tid, 0);

    if (!tid) {
#ifdef CUDA_ENABLED
        memset(tmpbuf, 0, input->winbuf_size);
        ret = cudaMemset(input->winbuf, 0, input->winbuf_size);
        if (ret != cudaSuccess) {
            fprintf(stderr, "cudaMemset failed"); exit(1);
        }
#else
        memset(input->winbuf, 0, sizeof(input->winbuf));
#endif
        MPI_Barrier(MPI_COMM_WORLD);
    }
    thread_barrier(tid, 0);

#ifdef DEBUG_TIMERS
    int start_wallclock=0;
    if (!tid) {
        start_wallclock = MPI_Wtime();
    }
#endif

#ifdef DEBUG
   fprintf(stderr, "Thread %d started \n", tid);
#endif
    for (int i = 0; i < ITER+WARMUP; i++) {
        if (!tid) {
#ifdef VALIDATION
#ifdef CUDA_ENABLED
            memset(tmpbuf, 0, input->winbuf_size);
            ret = cudaMemset(input->winbuf, 0, input->winbuf_size);
            if (ret != cudaSuccess) {
                fprintf(stderr, "cudaMemset failed"); exit(1);
            }
#else
            memset(input->winbuf, 0, sizeof(input->winbuf));
#endif
#endif
            MPI_Barrier(MPI_COMM_WORLD);
        }

        thread_barrier(tid, 0);

        if (!tid) {
            start = MPI_Wtime();
        }

#ifdef DEBUG_TIMERS
        if (!tid) {
            start_puts = MPI_Wtime();
        }
#endif
#ifdef MESSAGE_PIPELINING
        recv_disp = input->my_rank*input->msg_size;
        for (int peer=0; peer < input->comm_size/THREAD_NUM; peer++) {
            target = (input->my_rank + peer + tid*input->comm_size/THREAD_NUM) % input->comm_size;
            send_disp = target*input->msg_size;
            MPI_Put(input->sendbuf + send_disp, input->msg_size, MPI_CHAR, target,
                    recv_disp, input->msg_size, MPI_CHAR, *input->window);

        }
#else
        recv_disp = input->my_rank*input->msg_size + tid*(input->msg_size/THREAD_NUM);
        for (int peer=0; peer < input->comm_size; peer++) {
            target = (input->my_rank + peer) % input->comm_size;
            send_disp = target*input->msg_size + tid*(input->msg_size/THREAD_NUM);
            MPI_Put(input->sendbuf + send_disp, input->msg_size/THREAD_NUM, MPI_CHAR, target,
                    recv_disp, input->msg_size/THREAD_NUM, MPI_CHAR, *input->window);

        }
#endif
#ifdef DEBUG_TIMERS
        if (!tid && i >= WARMUP) {
            put_time += MPI_Wtime() - start_puts;
        }
#endif

        (i >= WARMUP)?thread_barrier(tid, 1):thread_barrier(tid, 0);

#ifdef DEBUG_TIMERS
        if (!tid) {
            start_fetch = MPI_Wtime();
        }
#endif
#ifdef MESSAGE_PIPELINING
        if (USE_FETCH) {
            for (int peer=0; peer < input->comm_size/THREAD_NUM; peer++) {
                target = (input->my_rank + peer + tid*input->comm_size/THREAD_NUM) % input->comm_size;
                MPI_Fetch_and_op(&fetch, &result, MPI_INT, target, input->msg_size *
                        input->comm_size * sizeof(char) + input->my_rank, MPI_SUM,
                        *input->window);

            }

        }
#else
        if (!tid && USE_FETCH) {
            for (int peer=0; peer < input->comm_size; peer++) {
                target = (input->my_rank + peer) % input->comm_size;
                MPI_Fetch_and_op(&fetch, &result, MPI_INT, target, 0 /*input->msg_size *
                        input->comm_size * sizeof(char) + input->my_rank*/, MPI_NO_OP,
                        *input->window);

            }
        }
#endif
#ifdef DEBUG_TIMERS
        if (!tid && i >= WARMUP) {
            fetch_time += MPI_Wtime() - start_fetch;
        }
#endif
        if (USE_FETCH) {
            (i >= WARMUP)?thread_barrier(tid, 1):thread_barrier(tid, 0);
        }

        if (!tid) {
#ifdef DEBUG_TIMERS
        if (!tid) {
            start_fence = MPI_Wtime();
        }
#endif
            if (!USE_FETCH || FORCE_FENCE) {
                MPI_Win_fence(0, *input->window);
            } else {
                MPI_Barrier(MPI_COMM_WORLD);
            }
#ifdef DEBUG_TIMERS
        if (!tid && i >= WARMUP) {
            fence_time += MPI_Wtime() - start_fence;
        }
#endif

            if (i >= WARMUP)
                total += MPI_Wtime() - start;


#ifdef VALIDATION
#ifdef DEBUG_TIMERS
        if (!tid) {
            start_validation = MPI_Wtime();
        }
#endif
            /* data validation */
#ifdef CUDA_ENABLED
            ret = cudaMemcpy(tmpbuf, input->winbuf, input->winbuf_size,
                    cudaMemcpyDeviceToHost);
            if (ret != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed"); exit(1);
            }
#endif
            for (int j=0; j < input->comm_size; j++) {
                for (int k=0; k < input->msg_size; k++) {
                    val = j;
#ifdef CUDA_ENABLED
                    if (tmpbuf[j*input->msg_size+k] != (char)j)
#else
                    if (((char *)input->winbuf)[j*input->msg_size+k] != (char)j)
#endif
                    {
                        fprintf(stderr, "validation failed (iter=%d,k=%d,j=%d != %d)\n"
                                , i, k, j,
#ifdef CUDA_ENABLED
                                tmpbuf[j*input->msg_size+k]
#else
                                ((char *)input->winbuf)[j*input->msg_size+k]
#endif
                                );
                        exit(1);
                    }
                }
            }
#ifdef DEBUG_TIMERS
        if (!tid && i >= WARMUP) {
            validation_time += MPI_Wtime() - start_validation;
        }
#endif
            /* end of data validation */
#endif
            MPI_Barrier(MPI_COMM_WORLD);
        }
        thread_barrier(tid, 0);
    }
#ifdef DEBUG_TIMERS
    if (!tid) {
        wallclock = MPI_Wtime() - start_wallclock;
    }
#endif
#ifdef CUDA_ENABLED
    if (!tid) {
        free(tmpbuf);
    }
#endif
}


int main(int argc, char* argv[])
{
    int comm_size;
    int my_rank;
    int errors=0, msg_size = MSG_SIZE;
    MPI_Request request;
    char cuda_aware = 0;
    char message_pipelining = 0;
#ifdef CUDA_ENABLED
    cuda_aware = 1;
    char *winbuf = NULL;
    char *sendbuf = NULL;
    int winbuf_size = MSG_SIZE * MAX_NPROCS + FETCH_SZ;
    int sendbuf_size = MSG_SIZE * MAX_NPROCS;
    cudaMalloc((void **)(&winbuf), winbuf_size);
    cudaMemset(winbuf, 0, winbuf_size);
    cudaMalloc((void **)(&sendbuf), sendbuf_size);
    cudaMemset(sendbuf, 0, sendbuf_size);
#else
    char winbuf[MSG_SIZE * MAX_NPROCS + FETCH_SZ] = {0};
    char sendbuf[MSG_SIZE * MAX_NPROCS] = {0};
#endif

    MPI_Win window;

    pthread_t threads[THREAD_NUM];
    struct_args inputs[THREAD_NUM];

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    assert(comm_size <= MAX_NPROCS && msg_size <= MSG_SIZE);
    assert(msg_size % THREAD_NUM == 0);
#ifdef MESSAGE_PIPELINING
    message_pipelining = 1;
    assert(comm_size % THREAD_NUM == 0); /* TODO remove this requirement */
#endif

    pthread_barrier_init(&barrier, NULL, THREAD_NUM);

    /* Create window with enough size to hold entire recv buf of alltoall (here
     * window buffer act as recv buf in MPI_Alltoall */
    MPI_Win_create(winbuf, MSG_SIZE * comm_size * sizeof(char) + FETCH_SZ, sizeof(char),
            MPI_INFO_NULL, MPI_COMM_WORLD, &window);
    MPI_Win_fence(0, window);


    char val = my_rank;

#ifdef CUDA_ENABLED
    cudaMemset(sendbuf, val, sendbuf_size);
#else
    memset(sendbuf, val, sizeof(sendbuf));
#endif


    for (int tid=0; tid < THREAD_NUM; tid++) {
        inputs[tid].tid     = tid;
        inputs[tid].window  = &window;
        inputs[tid].sendbuf = sendbuf;
        inputs[tid].winbuf  = winbuf;
        inputs[tid].msg_size    = msg_size;
        inputs[tid].my_rank     = my_rank;
        inputs[tid].comm_size   = comm_size;
#ifdef CUDA_ENABLED
        inputs[tid].winbuf_size= winbuf_size;
        inputs[tid].sendbuf_size= sendbuf_size;
#endif

        pthread_create(&threads[tid], NULL, do_alltoall, (void *) &inputs[tid]);
    }

    for (int tid=0; tid < THREAD_NUM; tid++) {
        pthread_join(threads[tid], NULL);
#ifdef DEBUG
        fprintf(stderr, "Main thread join thread %d \n", tid);
#endif

    }

    if (my_rank == 0) {
        double tmp = msg_size / 1e6 * ITER * comm_size * comm_size;
        double bw  = tmp / total;
        printf("CUDA Aware:\t%s \nMsg Pipelining:\t%s \nMsg Size:\t%d \nComm Size:\t%d \n"
                "Threads:\t\t%d\nUsing Fetch-ops:\t%s \nBandwidth:\t%.0f MB/s\n",
                (cuda_aware?"Yes":"No"), (message_pipelining?"Yes":"No"), msg_size,
                comm_size, THREAD_NUM, USE_FETCH?"Yes":"No", bw);
#ifdef DEBUG_TIMERS

        printf("================== PROFILING TIMERS ===================\n");
        printf("Wallclock Time:\t%.2fms   \nAlltoall Time:\t%.2fms  \nBarrier Time:\t%.2fms   \n"
            "Fetch Time:\t%.2fms   \nPut Time:\t%.2fms   \nFence Time:\t%.2fms  \nCheck Time:\t%.2fms\n",
                wallclock*1e3, total*1e3, t_barrier_lat*1e3,
                fetch_time*1e3, put_time*1e3, fence_time*1e3, validation_time*1e3);
        printf("=======================================================\n");
#endif
    }

    MPI_Win_fence(0, window);
    MPI_Win_free(&window);

end:

    pthread_barrier_destroy(&barrier);
    MPI_Finalize();

    return EXIT_SUCCESS;
}

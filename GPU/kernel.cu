
#include "cuda_runtime.h"
#include "RandomString.h"
#include "device_launch_parameters.h"
#include "sha256.h"
#include <string>
#include <math.h> 
#include <iostream>
#include <stdio.h>
#include <chrono>

#define BLOCKS 1

#define LEN 5
#define DIVISIONS 1024
#define BASE 74
#define BOTTOM 48
#define TOP 122


using std::string;
using std::cout;
using std::endl;

cudaError_t crackWithCuda(int len, int base, int div, int bottom, int top, int* password, int* quit, int** arr);

int** preprocess(int len, int base, int div, int bottom, int top);

__device__ bool checkArrays(int size, int* password, int* guess) {
    for (int i = 0; i < size; ++i) 
        if (password[i] != guess[i]) return false;   

    return true;
}

__global__ void crackKernal(int len, int base, int top, int bottom, int* quit, int* password, int** ranges) {
    
    int idx = 2 * (threadIdx.x + blockIdx.x * blockDim.x);

    int* bot_arr = ranges[idx];
    int* top_arr = ranges[idx + 1];

    int i = 0;    

    if (checkArrays(len, password, bot_arr)) {
        quit[0] = 1;
        printf("Password found: ");
        for (int a = 0; a < len; ++a) {
            printf("%c", (char)bot_arr[a]);
        }
        printf("\n");

        printf("Password code: ");
        for (int a = 0; a < len; ++a) {
            printf("%d, ", bot_arr[a]);
        }
        printf("\n");

        printf("Found by: %d\n", idx / 2 + 1);
        goto DONE;
    }

    do {
        for (i = 0; i < len; ) {
            if (bot_arr[i] < top) {
                bot_arr[i]++;
                break;
            }
            else {
                bot_arr[i] = bottom;
                ++i;
            }
        }

        if (checkArrays(len, password, bot_arr)) {
            quit[0] = 1;
            printf("Password found: ");
            for (int a = 0; a < len; ++a) {
                printf("%c", (char)bot_arr[a]);
            }
            printf("\n");

            printf("Password code: ");
            for (int a = 0; a < len; ++a) {
                printf("%d, ", bot_arr[a]);
            }
            printf("\n");

            printf("Found by: %d\n", idx / 2 + 1);
        }

    } while (quit[0] == 0 && !checkArrays(len, password, bot_arr) && !checkArrays(len, top_arr, bot_arr));

DONE:
    
    
}


int main()
{
    RandomString RS(LEN);

    cout << endl;
    cout << "Password: " << RS.getPassword() << endl;
    //cout << "Password Hash: " << RS.getHashPassword() << endl;
    cout << endl;
    
    int* passwordInt = RS.convertToIntArr(RS.getPassword(), LEN); 
    int* quit = new int[1];
    quit[0] = 0;

    int** ranges = preprocess(LEN, BASE, DIVISIONS, BOTTOM, TOP);
    

    auto start = std::chrono::high_resolution_clock::now();
    cudaError_t cudaStatus = crackWithCuda(LEN, BASE, DIVISIONS, BOTTOM, TOP, passwordInt, quit, ranges);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);

    std::cout << " Time: " << duration.count() << "ns (nanoseconds)" << std::endl;
    std::cout << " Time: " << duration.count() / 1000000.0 << " (milliseconds)" << std::endl;
    std::cout << " Time: " << duration.count() / 1000000000.0 << " (seconds)" << std::endl;

    
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "crackWithCuda failed!");
        return 1;
    }


    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    delete quit;
    delete[] ranges;
    delete passwordInt;
    return 0;
}


/// <summary>
/// This function...
/// </summary>
/// <param name="len">: The length of the password.</param>
/// <param name="base">: The number of different symbols used in the password.</param>
/// <param name="div">: The number of subdivisions to take of the password.</param>
/// <param name="bottom">: The lowest code from the symbols.</param>
/// <param name="top">: The highest code from the symbols.</param>
/// <returns></returns>
int** preprocess(int len, int base, int div, int bottom, int top) {

    unsigned long long* base10 = new unsigned long long[2*div];
    
    int** baseB = new int*[2 * div];
    for (int j = 0; j < 2 * div; ++j)
        baseB[j] = new int[len];

    unsigned long long total_permutations = (unsigned long long) pow(base, len);
    unsigned long long subdiv = total_permutations / div;

    base10[0] = 0;

    int i;
    for (i = 1; i < 2*div-1; i+=2) {
        base10[i] = base10[i - 1] + subdiv;
        base10[i + 1] = base10[i] + 1;
    }
    base10[i] = total_permutations;
    
    /*for (int i = 0; i < 2 * div; ++i) {
        cout << base10[i] << ", ";
    }
    cout << std::endl;*/

    for (i = 0; i < len; ++i) {
        baseB[0][i] = bottom;
        baseB[2 * div-1][i] = top;
    }

    
    unsigned long long k;
    int l;
    for (i = 1; i < 2 * div-1; ++i) {
    
        k = base10[i];
        for (int j = 0; j < len; ++j) {
            
            l = k % base;
            k = (k - l) / base;
            
            baseB[i][j] = l+bottom;
        }
    }

    /*for (i = 0; i < 2 * div; ++i) {
        for (int j = 0; j < len; ++j) {
            cout << baseB[i][j] << ", ";
        }
        cout << std::endl;
    }
    cout << std::endl;*/

    return baseB;
    //delete[] baseB;
    delete base10;
}

cudaError_t crackWithCuda(int len, int base, int div, int bottom, int top, int* password, int* quit, int** arr)
{
    int rangeSize = (2 * div);
    int* dev_password = 0;
    int** dev_ranges = 0;
    int* dev_quit = 0;
    
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_quit, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_password, len * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_ranges, sizeof(int*) * rangeSize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // temp array of device pointers
    int** temp_d_ptrs = (int**)malloc(sizeof(int*) * rangeSize);
    for (int i = 0; i < rangeSize; i++) {
        cudaStatus = cudaMalloc((void**)&temp_d_ptrs[i], sizeof(int) * len);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(temp_d_ptrs[i], arr[i], sizeof(int) * len, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }
    }
    cudaStatus = cudaMemcpy(dev_ranges, temp_d_ptrs, sizeof(int*) * rangeSize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_password, password, len * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_quit, quit, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    


    // Launch a kernel on the GPU.
    crackKernal<<<BLOCKS, DIVISIONS>>>(LEN, BASE, TOP, BOTTOM, dev_quit, dev_password, dev_ranges);

    



    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    /*cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }*/

Error:
    for (int i = 0; i < rangeSize; i++) {
        cudaFree(temp_d_ptrs[i]);
    }
    cudaFree(dev_password);
    cudaFree(dev_ranges);
    cudaFree(dev_quit);
    free(temp_d_ptrs);

    return cudaStatus;
}










// Links
// https://stackoverflow.com/questions/46555270/cuda-copying-an-array-of-arrays-filled-with-data-from-host-to-device


// Known Problems with this version
/*

1.) No hash function used by threads
    - Couldn't find implementation online.

2.) No stopping of other threads when the password is found (which leads to inaccurate timing).
    - SOLVED

3.) Sometimes doesn't work and loops forever (Unknown cause).
    - Perhaps this is caused by incorrect indexing of the threads and the ranges assigned to them.






*/

// Old stuff

/*
    while (!checkArrays(len, password, bot_arr)) {

        // Reached the end of the search for this thread.
        if (checkArrays(len, top_arr, bot_arr) || doneFlag != 0) {
            printf("Stopped %d\n", idx/2+1);
            flag = false;
            break;
        }

        for (i = 0; i < len; ) {
            if (bot_arr[i] < top) {
                bot_arr[i]++;
                break;
            }
            else {
                bot_arr[i] = bottom;
                ++i;
            }
        }

        printf("Guess(%d): ", idx / 2 + 1);
        for (int a = 0; a < len; ++a) {
            printf("%d, ", bot_arr[a]);
        }
        printf("\n");
    }*/
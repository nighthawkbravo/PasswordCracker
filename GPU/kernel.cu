
#include "cuda_runtime.h"
#include "RandomString.h"
#include "device_launch_parameters.h"
#include "sha256.h"
#include <string>
#include <math.h> 
#include <iostream>
#include <stdio.h>

#define LEN 3
#define BASE 74
#define DIVISIONS 5
#define BOTTOM 48
#define TOP 122





using std::string;
using std::cout;
using std::endl;

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

cudaError_t crackWithCuda(int len, int base, int div, int bottom, int top, int* password, int** arr);

int** preprocess(int len, int base, int div, int bottom, int top);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    
    RandomString RS(LEN);

    cout << endl;
    cout << "Password: " << RS.getPassword() << endl;
    //cout << "Password Hash: " << RS.getHashPassword() << endl;
    cout << endl;
    
    int* passwordInt = RS.convertToIntArr(RS.getPassword(), LEN);    

    
    int** ranges = preprocess(LEN, BASE, DIVISIONS, BOTTOM, TOP);
    cudaError_t cudaStatus = crackWithCuda(LEN, BASE, DIVISIONS, BOTTOM, TOP, passwordInt, ranges);
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

    delete[] ranges;
    delete passwordInt;
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
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



cudaError_t crackWithCuda(int len, int base, int div, int bottom, int top, int* password, int** arr)
{
    int rangeSize = (2 * div);
    int* dev_password = 0;
    int** dev_ranges = 0;
    
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
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
    


    // Launch a kernel on the GPU with one thread for each element.
    //addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

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
    free(temp_d_ptrs);

    return cudaStatus;
}


// Links



// https://stackoverflow.com/questions/46555270/cuda-copying-an-array-of-arrays-filled-with-data-from-host-to-device
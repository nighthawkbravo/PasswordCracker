all: passwordCrackerGPU
.PHONY: all

# CC compiler options:
CC=g++

# NVCC compiler options:
NVCC=nvcc

sha256.o: sha256.cpp sha256.h
	$(CC) -c sha256.cpp

randomString.o: randomString.cpp randomString.h
	$(CC) -c randomString.cpp

kernel.o: kernel.cu
	$(NVCC) -c kernel.cu

passwordCrackerGPU: sha256.o randomString.o kernel.o
	$(CC) -o passwordCrackerGPU randomString.o sha256.o kernel.o

clean:
	rm passwordCrackerGPU kernel.o sha256.o randomString.o
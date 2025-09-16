# Compiler and flags
CC = gcc
ICC = icc
MPICC = mpicc
MPIICC = mpiicc
CFLAGS = -O3 -fopenmp -std=c99
ICFLAGS = -O3 -qopenmp -std=c99

all: gccnearly iccnearly gcccomplete icccomplete

# GCC Nearly-Serial Implementation
gccnearly: k-folds.c knnomp.c file-reader.c
	$(CC) $(CFLAGS) k-folds.c knnomp.c file-reader.c -o k-folds-gcc -lm

# Intel ICC Nearly-Serial Implementation
iccnearly: k-folds.c knnomp.c file-reader.c
	$(ICC) $(ICFLAGS) k-folds.c knnomp.c file-reader.c -o k-folds-icc -lm

# GCC Distributed Implementation
gcccomplete: mpi-k-folds.c knnomp.c file-reader.c
	$(MPICC) $(CFLAGS) mpi-k-folds.c knnomp.c file-reader.c -o k-folds-complete-gcc -lm

# Intel ICC Distributed Implementation
icccomplete: mpi-k-folds.c knnomp.c file-reader.c
	$(MPIICC) $(ICFLAGS) mpi-k-folds.c knnomp.c file-reader.c -o k-folds-complete-icc -lm

# Clean target
clean:
	rm -f k-folds-gcc k-folds-icc k-folds-complete-gcc k-folds-complete-icc

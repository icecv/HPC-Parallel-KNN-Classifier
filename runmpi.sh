#!/bin/bash

#SBATCH --job-name=knn_gcc_icc_test         
#SBATCH --output=knn_gcc_icc_output.txt     
#SBATCH --error=knn_gcc_icc_error.txt       
#SBATCH --time=02:00:00                     
#SBATCH --partition=lowpriority                 
#SBATCH --mem=32G                          
#SBATCH --ntasks=32                        
#SBATCH --cpus-per-task=8
#SBATCH --nodes=8


# load moudle
module load gcc
module load intel
module load openmpi

DATASET="your_data.csv"
OUTPUT="output.csv"
K=3
NUM_FOLDS=10

# Change Directory and use makefile 
cd $SLURM_SUBMIT_DIR

echo "Compiling with Makefile..."
make gccnearly
make iccnearly
make gcccomplete
make icccomplete

GCC_RUNTIME_LOG="runtime_results_gcc.txt"
ICC_RUNTIME_LOG="runtime_results_icc.txt"
echo "Runtime Results for GCC" > $GCC_RUNTIME_LOG
echo "Runtime Results for ICC" > $ICC_RUNTIME_LOG

# loop testing different threads
echo "Testing Serial Implementation with OpenMP Threads"
for THREADS in 1 2 4 8; do
    export OMP_NUM_THREADS=$THREADS

    #run gcc and record time
    echo "Running Serial (GCC) with $THREADS threads..."
    START=$(date +%s)
    ./k-folds-gcc $DATASET $OUTPUT $K $NUM_FOLDS
    END=$(date +%s)
    RUNTIME=$((END - START))
    echo "Serial with $THREADS threads: $RUNTIME seconds" >> $GCC_RUNTIME_LOG
    
    #run icc and record time
    echo "Running Nearly (ICC) with $THREADS threads..."
    START=$(date +%s)
    ./k-folds-icc $DATASET $OUTPUT $K $NUM_FOLDS
    END=$(date +%s)
    RUNTIME=$((END - START))
    echo "Serial with $THREADS threads: $RUNTIME seconds" >> $ICC_RUNTIME_LOG
done

FASTEST_THREADS=8
cd $SLURM_SUBMIT_DIR

# loop testing different process
echo "Testing Distributed Implementation with MPI Processes"
for PROCESSES in 1 2 4 8 16 32; do
    export OMP_NUM_THREADS=$FASTEST_THREADS

    #run gcc and record time
    echo "Running Distributed (GCC) with $PROCESSES processes..."
    START=$(date +%s)
    mpirun -np $PROCESSES ./k-folds-complete-gcc $DATASET $OUTPUT $K $NUM_FOLDS
    END=$(date +%s)
    RUNTIME=$((END - START))
    echo "Distributed with $PROCESSES processes: $RUNTIME seconds" >> $GCC_RUNTIME_LOG
    
    #run icc and record time
    echo "Running Distributed (ICC) with $PROCESSES processes..."
    START=$(date +%s)
    mpirun -np $PROCESSES ./k-folds-complete-icc $DATASET $OUTPUT $K $NUM_FOLDS
    END=$(date +%s)
    RUNTIME=$((END - START))
    echo "Distributed with $PROCESSES processes: $RUNTIME seconds" >> $ICC_RUNTIME_LOG
done




echo "Experiments completed. Check $GCC_RUNTIME_LOG and $ICC_RUNTIME_LOG for results."


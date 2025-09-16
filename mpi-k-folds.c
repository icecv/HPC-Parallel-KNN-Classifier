#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include<math.h>
#include<omp.h>
#include <mpi.h>
//#include"file-reader.c" -- If you include file-reader.c like this, make sure to remove file-reader.c function definitions just below and do not 

//Remove these if you do #include"file-reader.c"
int readNumOfPoints(char*);
int readNumOfFeatures(char*);
int readNumOfClasses(char *filename);
double *readDataPoints(char*, int, int);
void *writeResultsToFile(double*, int, char*);
extern void knnomp(double *traindata, int trainpointnum, int trainfeatnum, double *testdata, int testpointnum, int testfeatnum, int k, int classnum, int *predictedlabels);
void printArray(void *array, char array_name[500], int m, int n, char type);
double calacy(int *predictedlabels, double *test_data, int test_size, int feature_count);

int main(int argc, char *argv[]){

    printf("\n\n===============STARTING KNN (MPI VERSION)===============\n\n");

    //Initialize MPI and define variables
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size); 
    
    //File reading
    char *inFile = argv[1];
    char *outFile = argv[2];
    int k = atoi(argv[3]);
    int numFolds = atoi(argv[4]);

    int totalNumPoints, numFeatures, numClasses;
    double *originalData;

    //The main process reads file data
    if (rank == 0) {
        totalNumPoints = readNumOfPoints(inFile);
        numFeatures = readNumOfFeatures(inFile);
        numClasses = readNumOfClasses(inFile);
        originalData = readDataPoints(inFile, totalNumPoints, numFeatures);
    }

    //Broadcast the main process data to all processes  
    MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numFolds, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&totalNumPoints, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numFeatures, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numClasses, 1, MPI_INT, 0, MPI_COMM_WORLD);

    //Allocate memory for originaldata
    if (rank != 0){
        originalData = (double*)malloc(totalNumPoints * numFeatures * sizeof(double));
    }

    //Broadcast the entire dataset originaldata to all processes
    MPI_Bcast(originalData, totalNumPoints * numFeatures, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //Array contains the number of points in each fold
    int *pointsInFold = (int*)malloc(numFolds * sizeof(int));
    
    //Calculate the points in the fold and the remainder
    int pointsPerFold = totalNumPoints / numFolds;
    int pointsPerFoldRemainder = totalNumPoints % numFolds;

    //Allocates the number of points in the fold adding the remainder to the last fold. (Can handle unequally sized folds)
    int fold;
    for(fold = 0; fold < numFolds; fold++){
        pointsInFold[fold] = pointsPerFold + ((fold == numFolds - 1) ? pointsPerFoldRemainder : 0);
    }
    
    //Calculate and store the starting index of each fold in the dataset for divide the dataset into training and testing sets
    int *foldStartIndex = (int*)malloc(numFolds * sizeof(int));
    {
        int prefix = 0;
        for(int i = 0; i < numFolds; i++){
            foldStartIndex[i] = prefix;
            prefix += pointsInFold[i];
        }
    }
                
    size_t maxTrainBytes  = (totalNumPoints - pointsInFold[0]) * numFeatures * sizeof(double);
    size_t maxTestBytes   =  pointsInFold[numFolds - 1] * numFeatures * sizeof(double);
    
    //Allocate the memory for the train and test datasets
    double *currTrain = (double*)malloc(maxTrainBytes);
    double *currTest  = (double*)malloc(maxTestBytes);
    
    //Declare memcpy variables 
    size_t currTestSize, currTestSize_bytes, currTrainPoints, secondTrainOffset, secondTrainLen; 
    
    //Test offset is used to track where we are in the originalData
    size_t testOffset = 0;
    
    //Allocate the memory for store accuracy
    double *accuracy = (double*)calloc((numFolds+1), sizeof(double));

    for(int fold = 0; fold < numFolds; fold++){

        //Skip folds that do not belong to the current process
        if (fold % size != rank) {
     
            continue;
        }

        testOffset = foldStartIndex[fold]; 
        currTestSize        = pointsInFold[fold] * numFeatures;
        currTestSize_bytes  = currTestSize * sizeof(double);
        
        //Copy the current fold into the currTest
        memcpy(currTest, originalData + testOffset * numFeatures, currTestSize_bytes);

        currTrainPoints = totalNumPoints - pointsInFold[fold];
        
        //Copy the remaining folds into currTrain
        //To avoid the test set being included, the data is divided into two parts
        memcpy(currTrain, originalData, testOffset * numFeatures * sizeof(double));
        secondTrainOffset = (testOffset + pointsInFold[fold]) * numFeatures;
        secondTrainLen = (totalNumPoints - testOffset - pointsInFold[fold]) * numFeatures;
        memcpy(currTrain + testOffset * numFeatures, originalData + secondTrainOffset, secondTrainLen * sizeof(double));


        //For debugging. (Don't print asteroids)
        printArray(currTest, "test", pointsInFold[fold], numFeatures, 'd');
        printArray(currTrain, "train", currTrainPoints, numFeatures, 'd');
        
         /*
         * FROM HERE THE ARRAYS ARE ALLOCATED
         * IMPLEMENT YOUR CODE FOR KNN 
         * FOR EACH FOLD HERE
        */
        int *predictedlabels = (int*)malloc(pointsInFold[fold] * sizeof(int));
        knnomp(currTrain, currTrainPoints, numFeatures - 1, currTest, pointsInFold[fold], numFeatures - 1, k, numClasses, predictedlabels );
        
        //Call the function that calculates the accuracy and assigns the value to the array that records the accuracy
        double acy = calacy(predictedlabels, currTest, pointsInFold[fold], numFeatures);
        accuracy[fold] = acy;
        free(predictedlabels);
    }

    //Initialize and create an array to receive the results
    double *Accuracy = NULL;

     if (rank == 0){
        Accuracy = (double*)malloc((numFolds+1)*sizeof(double));
    }

    //Aggregate the results of each process to the main process
    MPI_Reduce(accuracy, Accuracy, numFolds+1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    //Add up all the accuracy rates and divide by the fold to get the average accuracy rate
    if (rank == 0){
        double totalaccuracy = 0.0;
        for(int i = 0; i < numFolds; i++){
            totalaccuracy += Accuracy[i];
        }
        Accuracy[numFolds] = totalaccuracy / numFolds;
        //Write the array containing the accuracy to the file
        writeResultsToFile(Accuracy, numFolds + 1, outFile);
        free(Accuracy);
    }

    free(accuracy);
    free(currTest);
    free(currTrain);
    free(pointsInFold);
    free(foldStartIndex);
    free(originalData);
    MPI_Finalize();
    return 0;
}


/**
 * Prints a given array
 * @param array: the array being given
 * @param array_name: the name of the array
 * @param m: the number of elements for the m dimension
 * @param n: the number of elements for the n dimension
 * @param type: the type of the array. e.g. d=double i=integer.
 **/
void printArray(void *array, char array_name[500], int m, int n, char type){
    printf("\n\n====Printing %s=====\n\n", array_name);
    int i, j;
    for(i = 0; i < m; i++){
        for(j = 0; j < n; j++){
            if(type == 'd') printf("%f, ", ((double*)array)[i * n + j]);
            else if (type == 'i') printf("%d, ", ((int*)array)[i * n + j]);

        }
        printf("\n");
    }
}
//A function that calculates the accuracy by counting the correct number of predicted labels
double calacy(int *predictedlabels, double *test_data, int test_size, int feature_count){
	int correctnum = 0;
    for(int i = 0; i < test_size; i++){
        int label = (int)test_data[i * feature_count + (feature_count - 1)];
        if (predictedlabels[i] == label)
        {
            correctnum++;
        }
    } 
    double accuracynum = (double)correctnum / (double)test_size;
    return accuracynum;
}

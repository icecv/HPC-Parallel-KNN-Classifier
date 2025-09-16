#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include<math.h>
#include<omp.h>
//#include"file-reader.c" -- If you include file-reader.c like this, make sure to remove file-reader.c function definitions just below and do not 

//Remove these if you do #include"file-reader.c"
int readNumOfPoints(char*);
int readNumOfFeatures(char*);
int readNumOfClasses(char *filename);
double *readDataPoints(char*, int, int);
void *writeResultsToFile(double*, int, char*);
extern void knnomp(double *traindata, int trainpointnum, int trainfeatnum, double *testdata, int testpointnum, int testfeatnum, int k, int classnum, int *predictedlabels);
void printArray(void *array, char array_name[500], int m, int n, char type);

int main(int argc, char *argv[]){

    printf("\n\n===============STARTING KNN===============\n\n");   

    //File reading
    char *inFile = argv[1];
    char *outFile = argv[2];
    int k = atoi(argv[3]);
    int numFolds = atoi(argv[4]);

    int totalNumPoints, numFeatures, numClasses;
    double *originalData;
    
    totalNumPoints = readNumOfPoints(inFile);
    numFeatures = readNumOfFeatures(inFile);
    numClasses = readNumOfClasses(inFile);
    originalData = readDataPoints(inFile, totalNumPoints, numFeatures);

    double calacy(int *predictedlabels, double *test_data, int test_size, int feature_count);
    
    //Array contains the number of points in each fold
    int *pointsInFold = (int *)malloc(numFolds * sizeof(int));
    
    //Calculate the points in the fold and the remainder
    int pointsPerFold = totalNumPoints / numFolds;
    int pointsPerFoldRemainder = totalNumPoints % numFolds;

    //Allocates the number of points in the fold adding the remainder to the last fold. (Can handle unequally sized folds)
    int fold;
    for(fold = 0; fold < numFolds; fold++){
        pointsInFold[fold] = pointsPerFold + ((fold == numFolds - 1) ? pointsPerFoldRemainder : 0);
    }

    /**
     * Calculate the maximum size of the train and test folds
     * Since the last fold always has the amount, we use this to our advantage
    **/
    size_t maxTrain_bytes = (totalNumPoints - pointsInFold[0]) * numFeatures * sizeof(double);
    size_t maxTest_bytes = pointsInFold[numFolds - 1] * numFeatures * sizeof(double);

    //Allocate the memory for the train and test datasets
    double *currTrain = (double*)malloc(maxTrain_bytes);
    double *currTest = (double*)malloc(maxTest_bytes);

    //Declare memcpy variables 
    size_t currTestSize, currTestSize_bytes, currTrainPoints, currTrainSize, currTrainSize_bytes, testOffset_bytes;

    //Test offset is used to track where we are in the originalData
    size_t testOffset = 0;

    //Allocate the memory for store accuracy
    double *accuracy = (double *)malloc((numFolds + 1) * sizeof(double));

    int currFold;
    for(currFold = 0; currFold < numFolds; currFold++){
        
        currTestSize = pointsInFold[currFold] * numFeatures;
        currTestSize_bytes = currTestSize * sizeof(double);
        testOffset_bytes = testOffset * sizeof(double);

        //Copy the current fold into the currTest
        memcpy(currTest, originalData + testOffset, currTestSize_bytes);

        currTrainPoints = totalNumPoints - pointsInFold[currFold];
        currTrainSize = currTrainPoints * numFeatures;
        currTrainSize_bytes = currTrainSize * sizeof(double);

        //Copy the remaining folds into currTrain
        memcpy(currTrain, originalData, testOffset_bytes);
        memcpy(currTrain + testOffset, originalData + testOffset + currTestSize, currTrainSize_bytes - testOffset_bytes);

        //For debugging. (Don't print asteroids)
        printArray(currTest, "test", pointsInFold[currFold], numFeatures, 'd');
        printArray(currTrain, "train", currTrainPoints, numFeatures, 'd');
        
        /*
         * FROM HERE THE ARRAYS ARE ALLOCATED
         * IMPLEMENT YOUR CODE FOR KNN 
         * FOR EACH FOLD HERE
        */
        int *predictedlabels = malloc(pointsInFold[currFold] * sizeof(int));
        knnomp(currTrain, currTrainPoints, numFeatures - 1, currTest, pointsInFold[currFold], numFeatures - 1, k, numClasses, predictedlabels);
       
        //Call the function that calculates the accuracy and assigns the value to the array that records the accuracy
        double acy = calacy(predictedlabels, currTest, pointsInFold[currFold], numFeatures);
        accuracy[currFold] = acy;
        testOffset += pointsInFold[currFold] * numFeatures;
        free(predictedlabels);
    }
    //Add up all the accuracy rates and divide by the fold to get the average accuracy rate
    double totalaccuracy = 0;
    for (int i = 0; i < numFolds; i++) {
        totalaccuracy += accuracy[i];
    }
    accuracy[numFolds] = totalaccuracy / numFolds; 
    //Write the array containing the accuracy to the file
    writeResultsToFile(accuracy, numFolds + 1, outFile);
    free(accuracy);  
    free(currTrain);
    free(currTest);

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
    double correctnum = 0;
    double accuracynum = 0;
    for(int i = 0; i < test_size; i++){
        int label = (int)test_data[i * feature_count + (feature_count - 1)];
        if (predictedlabels[i] == label)
        {
            correctnum++;
        }
    } 
    accuracynum = correctnum / test_size;
    return accuracynum;
}
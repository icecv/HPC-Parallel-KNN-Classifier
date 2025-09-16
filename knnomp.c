#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h> 
#include <limits.h>
#include <math.h>
#include <omp.h> 
// Define variables and method functions after importing necessary extension packages
extern int readNumOfPoints(char *filename);
extern int readNumOfFeatures(char *filename);
extern int readNumOfClasses(char *filename);
extern double *readDataPoints(char *filename, int numOfPoints, int numOfFeatures);
extern void *writeResultsToFile(double *output, int numOfPoints, int numOfFeatures, char *filename);
double Edistance(double *point1, double *point2, int featnum);
void findneighbor(double *distance, int trainpointnum, int k, int *saveneighbor);
int cate(int k, double *traindata, int trainfeatnum, int *saveneighbor, int classnum, double *distance);
void knnomp(double *traindata, int trainpointnum, int trainfeatnum, double *testdata, int testpointnum, int testfeatnum, int k, int classnum, int *predictedlabels);

// The main program implements knnomp
int knnompmain(int argc, char *argv[]) {
    // Read the transferred data file
    char *trainfile = argv[1];    
    char *testfile =  argv[2];
    char *outfile = argv[3];
    int k = atoi(argv[4]);
    //  Define variables for calling
    int trainpointnum = readNumOfPoints(trainfile);
    int trainfeatnum = readNumOfFeatures(trainfile)-1;
    int testpointnum = readNumOfPoints(testfile);
    int testfeatnum = readNumOfFeatures(testfile)-1;
    int classnum = readNumOfClasses(trainfile);
    double *traindata = readDataPoints(trainfile, trainpointnum, trainfeatnum+1);
    double *testdata = readDataPoints(testfile, testpointnum, testfeatnum+1);
    double *predictedlabels = (double *)malloc(testpointnum * sizeof(double));
    // knnomp process
    #pragma omp parallel
    for (int i = 0; i < testpointnum; i++) {
        double *distance = (double *)malloc(trainpointnum * sizeof(double));
        // Find the nearest distance
        for (int j = 0; j < trainpointnum; j++) {
            distance[j] = Edistance(&traindata[j * (trainfeatnum+1)], &testdata[i * (testfeatnum+1)], trainfeatnum);
        }
        // Find the k nearest neighbor
        int *saveneighbor = (int *)malloc(k * sizeof(int));
        findneighbor(distance, trainpointnum, k, saveneighbor);
        // classify k nearest neighbors  and filter out labels
        int prelabel = cate(k, traindata, trainfeatnum, saveneighbor, classnum, distance);
        predictedlabels[i] = prelabel;
        testdata[i * (testfeatnum + 1) + testfeatnum] = predictedlabels[i];
        free(distance);
        free(saveneighbor);
    }
    // Output result
    writeResultsToFile(testdata, testpointnum, testfeatnum+1 ,outfile);
    free(traindata);
    free(testdata);
    free(predictedlabels);
    return 0;
}

// Implement the method function to calculate the distance between points
double Edistance(double *point1, double *point2, int featnum){
    double sum = 0.0;
    // Calculate Euclidean distance
    for(int i = 0; i < featnum; i++){
        sum += pow(point1[i] - point2[i],2);
    }
    return sqrt(sum);
}
// Implement the method function to find the k nearest neighbor
void findneighbor(double *distance, int trainpointnum, int k, int *saveneighbor){
    // Initialize and reset the storage array
    for (int i = 0; i < k ; i++) {
        saveneighbor[i] = -1;
    }
    // By looping through the distance of each point, find the k neighbor points and store them
    for (int i = 0; i < k ; i++){
        double lowdis = __DBL_MAX__;
        for (int j = 0; j < trainpointnum ; j++){
            int include = 0 ;
            // Ensure same neighbor point not be store again
            for(int m=0 ; m < k ; m++){
                if (saveneighbor[m] == j){
                    include = 1;
                    break;
                }
            }
            // Find nearest neighbor point and store it
            if (distance[j] < lowdis && !include ){
                lowdis = distance[j];
                saveneighbor[i] = j;
            }
        }
    }   
}
// Implement the method function to classify k nearest neighbors  and filter out labels
int cate(int k,double *traindata,int trainfeatnum,int *saveneighbor,int classnum, double *distance){
    int maxcount = 0;
    int maxcountid = -1;
    int prelabel = -1;
    int maxlabelequal = 0;
    int nearnum = saveneighbor[0] * (trainfeatnum + 1) + trainfeatnum;
    int nearlabel = (int)traindata[nearnum];
    double mindistance = __DBL_MAX__;
    int *classcount = (int *)calloc(classnum, sizeof(int));
    int *pointlabel = (int *)malloc(k * sizeof(int));
    // Record the frequency of each label and the maximum frequency
    for(int i = 0; i < k; i++){
        int num = saveneighbor[i] * (trainfeatnum + 1) + trainfeatnum;
        pointlabel[i] = (int)traindata[num];
        classcount[pointlabel[i]]++;
        if(classcount[pointlabel[i]] > maxcount){
            maxcount = classcount[pointlabel[i]];
            maxcountid = i;
        }
    }
    // Checking for a tie
    for(int i = 0; i < classnum; i++){
        if (classcount[i] == maxcount){
            maxlabelequal++;
        }
    }
    // If tie, uses the label of the nearest point, otherwise use the label of the point with the highest frequency 
    if ( maxlabelequal > 1 || maxcount == 1 ) {
            prelabel = nearlabel;
    } 
    else {
            prelabel = pointlabel[maxcountid];
    }   
    free(classcount);
    free(pointlabel);
    return prelabel;
} 
// Bulid a new fuction for k-folds calling  
void knnomp(double *traindata, int trainpointnum, int trainfeatnum, double *testdata, int testpointnum, int testfeatnum, int k, int classnum, int *predictedlabels) {
    #pragma omp parallel
    for (int i = 0; i < testpointnum; i++) {
        double *distance = (double *)malloc(trainpointnum * sizeof(double));
        // Find the nearest distance
        for (int j = 0; j < trainpointnum; j++) {
            distance[j] = Edistance(&traindata[j * (trainfeatnum+1) + 1], &testdata[i * (testfeatnum+1) + 1], trainfeatnum - 1);
        }
        // Find the k nearest neighbor
        int *saveneighbor = (int *)malloc(k * sizeof(int));
        findneighbor(distance, trainpointnum, k, saveneighbor);
        // classify k nearest neighbors  and filter out labels
        int prelabel = cate(k, traindata, trainfeatnum, saveneighbor, classnum, distance);
        predictedlabels[i] = prelabel;

        free(distance);
        free(saveneighbor);
    }
}
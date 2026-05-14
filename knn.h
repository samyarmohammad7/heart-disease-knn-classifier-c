/*
 * Heart Disease Prediction using KNN Algorithm
 */

#ifndef KNN_H     
#define KNN_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>

/* 
 * Configuration Constants
 */

#define MAX_SAMPLES 400          // Maximum number of data samples
#define NUM_FEATURES 13          // Number of features 
#define MAX_LINE_LENGTH 1024     // Max length of CSV line
#define DEFAULT_K_VALUE 5        // Default K for KNN
#define TRAIN_RATIO 0.8          // 80% training, 20% testing

/* 
 * Structures to hold a single patient record
 */

typedef struct 
{
    double features[NUM_FEATURES];    // Array to store all 13 features
    int label;                        // 0 = No heart disease, 1 = Heart disease
} 
DataPoint;    

/*
 * Structure to hold distance and index for KNN sorting
 */

typedef struct
{
    double distance;    // Euclidean distance to test point
    int index;          // Index of the training sample
    int label;          // Label of the training sample
}
DistanceLabel;    


/*
 * Structure to hold the entire dataset
 */

typedef struct
{
    DataPoint* samples;         // Array of all data points
    int total_count;            // Total number of samples loaded
    int train_count;            // Number of training samples
    int test_count;             // Number of testing samples
    double* feature_min;        // Minimum values for normalization
    double* feature_max;        // Maximum values for normalization
} 
Dataset;    

/*
 * Structure to hold evaluation metrics
 */

typedef struct 
{
    int true_positives;     // Correctly predicted heart disease
    int true_negatives;     // Correctly predicted no heart disease
    int false_positives;    // Incorrectly predicted heart disease
    int false_negatives;    // Incorrectly predicted no heart disease
    double accuracy;        // Overall accuracy percentage
    double precision;       // Precision metric
    double recall;          // Recall (sensitivity) metric
    double f1_score;        // F1 Score metric
}
EvaluationMetrics;   

/* 
 * Functions Declarations
 */

// Data Loading Functions

Dataset* create_dataset(void);                          
void liberated_dataset(Dataset* dataset);                    
int load_csv_data(const char* filename, Dataset* dataset);  

// Data Preprocessing Functions 

void normalized_fea(Dataset* dataset);              
void split_bt_training_testing(Dataset* dataset);                
void shuffling_data(Dataset* dataset);                    

// KNN Core Algorithm Functions 

double euclidean_distance_calculated(double* point1, double* point2, int num_features); 
int comparing_distances(const void* a, const void* b);    
int knn_predicter(Dataset* dataset, double* test_point, int k);  
int knn_classifyer(DistanceLabel* neighbors, int k);      

//Evaluation Functions 

EvaluationMetrics evaluate_model(Dataset* dataset, int k);  
void printing_metrics(EvaluationMetrics metrics);         
void printing_ConMtrx(EvaluationMetrics metrics); 

// Utility Functions 

void printing_dataset_info(Dataset* dataset);              
void sample_print(DataPoint* sample, int index);        
int Optimal_K(Dataset* dataset, int max_k);        

#endif /* KNN_H */

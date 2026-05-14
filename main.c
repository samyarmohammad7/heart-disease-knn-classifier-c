/*
 * MAIN.C - Heart Disease Prediction using K-Nearest Neighbour Algorithm
 * The program implements a KNN-base machine learning system to predict heart disease, its based on patient health metrics, thanks to the dataset carried out by UCI Cleveland
 * Dataset: UCI Cleveland Heart Disease Dataset
 * Author: Sam Yarmohammad Touski Donadelli                    
 * Student ID: 22038018                                      
 * Course: Machine Learning                  
 * University: University of Hertfordshire            
 */

#include "knn.h"

/* 
 * PROGRAM CONFIGURATION
 * YOU CAN CHANGE THESE VALUES
 */
#define CSV_FILENAME "data_set.csv"    
#define K_VALUE 5                    
#define FIND_BEST_K 1                // Set to 1 to search for optimal K
#define MAX_K_SEARCH 15              // Maximum K to test when finding optimal

/* 
 * PRINT PROGRAM HEADER
 * Displays program information at startup
 */

void print_header_files(void) 
{    
    printf("\n");
    printf("Heart Disease Prediction System \n");   
    printf("Using K-Nearest Neighbour Algorithm \n");   
    printf("\n");
}

/* 
 * PRINT ALGORITHM EXPLANATION
 * Explains how KNN works - good for demonstrating understanding
 */
void algorithm_information(void)
{    
    printf("What is a KNN Algorithm? \n");
    printf("\n");
    printf("K-Nearest Neighbour (KNN) is a supervised machine learning \n");
    printf("algorithm used for classification tasks.\n");
    printf("\n");
    printf("The way the algorithm works is: \n");
    printf(" 1. Calculate distance from test point to all training points \n");
    printf(" 2. Sort distances and select K nearest neighbors \n");
    printf(" 3. Use majority voting to determine the class \n");
    printf("\n");
    printf("Distance Metric: Euclidean Distance \n");
    printf(" d = sqrt( sum( (x1[i] - x2[i])^2 ) ) \n");
    printf("\n");
    printf("For the algorithm to work the following implementation was used: \n");
    printf(" - Min and Max used for the  normalization of feature scaling \n");
    printf(" - %d features from patient health data gathered from the data set \n", NUM_FEATURES);
    printf(" - Binary classification (Disease / No Disease) \n");
}

/* 
 * DEMONSTRATE SINGLE PREDICTION
 * Shows how the model predicts for a single sample
 */

void prediction_demonstration(Dataset* dataset, int k)
{    
    printf("\n");
    printf("   Single Prediction Demonstration\n");
    printf("\n");
    
    // Use first test sample for demonstration
    int index_test = dataset->train_count;    
    DataPoint* sample = &dataset->samples[index_test];    
    
    printf("Testing sample at index %d: \n", index_test);
    
    // Displays feature values

    printf("Feature Values (normalized): \n");
    
    // Words used as names for display

    const char* feature_names[] = {    // YOU CAN CHANGE THESE NAMES
        "Age", "Sex", "Chest Pain", "Blood Pressure", "Cholesterol",
        "Blood Sugar", "ECG (Electrocardiogram)", "Max Heart Rate", "Exercise Angina",
        "ST Depression", "Slope", "Major Vessels", "Thalassemia"
    };
    
    for (int i = 0; i < NUM_FEATURES; i++) {
        printf(" %-20s: %.4f \n", feature_names[i], sample->features[i]);
    }
    
    printf("\n");
    printf("Actual Label: %d (%s) \n", 
           sample->label,
           sample->label == 1 ? "Heart Disease" : "No Heart Disease");
    
    // Making a  prediction

    int knn_prediction = knn_predicter(dataset, sample->features, k);    
    
    printf("Predicted:    %d (%s) \n", 
           knn_prediction,
           knn_prediction == 1 ? "Heart Disease" : "No Heart Disease");
    
    // Checks the prediction was done correctly

    if (knn_prediction == sample->label) {
        printf("\n Result: Correct! \n");    
    } else {
        printf("\n Result: Incorrect \n");   
    }
}

/* 
 * INTERACTIVE PREDICTION MODE
 * Allows user to input custom values for prediction
 */

void user_mode(Dataset* dataset, int k) 
{    
    char choice;    
    
    while (1) {  // Loop to allow multiple predictions
        printf("\n");
        printf(" Interactive Prediction Mode \n");
        printf("\n");
        
        printf("Would you like to enter custom patient data? (y/n): ");
        scanf_s(" %c", &choice);
        
        if (choice != 'y' && choice != 'Y')
        {
            printf(" Exiting interactive mode. \n");
            return;
        }
        
        double custom_features[NUM_FEATURES];    
        
        printf("\n");
        printf(" Enter Patient Health Data \n");
        printf("\n");
        printf("(Enter values within the suggested ranges) \n");
        
        // Prompt for each feature with clear instructions
        
        printf("[1/13] Age (years, e.g., 29-77): ");
        scanf_s("%lf", &custom_features[0]);
        
        printf("[2/13] Sex (0 = Female, 1 = Male): ");
        scanf_s("%lf", &custom_features[1]);
        
        printf("[3/13] Chest Pain Type:\n");
        printf("  0 = Typical Angina \n");
        printf("  1 = Atypical Angina \n");
        printf("  2 = Non-anginal Pain \n");
        printf("  3 = Asymptomatic \n");
        printf("  Enter a choice (0-3): ");
        scanf_s("%lf", &custom_features[2]);
        
        printf("[4/13] Resting Blood Pressure (e.g: 94-200): ");
        scanf_s("%lf", &custom_features[3]);
        
        printf("[5/13] Serum Cholesterol (e.g: 126-564): ");
        scanf_s("%lf", &custom_features[4]);
        
        printf("[6/13] Fasting Blood Sugar > 120 mg/dl? \n");
        printf(" (0 = No, 1 = Yes): ");
        scanf_s("%lf", &custom_features[5]);
        
        printf("[7/13] Resting ECG Results: \n");
        printf("  0 = Normal \n");
        printf("  1 = ST-T Wave Abnormality \n");
        printf("  2 = Left Ventricular Hypertrophy \n");
        printf("  Enter choice (0-2): ");
        scanf_s("%lf", &custom_features[6]);
        
        printf("[8/13] Maximum Heart Rate Achieved (e.g: 71-202): ");
        scanf_s("%lf", &custom_features[7]);
        
        printf("[9/13] Exercise Induced Angina? \n");
        printf("  (0 = No, 1 = Yes): ");
        scanf_s("%lf", &custom_features[8]);
        
        printf("[10/13] ST Depression (e.g: 0.0-6.2): ");
        scanf_s("%lf", &custom_features[9]);
        
        printf("[11/13] Slope of Peak Exercise ST Segment: \n");
        printf("  0 = Upsloping \n");
        printf("  1 = Flat \n");
        printf("  2 = Downsloping \n");
        printf("  Enter choice (0-2): ");
        scanf_s("%lf", &custom_features[10]);
        
        printf("[12/13] Number of Major Vessels (colored by fluoroscopy, 0-4): ");
        scanf_s("%lf", &custom_features[11]);
        
        printf("[13/13] Thalassemia: \n");
        printf("  0 = Normal \n");
        printf("  1 = Fixed Defect \n");
        printf("  2 = Reversible Defect \n");
        printf("  3 = Unknown \n");
        printf("  Enter choice (0-3): ");
        scanf_s("%lf", &custom_features[12]);
        
        // Display entered values for confirmation
        printf("\n");
        printf("   Data Entered - Summary \n");
        printf("\n");
        printf("  Age:                    %.0f years \n", custom_features[0]);
        printf("  Sex:                    %s \n", custom_features[1] == 1 ? "Male" : "Female");
        printf("  Chest Pain Type:        %.0f \n", custom_features[2]);
        printf("  Resting Blood Pressure: %.0f mm Hg \n", custom_features[3]);
        printf("  Cholesterol:            %.0f mg/dl \n", custom_features[4]);
        printf("  Fasting Blood Sugar:    %s \n", custom_features[5] == 1 ? "> 120 mg/dl" : "<= 120 mg/dl");
        printf("  Resting ECG:            %.0f \n", custom_features[6]);
        printf("  Max Heart Rate:         %.0f bpm \n", custom_features[7]);
        printf("  Exercise Angina:        %s \n", custom_features[8] == 1 ? "Yes" : "No");
        printf("  ST Depression:          %.1f \n", custom_features[9]);
        printf("  Slope:                  %.0f \n", custom_features[10]);
        printf("  Major Vessels:          %.0f \n", custom_features[11]);
        printf("  Thalassemia:            %.0f \n", custom_features[12]);
        
        // Normalizing the input features using dataset 

        printf("\n");
        printf("Processing data... \n");
        
        // Creating a copy for normalization

        double features_normalized[NUM_FEATURES];    
        
        for (int i = 0; i < NUM_FEATURES; i++) 
        {
            double range = dataset->feature_max[i] - dataset->feature_min[i];
            if (range > 0) {
                features_normalized[i] = (custom_features[i] - dataset->feature_min[i]) / range;
                if (features_normalized[i] < 0) features_normalized[i] = 0;
                if (features_normalized[i] > 1) features_normalized[i] = 1;
            } 
            else
            {
                features_normalized[i] = 0;
            }
        }
        
        // Making a prediction

        int prediction = knn_predicter(dataset, features_normalized, k);
        
        // Results with a visual aid

        printf("\n");
        printf("Results Predictions \n");
        printf("\n");
        
        if (prediction == 1) {
            printf("  RISk IS HIGH: Heart Disease Likely \n");    
            printf("  Algorithm predicts this patient has \n");
            printf("  indicators consistent with heart diseases. \n");
            printf("  Recommendation: \n");
            printf("  - Consult a cardiologist immediately \n");
        } 
        else 
        {
            printf("  RISK IS LOW: No Heart Disease Indicated \n");    
            printf("  Algorithm predicts this patient does \n");
            printf("  not show indicators of heart disease.\n");
            printf("  Recommendation: \n");
            printf("  - Continue regular health check-ups\n");
        }
        
        
        // Asks if user wants to try another prediction
        printf("\n");
    }
}

/* 
 * MAIN FUNCTION
 * Program entry point - orchestrates the entire workflow
 */

int main(void) 
{
    // Prints program header

    print_header_files();
    
    // Print algorithm information

    algorithm_information();
    
    /* 
     * STEP 1: CREATE AND LOAD DATASET
     */

    printf("\n");
    printf("  Step 1: Loading Dataset \n");
    printf("\n");
    
    // Creates dataset structure

    Dataset* dataset = create_dataset();    
    
    if (dataset == NULL)
    {
        printf("ERROR-ERROR: DATASET NOT CREATED! \n");    
        return 1;
    }
    
    // Loads data from CSV (excel file)

    int count_loaded = load_csv_data(CSV_FILENAME, dataset);    
    
    if (count_loaded <= 0) 
    {
        printf("ERROR-ERROR: DATASET NOT CREATED FROM %s \n", CSV_FILENAME);
        printf("\n");
        printf("Please ensure the file exists in the same directory and named correctly. \n");
        printf("Download from: https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci \n");
        liberated_dataset(dataset);
        return 1;
    }
    
    // Displays dataset information

    printing_dataset_info(dataset);
    
    /* 
     * STEP 2: PREPROCESS DATA
     */

    printf("\n");
    printf("  Step 2: Preprocessing Data \n");
    printf("\n");
    
    // Shuffle the data for random distribution

    shuffling_data(dataset);
    
    // Normalize features to [0, 1] range

    normalized_fea(dataset);
    
    // Split into training and testing sets

    split_bt_training_testing(dataset);
    
    /* 
     * STEP 3: Evaluation and Training Model
     */
    printf("\n");
    printf("  Step 3: Evaluation and Training Model \n");
    printf("\n");
    
    int k_value_to_use = K_VALUE;    
    
    // Optionally find the optimal K value

    #if FIND_BEST_K
        printf("Searching for optimal K value... \n");
        k_value_to_use = Optimal_K(dataset, MAX_K_SEARCH);
    #endif
    
    // Evaluate model with selected K

    printf(" Final Evaluation with K = %d: \n", k_value_to_use);
    EvaluationMetrics final_metrics = evaluate_model(dataset, k_value_to_use);
    
    // Display results

    printing_ConMtrx(final_metrics);
    printing_metrics(final_metrics);
    
    /* 
     * STEP 4: DEMONSTRATION
     */
    prediction_demonstration(dataset, k_value_to_use);
    
    /* 
     * STEP 5: INTERACTIVE MODE
     * Allows user to input custom patient data for prediction
     */

    user_mode(dataset, k_value_to_use);    // NOW ENABLED
    
    /* 
     * CLEANUP
     */

    printf("\n");
    printf("   Program Complete \n");
    printf("\n");
    
    printf("Summary: \n");
    printf("  - Algorithm: K-Nearest Neighbour (K = %d) \n", k_value_to_use);
    printf("  - Training samples: %d \n", dataset->train_count);
    printf("  - Testing samples: %d \n", dataset->test_count);
    printf("  - Final Accuracy: %.2f%% \n", final_metrics.accuracy);
    
    // Free allocated memory

    liberated_dataset(dataset);
    
    printf("Predictions completed, thanks for using the model. \n");    
    
    return 0;
}

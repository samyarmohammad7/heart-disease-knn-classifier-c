/*
 * Core KNN algorithm for heart disease prediction
 * 
 * Author: Sam Yarmohammad Touski Donadelli                                      
 */

#include "knn.h"

/*
 * Computing the Euclidean distance between two points in n-dimensional space
 */

double euclidean_distance_calculated(double* point1, double* point2, int num_features) {
    double sum_diff = 0.0;    
    
    // Calculating the sum of squared differences for each variable

    for (int i = 0; i < num_features; i++) 
    {
        double difference = point1[i] - point2[i];    
        sum_diff += difference * difference;   // Square the difference
    }
    
    // Return the square root of the sum

    return sqrt(sum_diff);
}

/* 
 * Comparison function for qsort() to sort distances in ascending order
 * Used to find the K nearest neighbors
 */

int comparing_distances(const void* a, const void* b)
{
    DistanceLabel* distance_A = (DistanceLabel*)a;    
    DistanceLabel* distance_B = (DistanceLabel*)b;    
    
    // Compare distances for sorting

    if (distance_A->distance < distance_B->distance) return -1;
    if (distance_A->distance > distance_B->distance) return 1;
    return 0;
}

/* 
 * Determines the class label by majority voting among K nearest neighbors
 */

int knn_classifyer(DistanceLabel* neighbors, int k) {
    int positive_vote = 0;    // Voting for heart disease
    int negative_vote = 0;    // Voting for no heart disease
    
    // Count votes from each of the K nearest neighbors

    for (int i = 0; i < k; i++) 
    {
        if (neighbors[i].label == 1) 
        {
            positive_vote++;
        } 
        else 
        {
            negative_vote++;
        }
    }
    
    // Return the majority class
    // If votes come up to a tie, the model predicts positive as is safer for medical applications
  
    if (positive_vote >= negative_vote)
    {
        return 1;  // Heart disease
    } 
    else 
    {
        return 0;  // No heart disease
    }
}

/* 
 * Main prediction function, finds K nearest neighbors and returns prediction
 */

int knn_predicter(Dataset* dataset, double* test_point, int k)
{
    // Allocate array to store distances to all training samples

    DistanceLabel* distances = (DistanceLabel*)malloc(dataset->train_count * sizeof(DistanceLabel));
    
    if (distances == NULL)
    {
        printf("ERROR!!!: Allocation to memory failed \n");  // YOU CAN CHANGE THIS MESSAGE
        return -1;
    }
    
    // Calculate distance from test point to each training sample

    for (int i = 0; i < dataset->train_count; i++) 
    {
        distances[i].distance = euclidean_distance_calculated(
            test_point, 
            dataset->samples[i].features, 
            NUM_FEATURES
        );
        distances[i].index = i;
        distances[i].label = dataset->samples[i].label;
    }
    
    // Sort distances in ascending order (closest first)

    qsort(distances, dataset->train_count, sizeof(DistanceLabel), comparing_distances);
    
    // Uses majority voting among K nearest neighbors to get a prediction

    int predictions = knn_classifyer(distances, k);    
    
    // Free allocated memory

    free(distances);
    
    return predictions;
}

/* 
 * Tests the KNN model on the test set and calculates performance metrics
 *  The metrics calculated are as followed:
 *  Accuracy: (TP + TN) / Total
 *  Precision: TP / (TP + FP)
 *  Recall: TP / (TP + FN)
 *  F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
 */

EvaluationMetrics evaluate_model(Dataset* dataset, int k)
{
    EvaluationMetrics metrics;    
    
    // Initializes all counters to zero

    metrics.true_positives = 0;
    metrics.true_negatives = 0;
    metrics.false_positives = 0;
    metrics.false_negatives = 0;
    
    printf("  Evaluating the KNN Model by the use of: (K = %d) \n", k);            
    printf("\n");     
    
    printf("Testing the following number %d of samples...\n", dataset->test_count);  
    
    // Test samples are located after training samples in the array

    int index_test_start = dataset->train_count;    
    
    // Iterate through all test samples

    for (int i = 0; i < dataset->test_count; i++)
    {
        int index_sam = index_test_start + i;    
        
        // Get the actual label

        int actual_label = dataset->samples[index_sam].label;    
        
        // Get prediction from KNN

        int prediction_label = knn_predicter
        (                          
            dataset, 
            dataset->samples[index_sam].features, 
            k
        );
        
        // Update confusion matrix counts

        if (actual_label == 1 && prediction_label == 1)
        {
            metrics.true_positives++;    // Correctly predicted heart disease
        } 
        else if 
            (actual_label == 0 && prediction_label == 0)
        {
            metrics.true_negatives++;    // Correctly predicted no heart disease
        }
        else if
            (actual_label == 0 && prediction_label == 1) 
        {
            metrics.false_positives++;   // Incorrectly predicted heart disease
        } 
        else if 
            (actual_label == 1 && prediction_label == 0)
        {
            metrics.false_negatives++;   // Missed heart disease case
        }
    }
    
    // Calculates a performance of the metrics given

    int total_num_samples = dataset->test_count;    
    int correct_num_predictions = metrics.true_positives + metrics.true_negatives;  
    
    // Accuracy: percentage of correct predictions

    metrics.accuracy = (double)correct_num_predictions / total_num_samples * 100.0;
    
    // Precision: of all positive predictions, how many were correct

    if (metrics.true_positives + metrics.false_positives > 0) 
    {
        metrics.precision = (double)metrics.true_positives / 
                           (metrics.true_positives + metrics.false_positives) * 100.0;
    } 
    else 
    {
        metrics.precision = 0.0;
    }
    
    // Recall (Sensitivity): of all actual positives, how many were detected

    if (metrics.true_positives + metrics.false_negatives > 0) 
    {
        metrics.recall = (double)metrics.true_positives / 
                        (metrics.true_positives + metrics.false_negatives) * 100.0;
    } 
    else
    {
        metrics.recall = 0.0;
    }
    
    // F1 Score: harmonic mean of precision and recall

    if (metrics.precision + metrics.recall > 0)
    {
        metrics.f1_score = 2.0 * (metrics.precision * metrics.recall) / 
                          (metrics.precision + metrics.recall);
    } 
    else
    {
        metrics.f1_score = 0.0;
    }
    
    return metrics;
}

/* 
 * Printing a Confusion Matrix
 * Confusion Matrix Layout:
 * Predicted
 *               Neg    Pos
 *   Actual Neg  [TN]   [FP]
 *   Actual Pos  [FN]   [TP]
 */

void printing_ConMtrx(EvaluationMetrics metrics) 
{
    printf("  Confusion Matrix \n");                               
    
    printf("Predicted \n");
    printf("               Neg      Pos \n");
    printf("Actual Neg  |  %4d  |  %4d  |  (No Disease) \n", metrics.true_negatives, metrics.false_positives);
    printf("\n");
    printf("Actual Pos  |  %4d  |  %4d  |  (Disease)\n", metrics.false_negatives, metrics.true_positives);
    printf("\n");
    
    // Summary

    printf("Summary: \n");                                           
    printf("  TN (True Negative): %d, the model correctly predicted no disease \n", metrics.true_negatives);
    printf("  FP (False Positive): %d, the model incorrectly predicted disease \n", metrics.false_positives);
    printf("  FN (False Negative): %d, the model missed disease case \n", metrics.false_negatives);
    printf("  TP (True Positive): %d, orrectly predicted disease\n", metrics.true_positives);
}

/* 
 * Displaying all evaluation metrics in a formatted output
 */

void printing_metrics(EvaluationMetrics metrics) 
{
    printf(" Evaluation Metrics \n");                            
    printf("\n");         
    printf("Accuracy: %.2f%% \n", metrics.accuracy);               
    printf("Precision: %.2f%% \n", metrics.precision);              
    printf("Recall: %.2f%% \n", metrics.recall);                 
    printf("F1 Score: %.2f%% \n", metrics.f1_score);             
    
    // Interpretation of results
   
    printf("Interpretation of results: \n");
    
    if (metrics.accuracy >= 80.0)
    {
        printf("* ACCURACY: Excellent, the  model has correctly identified %.1f%% of cases accuratly \n", metrics.accuracy);
    }
    else if (metrics.accuracy >= 70.0)
    {
        printf("* ACCURACY: Good, the model has correctly identified %.1f%% of cases accuratly \n", metrics.accuracy);
    }
    else
    {
        printf("* ACCURACY: Could be better, the model  has only identified %.1f%% correct \n", metrics.accuracy);
    }
    

    if (metrics.precision >= 80.0)
    {
        printf("* PRECISION: High, the model has only a few false alarms \n");
    } 
    else
    {
        printf("* PRECISION: Moderate, the model has only some false positives occurring \n");  
    }
    
    if (metrics.recall >= 80.0)
    {
        printf("* RECALL: High, the modelhas identified most disease cases are correctly \n");
    } 

    else 
    {
        printf("* RECALL: Moderatem the model has only a few disease cases being missed \n");
    }
    
    printf("\n");
}

/* 
 * Finding the best K value
 * Tests different values of K to find the one with best accuracy
 */

int Optimal_K(Dataset* dataset, int max_k) 
{
    printf("\n");         
    printf(" Finding the best optimal K Value \n");                        
    printf("\n");        
    
    int best_k = 1;                    
    double best_accuracy = 0.0;        
    
    // Test odd values of K from 1 to max_k (odd to avoid ties)
    for (int k = 1; k <= max_k; k += 2) 
    {    
        EvaluationMetrics temp_metrics = evaluate_model(dataset, k);
        
        printf("K = %2d: Accuracy = %.2f%% \n", k, temp_metrics.accuracy);
        
        if (temp_metrics.accuracy > best_accuracy) 
        {
            best_accuracy = temp_metrics.accuracy;
            best_k = k;
        }
    }
    
    printf("\n Optimal has benn K found: %d (Accuracy: %.2f%%)\n", best_k, best_accuracy);
    
    return best_k;
}

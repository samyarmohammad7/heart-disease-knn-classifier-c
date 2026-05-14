/*
 * DATA_LOADER.C: This file is in charge of data loading and preprocessing functions
 * Furthermore handles CSV parsing, data normalization, and train/test splitting
 */

#include "knn.h"
#include <time.h>

/* 
 * FEATURE NAMES
 */
const char* FEATURE_NAMES[NUM_FEATURES] = 
{   
    "Age",                      // Feature 0
    "Sex",                      // Feature 1: 1=male, 0=female
    "Chest Pain Type",          // Feature 2: 1-4
    "Resting Blood Pressure",   // Feature 3: mm Hg
    "Cholesterol",              // Feature 4: mg/dl
    "Fasting Blood Sugar",      // Feature 5: >120 mg/dl (1=true, 0=false)
    "Resting ECG",              // Feature 6: 0-2
    "Max Heart Rate",           // Feature 7
    "Exercise Angina",          // Feature 8: 1=yes, 0=no
    "ST Depression (Oldpeak)",  // Feature 9
    "Slope",                    // Feature 10: 1-3
    "Num Major Vessels",        // Feature 11: 0-3
    "Thalassemia"               // Feature 12: 3=normal, 6=fixed, 7=reversible
};

/*
 * Allocating part of memory for a new dataset structure
 */
Dataset* create_dataset(void)
{
    // Allocating memory for the dataset structure

    Dataset* dataset = (Dataset*)malloc(sizeof(Dataset));
    
    if (dataset == NULL) 

    {
        printf("ERROR: Failed to allocate memory for dataset! \n");  
        return NULL;
    }
    
    dataset->samples = (DataPoint*)malloc(MAX_SAMPLES * sizeof(DataPoint));
    
    if (dataset->samples == NULL)
    {
        printf("ERROR: Failed to allocate part of memory for the samples! \n");  
        free(dataset);
        return NULL;
    }
    
    // Allocating memory for the normalization of arrays 

    dataset->feature_min = (double*)malloc(NUM_FEATURES * sizeof(double));
    dataset->feature_max = (double*)malloc(NUM_FEATURES * sizeof(double));
    
    if (dataset->feature_min == NULL || dataset->feature_max == NULL)
    {
        printf("ERROR: Failed to allocate part of memory for normalization of arrays! \n");  
        free(dataset->samples);
        free(dataset);
        return NULL;
    }
    
    // Initialize counters

    dataset->total_count = 0;
    dataset->train_count = 0;
    dataset->test_count = 0;
    
    // Initialize min/max arrays

    for (int i = 0; i < NUM_FEATURES; i++) 
    {
        dataset->feature_min[i] = DBL_MAX;   // Starting with largest possible value
        dataset->feature_max[i] = -DBL_MAX;  // Starting with smallest possible value
    }
    
    return dataset;
}

/* 
 * Releases all memory used by the dataset
 */
void liberated_dataset(Dataset* dataset) 
{
    if (dataset != NULL)
    {
        if (dataset->samples != NULL)
        {
            free(dataset->samples);
        }
        if (dataset->feature_min != NULL) 
        {
            free(dataset->feature_min);
        }
        if (dataset->feature_max != NULL) 
        {
            free(dataset->feature_max);
        }
        free(dataset);
    }

    printf("Dataset memory liberated successfully \n");  
}

/* 
 * Reads the heart disease CSV file and fills the dataset
 */
int load_csv_data(const char* filename, Dataset* dataset) 
{
    FILE* file = fopen(filename, "r");
    
    if (file == NULL) 
    {
        printf("ERROR: Cannot open file '%s', please try again \n", filename);  
        printf("Make sure the CSV file is in the same directory \n");  
        return -1;
    }
    
    char line[MAX_LINE_LENGTH];    
    int line_count = 0;            
    int sample_index = 0;          
    
    printf("   Loading Heart Disease Dataset \n");                
    printf("\n");      
    
    // Reads the file line by line

    while (fgets(line, MAX_LINE_LENGTH, file) != NULL && sample_index < MAX_SAMPLES)
    {
        line_count++;
        
        // Skips header line as first line contains column names

        if (line_count == 1)
        {
            printf("Skipping header line \n"); 
            continue;
        }
        
        // Parsing the CSV line

        DataPoint* current_sample = &dataset->samples[sample_index];    
        
        // Temporary variables for parsing

        double age, sex, cp, trestbps, chol, fbs, restecg;
        double thalach, exang, oldpeak, slope, ca, thal, target;
        
        // Parse all 14 values from the CSV line (13 features + 1 label)

        int fields_read = sscanf(line, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf",
                                 &age, &sex, &cp, &trestbps, &chol, &fbs, &restecg,
                                 &thalach, &exang, &oldpeak, &slope, &ca, &thal, &target);
                                 
        // Check if all fields were read correctly

        if (fields_read != 14)
        {
            printf("WARNING: Line %d has invalid format (read %d fields). Skipping \n", line_count, fields_read);  
            continue;
        }
        
        // Store features in the data point

        current_sample->features[0] = age;
        current_sample->features[1] = sex;
        current_sample->features[2] = cp;
        current_sample->features[3] = trestbps;
        current_sample->features[4] = chol;
        current_sample->features[5] = fbs;
        current_sample->features[6] = restecg;
        current_sample->features[7] = thalach;
        current_sample->features[8] = exang;
        current_sample->features[9] = oldpeak;
        current_sample->features[10] = slope;
        current_sample->features[11] = ca;
        current_sample->features[12] = thal;
        
        // Store the label (0 = no disease, 1 = disease)

        current_sample->label = (int)target;
        
        // Update min/max values for each feature (needed for normalization)

        for (int i = 0; i < NUM_FEATURES; i++)
        {
            if (current_sample->features[i] < dataset->feature_min[i])
            {
                dataset->feature_min[i] = current_sample->features[i];
            }
            if (current_sample->features[i] > dataset->feature_max[i]) 
            {
                dataset->feature_max[i] = current_sample->features[i];
            }
        }
        
        sample_index++;
    }
    
    fclose(file);
    
    dataset->total_count = sample_index;
    
    printf("Successfully loaded %d samples from %d lines. \n", dataset->total_count, line_count);
    
    return dataset->total_count;
}

/* 
 * Scaling all features to range from  [0, 1] using Min-Max for normalization
 */

void normalized_fea(Dataset* dataset) 
{
    printf("Normalizing all the features using Min-Max scaling \n");  
    
    for (int i = 0; i < dataset->total_count; i++) 
    {
        for (int j = 0; j < NUM_FEATURES; j++)
        {
            double minimum_value = dataset->feature_min[j];    
            double maximum_val = dataset->feature_max[j];    
            double range = maximum_val - minimum_value;            
            
            // Avoid division by zero

            if (range > 0) 
            {
                dataset->samples[i].features[j] = 
                    (dataset->samples[i].features[j] - minimum_value) / range;
            } 
            else 
            {
                dataset->samples[i].features[j] = 0.0;
            }
        }
    }
    
    printf("Normalization done, all features have been scaled to [0, 1] \n");  
}

/* 
 * Randomly shuffles the dataset using Fisher-Yates algorithm, ensuring random distribution of between samples in train/test splits
 */

void shuffling_data(Dataset* dataset) 
{
    printf("Shuffling dataset \n");  
    
    // Seed the random number generator

    srand((unsigned int)time(NULL));    
    
    // Fisher-Yates shuffle algorithm

    for (int i = dataset->total_count - 1; i > 0; i--) 
    {
        // Generate random index between 0 and i

        int j = rand() % (i + 1);    
        
        // Swap samples[i] and samples[j]

        DataPoint temp = dataset->samples[i];    
        dataset->samples[i] = dataset->samples[j];
        dataset->samples[j] = temp;
    }
    
    printf("Shuffle done \n");  
}

/* 
 * Dividing the dataset into training and testing sets
 */

void split_bt_training_testing(Dataset* dataset) 
{
    // Calculating a split point

    dataset->train_count = (int)(dataset->total_count * TRAIN_RATIO);
    dataset->test_count = dataset->total_count - dataset->train_count;
    
    printf(" Train/Test Split Completed \n");
    printf("Total samples: %d \n", dataset->total_count);        
    printf("Training samples: %d (%.0f%%) \n",  dataset->train_count, TRAIN_RATIO * 100);
    printf("Testing samples:  %d (%.0f%%) \n", dataset->test_count, (1.0 - TRAIN_RATIO) * 100);
}

/* 
 * Displaying a summary information about the loaded dataset
 */

void printing_dataset_info(Dataset* dataset)
{
    printf(" Dataset Information \n");
    
    // Count positive and negative cases

    int positive_count = 0;    
    int negative_count = 0;    
    
    for (int i = 0; i < dataset->total_count; i++) 
    {
        if (dataset->samples[i].label == 1) 
        {
            positive_count++;
        } 
        else 
        {
            negative_count++;
        }
    }
    
    printf("Total Samples Found: %d \n", dataset->total_count);
    printf("Heart Disease Cases Found as Positive: %d (%.1f%%) \n", positive_count, (double)positive_count / dataset->total_count * 100);
    printf("No Heart Disease Found Also Seen As Negative: %d (%.1f%%) \n", negative_count, (double)negative_count / dataset->total_count * 100);
    
    printf("Features (%d total):\n", NUM_FEATURES);
    printf("\n"); 

    for (int i = 0; i < NUM_FEATURES; i++) 
    {
        printf(" [%2d] %s \n", i, FEATURE_NAMES[i]);
    }
    printf("\n");
}

/* 
 * Displays details of a single data point
 */

void sample_print(DataPoint* sample, int index)
{
    printf(" Sample #%d: \n", index);
    printf(" Features: ");

    for (int i = 0; i < NUM_FEATURES; i++)
    {
        printf(" %.2f ", sample->features[i]);
    }
    printf("\n");
    printf(" Label: %d (%s) \n",sample->label, sample->label == 1 ? "Heart Disease" : "No Heart Disease");
}

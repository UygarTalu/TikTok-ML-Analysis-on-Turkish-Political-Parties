########################################################################################################################
########################################################################################################################
########################################################################################################################

############################################ CORRELATION ANALYSIS - KMEANS ANALYSIS  ###################################


###INFORMATION ABOUT THE SCRIPT(ANALYSIS LAYER 7)
"""
IN THE ANALYSIS LAYER ONE WE HAVE 4 DIFFERENT COMPLEX FUNCTIONS.

1- "find_highly_correlated_features"
2- "preprocess_data_correlation_analysis"
3- "run_kmeans"
4- "visualize_clusters"

Results for Layer 7- At the end of execution of layer 7 We preprocess the data file for both correlation analysis and
clustering analysis. So at the end of the execution we have the correlation values in bwteen the features and the cluster
numbers for each video
"""




########### CORRELATION ANALYSIS #######################################################################################






#PREPROCESSING THE DATA BEFORE CORRELATION ANALYSIS - LABEL ENCODING, STANDART SCALING - IDENTIFYING REDUNDANT FEATURES
########################################################################################################################

#The function finds_highly_correlated_features takes a dataframe (data) and a correlation threshold
#(correlation_threshold) as input. It calculates the correlation matrix of the dataframe using the corr() method.
#Then, it iterates over the upper triangular portion of the correlation matrix to find pairs of highly correlated
#features.
#For each iteration, it checks if the correlation value between two features (correlation_matrix.iloc[i, j]) is
#greater than the specified threshold (correlation_threshold). If the condition is met, it adds the pair of feature
#names (feature_i and feature_j) to the highly_correlated_pairs list as a tuple.
#Finally, the function returns the list of highly correlated feature pairs.

def find_highly_correlated_features(data, correlation_threshold):
    """
    Finds pairs of highly correlated features in a given dataframe based on a correlation threshold.

    Args:
        data (pandas.DataFrame): The input dataframe containing the features.
        correlation_threshold (float): The threshold above which two features are considered highly correlated.

    Returns:
        list: A list of tuples representing the highly correlated feature pairs.
    """
    correlation_matrix = data.corr().abs()
    highly_correlated_pairs = []

    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            if correlation_matrix.iloc[i, j] > correlation_threshold:
                feature_i = correlation_matrix.columns[i]
                feature_j = correlation_matrix.columns[j]
                highly_correlated_pairs.append((feature_i, feature_j))

    return highly_correlated_pairs

# Example usage
correlation_threshold = 0.8
highly_correlated_pairs = find_highly_correlated_features(ready_final_data_correlation_analysis,
                                                          correlation_threshold)

# Print the highly correlated feature pairs
for pair in highly_correlated_pairs:
    print(pair)
########################################################################################################################



#CORRELATION COMPUTATION AND HEATMAP VISUALIZATION
########################################################################################################################

#The code computes the correlation matrix by calling the corr() method on the ready_final_data_correlation_analysis_copy
#dataframe.
#Also depending on the correlation values it creates the heatmap so that it can be more understandable to understand
#the relations and also capture the redundant features


correlation_matrix = ready_final_data_correlation_analysis_copy.corr()


plt.figure(figsize=(19, 19))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
########################################################################################################################




#EXECUTION FUNCTION FOR PREPROCESSING THE DATA FOR CORRELATION ANALYSIS AND CORRELATION CALCULATOR
########################################################################################################################
#The function preprocess_data_correlation_analysis takes an input dataframe (general_data_correlation_analysis) and
#performs the following preprocessing steps:
#It creates a copy of the dataframe to avoid modifying the original data.
#It removes specified columns using the drop() method.
#It encodes categorical columns ("Dominant Emotion", "Label", "Topic_Labels_Comments", "Topic_Labels_Recognized_Speeches")
#using LabelEncoder. The encoded labels are stored in the dataframe, and the original labels before encoding are
#stored in a dictionary (label_mappings) for each column.
#It saves the modified dataframe to a CSV file using the to_csv() method.
#It prints the saved file path and label mappings for each column.
#Finally, it returns the preprocessed dataframe and the dictionary of label mappings.


def preprocess_data_correlation_analysis(general_data_correlation_analysis):
    """
    Preprocesses the data for correlation analysis by removing columns, encoding categorical columns, and saving the modified dataframe.

    Args:
        general_data_correlation_analysis (pandas.DataFrame): The input dataframe to be preprocessed.

    Returns:
        tuple: A tuple containing the preprocessed dataframe and a dictionary of label mappings.
    """
    # Copy the dataframe
    general_data_correlation_analysis_preprocessed = general_data_correlation_analysis.copy()

    # Remove columns
    columns_to_remove = ["video_diggcount", "video_diggcount_STANDARDIZED",
                     "video_sharecount", "video_sharecount_STANDARDIZED",
                     "video_commentcount","video_commentcount_STANDARDIZED",
                     "video_playcount","video_playcount_STANDARDIZED"]

    general_data_correlation_analysis_preprocessed.drop(columns=columns_to_remove, inplace=True)

    # Encode columns
    columns_to_encode = ["Dominant Emotion", "Label",
                         "Topic_Labels_Comments", "Topic_Labels_Recognized_Speeches"]
    label_encoder = LabelEncoder()

    # Create dictionaries to store the label mappings
    label_mappings = {}

    for column in columns_to_encode:
        label_encoder.fit(general_data_correlation_analysis_preprocessed[column])
        encoded_labels = label_encoder.transform(general_data_correlation_analysis_preprocessed[column])
        general_data_correlation_analysis_preprocessed[column] = encoded_labels + 1

        # Store the original labels before encoding
        original_labels = label_encoder.inverse_transform(encoded_labels)
        unique_labels = list(set(original_labels))
        label_mappings[column] = dict(zip(range(1, len(unique_labels) + 1), unique_labels))

    # Save the modified dataframe to a CSV file
    output_path = "C:/Users/Uygar TALU/Desktop/"
    output_filename = "ready_final_data_correlation_analysis.csv"
    output_file = os.path.join(output_path, output_filename)
    general_data_correlation_analysis_preprocessed.to_csv(output_file, index=False)

    print("Modified dataframe saved successfully as:", output_file)

    # Print label mappings
    for column, mapping in label_mappings.items():
        print(f"Column: {column}")
        for encoded_label, label in mapping.items():
            print(f"{encoded_label}: {label}")

    return general_data_correlation_analysis_preprocessed, label_mappings

#EXECUTION
ready_final_data_correlation_analysis, label_mappings = preprocess_data_correlation_analysis()
########################################################################################################################




########### K-MEANS ANALYSIS ###########################################################################################





#KMEANS FUNCTION OPTIMIZES NUMBER OF CLUSTERS AND CREATES THE OPTIMAL MODEL
########################################################################################################################
#This function performs K-means clustering on the given dataset. The features used in the clustering process are all
#the columns in df, excluding those specified in label_encoded_features and the video_id column. Before running K-means,
#it scales the features using StandardScaler to ensure that the clustering algorithm does not get affected by the
#magnitude of the features.
#The function then performs K-means clustering for a range of cluster numbers specified by k_range, computes the
#distortion (within-cluster sum of squares) for each number of clusters, and plots these distortions. The elbow method
#is then used to determine the optimal number of clusters: the point at which adding another cluster
#doesn't significantly improve the total distortion.
#After determining the optimal number of clusters, the function fits the K-means algorithm with that
#number of clusters and appends the cluster labels to the original dataframe. The dataframe, now including
#a 'cluster' column indicating the cluster each row belongs to, is returned by the function.


def run_kmeans(df, k_range=(1, 10), label_encoded_features=[]):
    """
    This function performs K-means clustering on a given dataframe.
    It uses the elbow method to find the optimal number of clusters.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the data to be clustered
    k_range (tuple): A tuple of two integers indicating the range of cluster numbers to try (default is (1, 10))
    label_encoded_features (list): A list of column names that are label-encoded and should not be scaled (default is an empty list)

    Returns:
    df (pandas.DataFrame): DataFrame with an added 'cluster' column indicating the cluster each row belongs to
    """

    print("Starting K-means clustering...")

    # All other columns (excluding the categorical ones) will be used for K-means
    kmeans_features = [col for col in df.columns if col not in label_encoded_features + ['video_id']]
    print(f"Features used for clustering: {kmeans_features}")

    # The data to use in K-means
    X = df[kmeans_features]

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Elbow method to determine optimal number of clusters
    distortions = []
    K = range(*k_range)
    for k in K:
        print(f"Fitting K-means with {k} clusters...")
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X_scaled)
        distortions.append(kmeans.inertia_)
        print(f"For {k} clusters, the distortion is: {kmeans.inertia_}")

    # Determine the number of clusters from the elbow plot and fit KMeans
    diff = np.diff(distortions)
    diff_r = diff[1:] / diff[:-1]
    n_clusters = diff_r.argmin() + 2  # adding 2 because index starts at 0, and we're using diff_r

    print(f"Optimal number of clusters from elbow method: {n_clusters}")

    print(f"Fitting K-means with optimal number of clusters ({n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X_scaled)

    # Append the cluster labels to the original DataFrame
    df['cluster'] = kmeans.labels_
    print("Added cluster labels to dataframe")

    print("K-means clustering complete!")
    return df
########################################################################################################################




#CLUSTERS AND LABEL ENCODED VARIABLES TABLE CREATOR FUNCTION FOR INTERPRETATION
########################################################################################################################
#This function visualizes the results of clustering. It takes in a dataframe  that has been clustered, which means
#it contains a 'cluster' column indicating the cluster each row (video) belongs to. It also takes in a list of
#label-encoded features (label_encoded_features).
#The function creates a new dataframe (df_cluster_info) containing the video id, the cluster each video belongs to,
#and the label-encoded features for each video. This dataframe provides a comprehensive view of how the videos have
#been grouped by the clustering algorithm.
#The function prints this new dataframe to the console for immediate viewing, and also saves it as a CSV file in a
#specified location (in this case, the Desktop) for later analysis.
#After these actions, the function returns df_cluster_info.


def visualize_clusters(df, label_encoded_features):
    """
    This function visualizes the results of clustering by creating a dataframe
    with the video id, cluster assignment and label encoded features. This dataframe
    is printed to the console and also saved as a CSV file.

    Parameters:
    df (pandas.DataFrame): DataFrame that has been clustered, i.e., has a 'cluster' column
    label_encoded_features (list): A list of column names that are label-encoded features

    Returns:
    df_cluster_info (pandas.DataFrame): DataFrame containing video id, cluster assignment and label encoded features
    """

    print("Visualizing clusters...")

    # Create a new dataframe with video id, cluster assignment and label encoded features
    df_cluster_info = df[["video_id", "cluster"] + label_encoded_features].copy()

    # Print this dataframe to the console
    print(df_cluster_info)

    # Save this dataframe as a CSV file
    df_cluster_info.to_csv("C:/Users/YourUsername/Desktop/Clustered_Analysis.csv", index=False)

    print("Visualization complete!")

    return df_cluster_info


########################################################################################################################
label_encoded_features = ["Label", "Topic_Labels_Comments", "Topic_Labels_Recognized_Speeches", "Dominant Emotion"]
general_data_correlation_analysis = pd.read_csv("C:/Users/Uygar TALU/Desktop/ready_final_data_correlation_analysis.csv")

#EXECUTION
df = run_kmeans(df=general_data_correlation_analysis,
                k_range=(1, 100),
                label_encoded_features=label_encoded_features)

#EXECUTION
df_cluster_info = visualize_clusters(df=general_data_correlation_analysis,
                                     label_encoded_features=label_encoded_features)
########################################################################################################################


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################


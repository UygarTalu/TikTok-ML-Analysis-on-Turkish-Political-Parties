########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################


################## ANOMALY DETECTION ANALYSIS ON ENGAGEMENT METRIC - ANALYSIS LAYER 1 ##################################

###INFORMATION ABOUT THE SCRIPT(ANALYSIS LAYER 1)
"""
IN THE ANALYSIS LAYER ONE WE HAVE 3 DIFFERENT FUNCTIONS.
1- "standardscaler_general"
2- "Anomaly_Detection"
3- "visualize_impactful_videos"

Results for Layer 1- In this layer of analysis, we apply standard scaler function on the all the metadata csv files for
the scrapped videos. We first standardize the engagement metrics of each video for each hashtags and the user profiles
and then create one variable called as "engagement_metric_Standardized" where each engagement metric of the videos
contribute equally. With the help of the second function we apply isolation forest algorithm to find out the significant
videos in terms of their engagement metrics. Those significant videos indicate the ones that have significant impact in
terms of engagement metrics. At the end of the analysis we obtain the anomaly detected videos and manually filter their
metadata based csv files and the videos itself to use them as inputs in the layer 2 where we apply face detection and
emotion recognition.
"""

#### MODELS OR ALGORITHMS USED IN ANALYSIS LAYER 4 - Standard Scaler - Isolation Forest
########################################################################################################################
########################################################################################################################


#FUNCTION-1:  #STANDART SCALER FUNCTION
########################################################################################################################

#This function performs standard scaling on specific columns of CSV files in a given input directory. It takes the
#input path (directory containing the CSV files) and the output path
#(directory where the standardized files will be saved) as input.

def standardscaler_general(input_path, output_path):
    """
    Apply standard scaling to specific columns of CSV files in the input directory and save the standardized files.

    Args:
        input_path (str): Path to the directory containing the CSV files.
        output_path (str): Path to the directory where the standardized files will be saved.

    Returns:
        None
    """

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    files = os.listdir(input_path)

    for file in files:
        dataframe = pd.read_csv(os.path.join(input_path, file))

        dataframe['video_timestamp'] = pd.to_datetime(dataframe['video_timestamp'])
        #dataframe["video_id"] = dataframe["video_id"].astype("int64")

        columns_to_standardize = ['video_diggcount', 'video_sharecount', 'video_commentcount', 'video_playcount']

        standardscaler = StandardScaler()
        dataframe[columns_to_standardize] = standardscaler.fit_transform(dataframe[columns_to_standardize])

        dataframe['engagement_metric_all'] = dataframe[columns_to_standardize].mean(axis=1)

        base_filename = os.path.splitext(file)[0]

        output_filename = base_filename + "_STANDARDIZED.csv"

        dataframe.to_csv(os.path.join(output_path, output_filename), index=False)
########################################################################################################################



#FUNCTION-2:  #ANOMALY DETECTION FUNCTION - ISOLATION FOREST(RULE BASED)
########################################################################################################################

#The Anomaly_Detection function performs anomaly detection on CSV files in the input directory using the Isolation Forest
#algorithm. It imports each file, preprocesses the data, applies the Isolation Forest model to detect anomalies, and
#saves the detected anomalies to separate files in the output directory. The function returns a dictionary that maps
#each processed file name to its corresponding anomaly-detected DataFrame.

def Anomaly_Detection(input_path, output_path):
    """
    Perform anomaly detection on CSV files in the input directory using the Isolation Forest algorithm,
    and save the detected anomalies to separate files.

    Args:
        input_path (str): Path to the directory containing the CSV files.
        output_path (str): Path to the directory where the anomaly-detected files will be saved.

    Returns:
        dict: A dictionary mapping each processed file name to its corresponding anomaly-detected DataFrame.
    """

    files = os.listdir(input_path)
    total_files = len(files)
    processed_files = 0
    total_rows = 0
    total_anomalies = 0
    anomaly_dict = {}

    print("TOTAL FILES TO PROCESS:", total_files)

    for file in files:
        print("IMPORTING FILE:", file)
        dataframe = pd.read_csv(os.path.join(input_path, file))
        original_dataframe = dataframe.copy()

        total_rows += len(dataframe)

        dataframe['video_timestamp'] = pd.to_datetime(dataframe['video_timestamp'])
        dataframe['video_timestamp'] = dataframe['video_timestamp'].apply(lambda x: x.timestamp())
        features_take = dataframe[["video_id", "engagement_metric_all", "video_timestamp"]]

        model_anomaly_detection = IsolationForest(contamination=0.02)
        print("APPLYING ISOLATION FOREST MODEL...")
        model_anomaly_detection.fit(features_take)
        predicted_anomaly = model_anomaly_detection.predict(features_take)
        dataframe['Anomaly'] = predicted_anomaly

        original_dataframe['Anomaly'] = predicted_anomaly
        anomalies_total = original_dataframe[original_dataframe['Anomaly'] == -1]
        num_anomalies = len(anomalies_total)
        total_anomalies += num_anomalies
        print("DETECTED ANOMALIES:", num_anomalies)

        anomaly_filename = file.split('.')[0] + "-Anomaly_Detected.csv"
        anomaly_filepath = os.path.join(output_path, anomaly_filename)
        anomalies_total.to_csv(anomaly_filepath, index=False)
        print("ANOMALY DETECTED FILE SAVED:", anomaly_filename)

        anomaly_dict[file] = anomalies_total
        processed_files += 1

    print("TOTAL FILES PROCESSED:", processed_files)
    print("TOTAL ROWS IMPORTED:", total_rows)
    print("TOTAL ANOMALY DETECTED FILES SAVED:", len(anomaly_dict))
    print("TOTAL ANOMALIES DETECTED:", total_anomalies)
    anomaly_percentage = (total_anomalies / total_rows) * 100
    print("PERCENTAGE OF ANOMALY DETECTED DATA:", round(anomaly_percentage, 2), "%")

    return anomaly_dict
########################################################################################################################



#FUNCTION-3:  #VISUALIZATION OF THE ANOMALY DETECTED VIDEOS
########################################################################################################################

def visualize_impactful_videos(anomaly_dict):
    for file, anomalies in anomaly_dict.items():
        video_ids = anomalies['video_id']
        engagement_metrics = anomalies['engagement_metric_all']

        plt.figure(figsize=(12, 6))
        plt.bar(video_ids, engagement_metrics)
        plt.xlabel('Video ID')
        plt.ylabel('Engagement Metric')
        plt.title('Impactful Videos - Anomaly Detected')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

########################################################################################################################


#### EXECUTION OF THE FUNTIONS
########################################################################################################################
input_path = "C:/Users/Uygar TALU/Desktop/WEBSCRAPPING_THESIS/METADA DIVIDED/All Metadata CSV Files"
output_path = "C:/Users/Uygar TALU/Desktop/WEBSCRAPPING_THESIS/METADA DIVIDED/All Metadata CSV Files - STANDARDIZED"

standardscaler_general(input_path, output_path)



input_path = "C:/Users/Uygar TALU/Desktop/WEBSCRAPPING_THESIS/METADA DIVIDED/All Metadata CSV Files - STANDARDIZED"
output_path = "C:/Users/Uygar TALU/Desktop/WEBSCRAPPING_THESIS/METADA DIVIDED/All Metadata CSV Files - STANDARDIZED - ANOMALY DETECTED"

anomaly_detected = Anomaly_Detection(input_path=input_path, output_path=output_path)


anomaly_dict = Anomaly_Detection(input_path, output_path)
visualize_impactful_videos(anomaly_dict)
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



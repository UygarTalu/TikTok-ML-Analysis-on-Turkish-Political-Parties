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






############################## ALL PACKAGE IMPORTS AND INSTALLATIONS USED IN THE PROJECT ###############################


##### PACKAGE INSTALLATIONS #####

pip install deepface
pip install pydub
pip install SpeechRecognition
pip install opencv-python
pip install moviepy
pip install numpy
pip install pandas
pip install soundfile
pip install deepspeech
pip install matplotlib
pip install scipy
pip install scikit-learn
pip install gensim
pip install nltk
pip install spacy
pip install stop-words
pip install wordcloud
pip install keras
pip install tensorflow
pip install pyktok
pip install Pillow
pip install python-docx

###### REQUIRED LIBRARIES IMPORTS ######

import ast
import csv
import cv2
import deepspeech
import glob
import imageio
import io
import json
import matplotlib.pyplot as plt
import moviepy.editor as mp
import numpy as np
import pandas as pd
import soundfile as sf
import sys
import zipfile
import requests
from IPython.display import Image, display
from PIL import Image as PIL_Image
from deepface import DeepFace
from deepface.commons import functions
from gensim import corpora
from gensim.models import CoherenceModel, LdaModel
import nltk
from nltk.corpus import brown, stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import pyktok as pyk
import spacy
from stop_words import get_stop_words
import string
import textblob
from textblob_tr import TextBlob as BlobTR
import gensim
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.losses import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import pydub
from pydub import AudioSegment, silence
import speech_recognition as sr
import cv2
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from deepface import DeepFace
from pydub.silence import split_on_silence
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Flatten, Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import load_img
import os
from deepface import DeepFace
download_corpora.main()
download_corpora()
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import ast
import re
import csv
import cv2
import deepspeech
import glob
import imageio
import io
import json
import matplotlib.pyplot as plt
import moviepy.editor as mp
import numpy as np
import pandas as pd
import soundfile as sf
import sys
import zipfile
import requests
from IPython.display import Image, display
from PIL import Image as PIL_Image
from deepface import DeepFace
from deepface.commons import functions
from gensim import corpora
from gensim.models import CoherenceModel, LdaModel
import nltk
from nltk.corpus import brown, stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import pyktok as pyk
import spacy
from stop_words import get_stop_words
import string
import textblob
from textblob_tr import TextBlob as BlobTR
import gensim
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.losses import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import pydub
from pydub import AudioSegment, silence
import speech_recognition as sr
import cv2
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from deepface import DeepFace
from pydub.silence import split_on_silence
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Flatten, Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import load_img
import os
from deepface import DeepFace
download_corpora.main()
download_corpora()
import nltk
nltk.download('all')
from pydub.utils import mediainfo
import platform
print(platform.architecture())
import os
import sys
import glob
import deepspeech
import soundfile as sf
from moviepy.editor import VideoFileClip
from pandas import read_excel

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################






################################## ANOMALY DETECTION ANALYSIS LAYER 1 ##################################################

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








################################## FACE RECOGNITION AND EMOTION RECOGNITION ANALYSIS LAYER 2 ###########################

#FUNCTION-1:  #VIDEO PATHS CAPTURING FUNCTION
########################################################################################################################

#The get_video_paths function retrieves the paths of all video files within a given directory and its subdirectories.
#It walks through the directory tree using os.walk, checks if each file has a ".mp4" extension, and constructs the full
#path of the video file. The function returns a list of video file paths.

def get_video_paths(root_dir):
    """
    Retrieves the paths of all video files within a given directory and its subdirectories.

    Args:
        root_dir (str): Root directory to start searching for video files.

    Returns:
        list: List of video file paths.
    """

    video_paths = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.mp4'):
                # Construct the full path of the video
                video_path = os.path.join(dirpath, filename)
                # Add the video path to the list
                video_paths.append(video_path)

    return video_paths
########################################################################################################################



#FUNCTION-2:  #VIDEO PATHS EXECUTION FUNCTION
########################################################################################################################

#The get_all_video_paths_video_specific function retrieves the paths of all video files within multiple directories.
#It takes a list of root directories and calls the get_video_paths function for each root directory. It extends the
#all_video_paths list with the video paths obtained from each root directory.
#The function returns a list of video file paths

def get_all_video_paths_video_specific(root_dirs):
    """
    Retrieves the paths of all video files within multiple directories.

    Args:
        root_dirs (list): List of root directories to search for video files.

    Returns:
        list: List of video file paths.
    """

    all_video_paths = []

    for root_dir in root_dirs:
        video_paths = get_video_paths(root_dir)
        all_video_paths.extend(video_paths)

    return all_video_paths
########################################################################################################################



#FUNCTION-3:  #VIDEO ID CAPTURING FUNCTION
########################################################################################################################
#The extract_video_label function extracts the video label from the video path. It checks
#if the video path contains "#" or "@". If "#" is found, it finds the index of "#" and extracts the
#label between "#" and the following space character. If "@" is found, it finds the index of "@" and extracts the
#label between "@" and the following space character. If neither "#" nor "@" is found, it returns None. The function
#returns the extracted video label or None.

def extract_video_label(video_path):
    """
    Extracts the video label from the video path.

    Args:
        video_path (str): Path of the video file.

    Returns:
        str or None: Extracted video label if found, otherwise None.
    """

    if '#' in video_path:
        label_start = video_path.index('#') + 1
        label_end = video_path.index(' ', label_start)
        return video_path[label_start:label_end]
    elif '@' in video_path:
        label_start = video_path.index('@') + 1
        label_end = video_path.index(' ', label_start)
        return video_path[label_start:label_end]
    else:
        return None
########################################################################################################################

#FUNCTION-4:  #EMOTION RECIGNITION FUNCTION - (FACE DETECTION & EMOTION DET & RESULTS FILE & RESULTS PERCENTAGES)
########################################################################################################################

#The auto_emotion_recognition function performs automatic emotion recognition on video files using face detection and
#emotion detection. It iterates over the directories and filenames within the base_path directory.
#For each video file, the function initializes variables to track emotion counts, emotion sums, and other statistics.
#It captures video frames, performs face detection, and analyzes emotions using the DeepFace library. The detected
#emotions are counted, and the emotion sums are accumulated. The function also prints progress information.
#After processing all video files, the function constructs a result dictionary containing the emotion recognition
#results for each video file. It also generates a CSV file containing the summarized results. The CSV file includes
#information such as video count, total frames, the number of frames with face detection, emotion counts, dominant
#emotion, and confidence levels for each emotion.
#The function returns the result_dict containing the emotion recognition results for each video file.

def auto_emotion_recognition(base_path, start_folder, cascade_classifier_path, output_name, output_file_path):
    """
        This function auto-generates emotion recognition data from a base directory of videos.

        Parameters:
        base_path (str): The base directory path where the folders of videos are located.
        start_folder (str): The name of the folder from where to start processing.
        cascade_classifier_path (str): The path to the Haar Cascade XML file to use for face detection.
        output_name (str): The name of the output file (without extension).
        output_file_path (str): The directory where the output file will be saved.

        Returns:
        dict: A dictionary that contains the emotion recognition results.
              Each key is the path to a video file, and its corresponding value is another dictionary with various statistics,
              such as total frames, frames where a face was detected, emotion counts, emotion percentages,
              emotion sums, emotion averages, dominant emotion and the label of the video.

        Note:
        This function uses OpenCV for video processing and face detection,
        and the DeepFace library for emotion recognition.
        """
    result_dict = {}
    video_counter = 1
    start_processing = False

    total_folders = sum([len(dir_names) for _, dir_names, _ in os.walk(base_path)])

    for dir_path, dir_names, filenames in os.walk(base_path):
        if filenames:
            folder_name = dir_path.split("\\")[-1]
            if folder_name == start_folder:
                start_processing = True
            if not start_processing:
                continue

            label_start = folder_name.index("#") + 1 if "#" in folder_name else folder_name.index("@") + 1
            label_end = folder_name.index("-")
            label = folder_name[label_start:label_end]
            print(f"Label extracted: {label}")

            total_videos = len(filenames)

            video_paths = [os.path.join(dir_path, filename) for filename in filenames if filename.endswith(".mp4")]

            print(f"\nEntering folder: {folder_name}")
            print(f"Total videos in folder: {total_videos}")

            for video_path in video_paths:
                try:
                    emotion_counts = {
                        "angry": 0,
                        "disgust": 0,
                        "fear": 0,
                        "happy": 0,
                        "sad": 0,
                        "surprise": 0,
                        "neutral": 0
                    }

                    emotion_sums = {
                        "angry": 0,
                        "disgust": 0,
                        "fear": 0,
                        "happy": 0,
                        "sad": 0,
                        "surprise": 0,
                        "neutral": 0
                    }

                    capture_video = cv2.VideoCapture(video_path)
                    total_frames = int(capture_video.get(cv2.CAP_PROP_FRAME_COUNT))
                    face_model = cv2.CascadeClassifier(cascade_classifier_path)

                    frame_counter = 0
                    face_detected_counter = 0

                    print(f"\nProcessing video: {video_path}")
                    print(f"Total frames: {total_frames}")

                    for i in range(total_frames):
                        ret, frame = capture_video.read()

                        if not ret:
                            break

                        try:
                            face_detection = face_model.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1,
                                                                         5)
                        except cv2.error as e:
                            print(f"Frame: {i}, Error: {str(e)}")
                            continue

                        frame_counter += 1

                        if len(face_detection) > 0:
                            face_detected_counter += 1

                            try:
                                emotion_detection = DeepFace.analyze(frame, actions=["emotion"],
                                                                     enforce_detection=False)
                                emotion_probabilities = emotion_detection[0]['emotion']
                                dominant_emotion = emotion_detection[0]['dominant_emotion']

                                emotion_counts[dominant_emotion] += 1

                                for emotion, probability in emotion_probabilities.items():
                                    emotion_sums[emotion] += probability
                            except ValueError as e:
                                print(f"Frame: {frame_counter}, Error: {str(e)}")

                        print(f"Processed frame: {frame_counter}/{total_frames}", end="\r")

                    if face_detected_counter > 0:
                        emotion_percentages = {emotion: (count / face_detected_counter) * 100 for emotion, count in
                                               emotion_counts.items()}
                        emotion_averages = {emotion: (sum_ / face_detected_counter) for emotion, sum_ in
                                            emotion_sums.items()}
                    else:
                        emotion_percentages = {emotion: 0 for emotion in emotion_counts.keys()}
                        emotion_averages = {emotion: 0 for emotion in emotion_counts.keys()}

                    result_dict[video_path] = {
                        "total_frames": total_frames,
                        "face_detected_frames": face_detected_counter,
                        "emotion_counts": emotion_counts,
                        "emotion_percentages": emotion_percentages,
                        "emotion_sums": emotion_sums,
                        "emotion_averages": emotion_averages,
                        "dominant_emotion": max(emotion_counts, key=emotion_counts.get),
                        "Label": label
                    }

                    print(f"\nProcessed video: {video_path}")
                    print(f"Total frames: {total_frames}")
                    print(f"Frames where face detected: {face_detected_counter}")
                    print(f"Number of frames where face is not detected: {total_frames - face_detected_counter}")
                    print(f"Emotion counts: {emotion_counts}")
                    print(f"Dominant emotion: {max(emotion_counts, key=emotion_counts.get)}")
                    print("\n")

                    video_counter += 1

                    if video_counter % 5 == 0:
                        continue_process = input("Should we continue processing audio files? (yes/no): ")
                        if continue_process.lower() != "yes":
                            break

                    output_file = os.path.join(output_file_path, f"{output_name}.csv")

                    if os.path.isfile(output_file):
                        df = pd.read_csv(output_file)
                    else:
                        df = pd.DataFrame()

                    data = result_dict[video_path]
                    no_face_detected_frames = data['total_frames'] - data['face_detected_frames']

                    result = {
                        "Video Count": "video_" + str(video_counter),
                        "Total Frame Of The Video": data['total_frames'],
                        "Number of Frames Face Detected": data['face_detected_frames'],
                        "Number of Frames Face Is Not Detected": no_face_detected_frames,
                        "Angry": data['emotion_counts']['angry'],
                        "Disgust": data['emotion_counts']['disgust'],
                        "Fear": data['emotion_counts']['fear'],
                        "Happy": data['emotion_counts']['happy'],
                        "Sad": data['emotion_counts']['sad'],
                        "Surprise": data['emotion_counts']['surprise'],
                        "Neutral": data['emotion_counts']['neutral'],
                        "Dominant Emotion": data['dominant_emotion'],
                        "Angry (Confidence level)": data['emotion_sums']['angry'] / data['face_detected_frames'] if
                        data['face_detected_frames'] > 0 else 0,
                        "Disgust (Confidence level)": data['emotion_sums']['disgust'] / data['face_detected_frames'] if
                        data['face_detected_frames'] > 0 else 0,
                        "Fear (Confidence level)": data['emotion_sums']['fear'] / data['face_detected_frames'] if data[
                                                                                                                      'face_detected_frames'] > 0 else 0,
                        "Happy (Confidence level)": data['emotion_sums']['happy'] / data['face_detected_frames'] if
                        data['face_detected_frames'] > 0 else 0,
                        "Sad (Confidence level)": data['emotion_sums']['sad'] / data['face_detected_frames'] if data[
                                                                                                                    'face_detected_frames'] > 0 else 0,
                        "Surprise (Confidence level)": data['emotion_sums']['surprise'] / data[
                            'face_detected_frames'] if data['face_detected_frames'] > 0 else 0,
                        "Neutral (Confidence level)": data['emotion_sums']['neutral'] / data['face_detected_frames'] if
                        data['face_detected_frames'] > 0 else 0,
                        "Label": data['Label'],
                    }
                    df = df.append(result, ignore_index=True)
                    df.to_csv(output_file, index=False)

                except Exception as e:
                    print(f"An error occurred while processing the video: {str(e)}")

            print(f"Total results gathered: {len(result_dict)}")

            try:
                output_file = os.path.join(output_file_path, f"{output_name}.csv")
                print(f"CSV file saved to: {output_file}")
            except Exception as e:
                print(f"An error occurred while saving the CSV file: {str(e)}")

    return result_dict
########################################################################################################################



#### EXECUTION OF THE FUNCTIONS
########################################################################################################################
base_path = "C:\\Users\\Uygar TALU\\Desktop\\WEBSCRAPPING_THESIS\\METADA DIVIDED\\ANOMALY DETECTED - VIDEOS"
cascade_classifier_path = "C:/Users/Uygar TALU/Desktop/haarcascade_frontalface_default.xml"
output_name = "Emotion_Detection-Face_Detection_Results.csv"
output_file_path = "C:/Users/Uygar TALU/Desktop/"

result_dict = auto_emotion_recognition(base_path=base_path,
                                       cascade_classifier_path=cascade_classifier_path,
                                       output_name=output_name,
                                       output_file_path=output_file_path)
########################################################################################################################



########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################







################################## SPEECH RECOGNITION ANALYSIS LAYER 3 #################################################

#FUNCTION-1:  #FUNCTION TO EXTRACT AUDIO FILE FROM THE VIDEOS THAT ARE SCRAPPED
########################################################################################################################

#The audio_extractor function extracts audio files from the videos in the specified base_video_dir directory.
#It saves the extracted audio files in the base_audio_dir directory. The function traverses through each video folder,
#processes the videos one by one, extracts the audio using MoviePy, and saves the extracted audio files.
#The function returns the paths of all processed video files and audio files.
#The function expects the video files to be in the ".mp4" format and follows a specific naming convention for the video
#folders. It extracts the label from the folder name and uses it to name the extracted audio file.
#During the processing, the function provides information about the progress, including the number of videos to process,
#the current video being processed, and the duration of the extracted audio. It also prompts the user to continue with
#the next folder or terminate the process.
#Once all videos are processed, the function prints a completion message and returns the paths of the
#processed video files and audio files.

def audio_extractor(base_video_dir, base_audio_dir):
    """
       Extracts audio from videos in a directory, and saves them to another directory.

       Args:
           base_video_dir (str): The path of the directory containing the video files.
           base_audio_dir (str): The path of the directory to save the extracted audio files.

       Returns:
           Tuple[List[str], List[str]]: The paths of the processed video files and the paths of the extracted audio files.
       """
    import os
    import sys
    import glob
    from moviepy.editor import VideoFileClip

    try:
        video_folder_paths = [f.path for f in os.scandir(base_video_dir) if f.is_dir()]
    except FileNotFoundError as e:
        print(f"Error finding video directory: {e}")
        sys.exit(1)

    all_video_paths = []
    all_audio_paths = []
    video_counter = 0
    folder_counter = 0

    for video_folder_path in video_folder_paths:
        folder_counter += 1
        print(f"\nTraversing through folder: {video_folder_path}")

        folder_name = os.path.basename(video_folder_path)
        label_start = folder_name.index("#") + 1 if "#" in folder_name else folder_name.index("@") + 1
        label_end = folder_name.index("-")
        label = folder_name[label_start:label_end]

        print(f"\nProcessing videos in folder: {folder_name}")

        video_paths = glob.glob(os.path.join(video_folder_path, '*.mp4'))
        print(f"Number of videos to process in this folder: {len(video_paths)}")

        for i, video_path in enumerate(video_paths, start=1):
            video_counter += 1
            print(f"\nExtracting audio from video: {video_path}")
            print(f"Processing video {i} out of {len(video_paths)}")

            all_video_paths.append(video_path)
            audio_filename = f"{label}EXTRACTED_AUDIO{i}.wav"  # Including counter in filename

            try:
                clip = VideoFileClip(video_path)
                print(f"Audio duration: {clip.duration} seconds")

                audio_path = os.path.join(base_audio_dir, audio_filename)
                clip.audio.write_audiofile(audio_path)
                all_audio_paths.append(audio_path)

                print(f"\nAudio extracted and saved to: {audio_path}")
                print(f"Finished processing video {i} out of {len(video_paths)}")
            except Exception as e:
                print(f"Failed to extract audio from video {video_path}. Error: {str(e)}")
                continue

            if i % 5 == 0:
                user_input = input(f"\n{i} videos are extracted. Shall we continue? (Type 'yes' to continue): ")
                if user_input.lower() != 'yes':
                    print("\nProcess terminated by user.")
                    sys.exit(0)

        print(f"\nFinished processing all {len(video_paths)} videos in folder: {folder_name}")

        if video_folder_path != video_folder_paths[-1]:
            user_input = input(
                "\nFinished processing current folder. Type 'yes' to continue with the next folder, 'no' to quit: ")
            if user_input.lower() == 'no':
                print("\nProcess terminated by user.")
                sys.exit(0)

    print("\nAudio extraction completed.")
    print(f"Number of folders entered: {folder_counter}")
    print(f"Total number of video files processed: {video_counter}")
    return all_video_paths, all_audio_paths
########################################################################################################################


#FUNCTION-2:  #SPEECH RECOGNITION FROM EXTRACTED AUDIO
########################################################################################################################

#The speech_to_text function transcribes audio files to text using the DeepSpeech model. It takes the base_audio_dir as
#the path to the directory containing the audio files, output_csv_path as the path to the directory to save the output.
#The function iterates through each audio folder in base_audio_dir. For each folder, it retrieves the audio files and
#initializes variables to track the number of successful and failed transcriptions. It then processes each audio file
#by reading the file, transcribing the audio using the Google's speech recognition  model, and storing the transcription in the
#recognized_speeches dictionary. The function also handles exceptions if an audio file cannot be transcribed.
#After processing all audio files in a folder, the function saves the recognized speeches in a CSV file named after the
#folder. It creates a DataFrame from the recognized_speeches dictionary
#and saves it to a CSV file in the output_csv_path directory.
#The function prints information about the progress, including the number of audio files and folders to process,
#the current audio file being processed, the number of successful and failed transcriptions, and the completion message.
#Once all folders and audio files are processed, the function prints the total number of audio files processed,
#the number of successful transcriptions, the number of failed transcriptions, and the number of folders processed.


def speech_to_text_google(audio_folder_path, output_csv_path, num_files_per_batch=25):
    """
        Transcribes audio files to text using Google's Speech Recognition.
        Saves the transcriptions in a CSV file for the given folder.

        Args:
            audio_folder_path (str): Path of the directory containing the audio files.
            output_csv_path (str): Path of the directory to save the output CSV file.
            num_files_per_batch (int): Number of files to process per batch. Default is 25.

        Returns:
            None
        """
    import os
    import glob
    import sys
    import pandas as pd
    import speech_recognition as sr

    recognizer = sr.Recognizer()

    audio_files = glob.glob(os.path.join(audio_folder_path, '*.wav'))
    print(f"Number of audio files to process in this folder: {len(audio_files)}")

    user_input = input("\nShall we start processing? (Type 'yes' to start): ")
    if user_input.lower() != 'yes':
        print("\nProcess terminated by user.")
        sys.exit(0)

    recognized_speeches = {}
    successful_transcriptions = 0
    failed_transcriptions = 0
    audio_counter = 0

    for batch_start in range(0, len(audio_files), num_files_per_batch):
        batch_end = batch_start + num_files_per_batch
        batch_files = audio_files[batch_start:batch_end]
        batch_size = len(batch_files)

        print(f"\nProcessing Batch {batch_start//num_files_per_batch + 1}")
        print(f"Number of audio files in this batch: {batch_size}")

        for i, audio_file in enumerate(batch_files, start=1):
            audio_counter += 1
            print(f"\nProcessing Audio File {i} out of {batch_size}: {audio_file}")

            audio_file_name = os.path.basename(audio_file)
            label_end = audio_file_name.index("_")
            label = audio_file_name[:label_end] + "_AUDIO"

            try:
                print(f"\nReading audio file {audio_file_name}...")
                with sr.AudioFile(audio_file) as audio_source:
                    audio_data = recognizer.record(audio_source)
                print(f"Audio file {audio_file_name} read successfully.")
            except Exception as e:
                print(f"\nCould not read {audio_file_name}. Error: {e}")
                user_input = input("\nThere was an error processing this audio file. Do you want to continue with the next file? (Type 'yes' to continue): ")
                if user_input.lower() != 'yes':
                    print("\nProcess terminated by user.")
                    sys.exit(0)
                continue

            try:
                print(f"Transcribing audio file {audio_file_name}...")
                transcription = recognizer.recognize_google(audio_data, language='tr-TR')
                recognized_speeches[label] = transcription
                successful_transcriptions += 1
                print(f"\nTranscription for {audio_file_name} completed.")
            except Exception as e:
                print(f"\nCould not transcribe {audio_file_name}. Error: {e}")
                user_input = input("\nThere was an error transcribing this audio file. Do you want to continue with the next file? (Type 'yes' to continue): ")
                if user_input.lower() != 'yes':
                    print("\nProcess terminated by user.")
                    sys.exit(0)
                failed_transcriptions += 1
                continue

        if batch_end < len(audio_files):
            user_input = input(f"\nProcessed {batch_size} audio files. Shall we continue with the next batch? (Type 'yes' to continue): ")
            if user_input.lower() != 'yes':
                print("\nProcess terminated by user.")
                sys.exit(0)

    pd.DataFrame(list(recognized_speeches.items()), columns=['filename', 'transcription']).to_csv(output_csv_path, index=False)
    print(f"\nTranscription for folder {audio_folder_path} completed. Results saved to '{output_csv_path}'.")
    print(f"Total number of audio files processed: {successful_transcriptions + failed_transcriptions}")
    print(f"Number of successful transcriptions: {successful_transcriptions}")
    print(f"Number of failed transcriptions: {failed_transcriptions}")
    print(f"Number of folders entered: 1")
    print(f"Total number of audio files processed: {audio_counter}")

########################################################################################################################



#### EXECUTION OF THE FUNCTIONS
########################################################################################################################
base_video_dir = "C:/Users/Uygar TALU/Desktop/WEBSCRAPPING_THESIS/METADA DIVIDED/ANOMALY DETECTED - VIDEOS"
base_audio_dir = "C:/Users/Uygar TALU/Desktop/WEBSCRAPPING_THESIS/METADA DIVIDED/EXTRACTED AUDIO FILES"
output_csv_path = "C:/Users/Uygar TALU/Desktop/"

video_paths, audio_paths = audio_extractor(base_video_dir = base_video_dir , base_audio_dir = base_audio_dir)
speech_recognition_results = speech_to_text_google(base_audio_dir = base_audio_dir,
                                                   output_csv_path = output_csv_path,
                                                   num_files_per_batch=25)
########################################################################################################################


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################








################################## NLP ANALYSIS ON COMMENTS ANALYSIS LAYER 4 ###########################################

#FUNCTION-1:  #TEXT PREPROCESSOR FUNCTION
########################################################################################################################
# TEXT PREPROCESSING FUNCTION - (Stopwords-stemming-lowercase-punctuation)
# This function preprocesses the given text by removing stopwords, punctuation, and lowercasing and tokenizing the words.
# If any error occurs during the text preprocessing, an exception is raised.

def text_preprocessor(text):
    """
    Preprocess a given text by removing non-alphanumeric characters, punctuation marks,
    converting the text to lowercase, tokenizing the text into words and removing stopwords.

    This function is specifically designed to handle Turkish texts.

    Args:
        text (str): The input text string to be preprocessed.

    Returns:
        list: A list of preprocessed tokenized words.

    Raises:
        ValueError: If the input is not a string.
        Exception: If an error occurs during text preprocessing.
    """
    if not isinstance(text, str):
        raise ValueError(f"Expected string but received {type(text)}")
    try:
        # Downloading Turkish stopwords
        stop_words = set(get_stop_words("turkish"))

        # Add your specific words to the stop words set
        custom_words = ["reis", "cumhur", "cumhurbaşkanı", "kazandı", "kazanmak", "oy", "kemal", "baykemal",
                        "sanasöz", "kemal kılıçdaroğlu", "Kemal Kılıçdaroğlu", "Recep Tayyip Erdoğan", "rte",
                        "reisicumhur", "secim", "seçim"]
        stop_words.update(custom_words)

        # Removing non-alphanumeric characters
        text_input = re.sub(r'\W+', ' ', text)

        # Removing punctuation marks
        text_input = text_input.translate(str.maketrans("", "", string.punctuation))

        # Converting the text to lowercase
        text_input = text_input.lower()

        # Tokenizing the text into words
        tokens = word_tokenize(text_input)

        # Removing stopwords
        tokens = [token for token in tokens if token not in stop_words]

        return tokens  # Returning the tokens as a list instead of a string
    except Exception as e:
        print(f"Error during text preprocessing: {e}")
        input("Press enter to continue...")
########################################################################################################################




#FUNCTION-2:  #TEXT PREPROCESSING EXECUTION FUNCTION
########################################################################################################################
#The process_text_data function takes a list of speech data comments and preprocesses each comment by
#applying the text_preprocessor function. It returns a list of processed comments where each comment is tokenized.

def process_text_data(speech_data):

    """
        Preprocesses the speech data by applying the text_preprocessor function to each comment.
        Returns a list of processed comments where each comment is tokenized.

        Args:
            speech_data (list): List of speech data comments.

        Returns:
            list: List of processed comments where each comment is tokenized.
        """

    processed_data_list = []
    for comment in speech_data:
        preprocessed_comment = text_preprocessor(comment)
        processed_data_list.append(preprocessed_comment)
    return processed_data_list
########################################################################################################################




#FUNCTION-3:  #SINGLE CSV FILE PROCESSOR
########################################################################################################################
#The process_single_csv function reads and processes a single CSV file specified by the file_path argument.
#It reads the file using pd.read_csv and handles exceptions if the file is not found or there is a parsing error.
#It processes each column in the DataFrame by extracting the comments, applying process_text_data to preprocess
#the comments, and stores the processed comments in a dictionary. The function returns this dictionary mapping
#column names to their processed comments. Main reason is that we have this function is, for comments because of the
#nature of the parser the comments of the videos are extracted into csv files video by video so in the given folder
#I need to be able take each of the csv files as an input and then I need to be able to process them.

def process_single_csv(file_path):
    """
    Reads and processes a single CSV file.
    Returns a dictionary where each column name is mapped to its processed comments.

    Args:
        file_path (str): Path of the CSV file to be processed.

    Returns:
        dict: Dictionary mapping column names to their processed comments.
    """

    try:
        df = pd.read_csv(file_path)
    except (FileNotFoundError, pandas.errors.ParserError) as e:
        print(f"Error while reading the file {file_path}: {e}")
        input("Press enter to continue...")
        return {}

    processed_data = {}
    for column in df.columns:
        try:
            comments = df[column].dropna().tolist()
            preprocessed_comments = process_text_data(comments)
            processed_data[column] = preprocessed_comments
        except Exception as e:
            print(f"Error while processing column {column} in file {file_path}: {e}")
            input("Press enter to continue...")

    return processed_data
########################################################################################################################




#FUNCTION-4:  #TEXT PREPROCESSOR FUNCTION FOR ALL CSV FILES
########################################################################################################################
#The process_all_csv_in_dir function automates the processing for all CSV files in a directory specified by
#the dir_path argument. It lists all files in the directory, checks if each file has a ".csv" extension, and
#processes each CSV file using process_single_csv. The processed data from all files is stored in a dictionary where
#column names are mapped to their processed comments. The function returns this dictionary.

def process_all_csv_in_dir(dir_path):
    """
    Processes all CSV files in a directory.
    Returns a dictionary where each column name from all CSV files is mapped to its processed comments.

    Args:
        dir_path (str): Path of the directory containing the CSV files.

    Returns:
        dict: Dictionary mapping column names to their processed comments from all CSV files.
    """

    files = os.listdir(dir_path)
    all_processed_data = {}
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(dir_path, file)
            single_file_data = process_single_csv(file_path)
            all_processed_data.update(single_file_data)
    return all_processed_data
########################################################################################################################



#FUNCTION-5:  #CORPUS CREATOR FUNCTION
########################################################################################################################
def corpus_creator_doc2bow(processed_data_list):

    """
        Creates a document-term matrix (corpus) and a dictionary (id2word) from a combined list of processed data.

        Args:
            processed_data_list_combined (list): Combined list of processed data where each element is a list of tokens.

        Returns:
            tuple: A tuple containing the corpus and id2word dictionary.
        """


    id2word = corpora.Dictionary(processed_data_list)
    corpus = [id2word.doc2bow(tokens) for tokens in processed_data_list]
    return corpus, id2word
########################################################################################################################



#FUNCTION-6:  #COHERENCE VALUE OPTIMIZATION FUNCTION
########################################################################################################################
#The corpus_creator_doc2bow function creates a document-term matrix (corpus) and a dictionary (id2word) from a combined
#list of processed data specified by the processed_data_list_combined argument. It uses the corpora.Dictionary function
#to create the dictionary from the combined list of tokens. It then uses a list comprehension to convert each tokenized
#comment into a bag-of-words representation using doc2bow.
#The function returns a tuple containing the corpus and id2word dictionary.

def calculate_coherence_values(corpus, texts, dictionary, start, limit, step):

    """
            Calculate the coherence values for LDA models with different numbers of topics.

            This function iterates over a range of topic numbers (from start to limit, in steps of step),
            creates an LDA model for each number, calculates its coherence value, and stores it in a list.
            Each LDA model is also stored in a separate list. At the end, both lists are returned.

            Args:
                corpus (list of list of (int, float)): The document-term matrix (corpus) in bag-of-words format.
                texts (list of list of str): The tokenized, preprocessed texts used to create the corpus.
                dictionary (gensim.corpora.dictionary.Dictionary): The dictionary mapping words to their integer ids.
                start (int): The starting number of topics to consider.
                limit (int): The maximum number of topics to consider.
                step (int): The step size for the number of topics.

            Returns:
                tuple: A tuple of two lists: The first list contains the LDA models, the second contains the coherence values.
            """



    model_list = []
    coherence_values = []

    for num_topics in range(start, limit, step):
        print("Calculating for", num_topics, "Topics")

        lda_model = LdaModel(corpus=corpus,
                             num_topics=num_topics,
                             id2word=dictionary)

        model_list.append(lda_model)

        coherence_model = CoherenceModel(model=lda_model,
                                         texts=texts,
                                         dictionary=dictionary,
                                         coherence='c_v')

        coherence_values.append(coherence_model.get_coherence())
        print("Coherence Value:", coherence_model.get_coherence())

    return model_list, coherence_values
########################################################################################################################



#FUNCTION-7:  #LDA MODELING FUNCTION
########################################################################################################################
#The optimal_LDA_modeling function performs LDA (Latent Dirichlet Allocation) modeling to find the
#optimal number of topics. It takes a document-term matrix (corpus), preprocessed texts, and a dictionary as inputs.
#It uses the coherence_values function to determine the coherence values for different numbers of topics.
#The function selects the number of topics with the highest coherence value and builds an LDA model using LdaModel.
#It returns the optimal LDA model.

def optimal_LDA_modeling(corpus, texts, dictionary, start=1, limit=50, step=1):
    """
    Performs LDA (Latent Dirichlet Allocation) modeling to find the optimal number of topics.
    Returns the optimal LDA model.

    Args:
        corpus: The document-term matrix (corpus) in bag-of-words format.
        texts: The preprocessed texts used to create the corpus.
        dictionary: The dictionary mapping words to their integer ids.
        start (int): The starting number of topics to consider.
        limit (int): The maximum number of topics to consider.
        step (int): The step size for the number of topics.

    Returns:
        LdaModel: The optimal LDA model.
    """
    model_list = []
    coherence_values = []

    try:
        model_list, coherence_values = calculate_coherence_values(corpus, texts, dictionary, start, limit, step)
        optimal_num_topics = coherence_values.index(max(coherence_values)) + start
        optimal_LDA_model = LdaModel(corpus=corpus, num_topics=optimal_num_topics, id2word=dictionary)
        return optimal_LDA_model
    except Exception as e:
        print(f"Error during LDA modeling: {str(e)}")
        print(f"Model list: {model_list}")
        print(f"Coherence values: {coherence_values}")
        input("Press enter to continue...")
########################################################################################################################



#FUNCTION-8:  #SENTIMENT ANALYSIS FUNCTION
########################################################################################################################
#The sentiment_analysis_func function performs sentiment analysis on a list of comments. It uses the BlobTR sentiment
#analysis tool to calculate the sentiment polarity for each comment. The function returns the average sentiment score.

def sentiment_analysis_func(comments):
    """
    Performs sentiment analysis on a list of comments.
    Returns the average sentiment score.

    Args:
        comments (list): List of comments to analyze.

    Returns:
        float: Average sentiment score.
    """

    try:
        sentiments = [TextBlob(comment).sentiment.polarity for comment in comments]
        return sum(sentiments) / len(sentiments) if sentiments else 0
    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        input("Press enter to continue...")
########################################################################################################################



#FUNCTION-9:  #EXECUTION FUNCTION FOR NLP PART
########################################################################################################################
#The process_comments function processes comments from CSV files in a directory. It takes the files_path as input and
#optionally the number of words to show per topic in the LDA model (n_words). The function reads each CSV file,
#preprocesses the comments using text_preprocessor, creates a document-term matrix and dictionary using
#corpus_creator_doc2bow, performs LDA modeling using optimal_LDA_modeling, conducts sentiment analysis
#using sentiment_analysis_func, and saves the results to new CSV files.

def process_comments(files_path, output_path, n_words=10):
    try:
        files = os.listdir(files_path)
    except FileNotFoundError as e:
        print(f"Error while listing files in directory {files_path}: {e}")
        input("Press enter to continue...")
        return

    for file in files:
        if file.endswith('.csv'):
            try:
                df = pd.read_csv(os.path.join(files_path, file))
            except (FileNotFoundError, pandas.errors.ParserError) as e:
                print(f"Error while reading the file {file}: {e}")
                input("Press enter to continue...")
                continue

            results = []
            for column in df.columns:
                try:
                    comments = df[column].dropna().tolist()
                    processed_comments = process_text_data(comments)
                    corpus, dictionary = corpus_creator_doc2bow(processed_comments)
                    lda_model = optimal_LDA_modeling(corpus, processed_comments, dictionary)
                    topics = lda_model.show_topics(formatted=False)
                    topics_words = [[word[0] for word in topic[1]] for topic in topics]
                    sentiment = sentiment_analysis_func(comments)
                    results.append([column, topics_words, sentiment])
                except Exception as e:
                    print(f"Error while processing comments for column {column} in file {file}: {e}")
                    input("Press enter to continue...")

            results_df = pd.DataFrame(results, columns=['Video ID', 'LDA Topics', 'Sentiment'])
            try:
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                results_df.to_csv(os.path.join(output_path, f"{os.path.splitext(file)[0]}_NLP_PROCESSED.csv"),
                                  index=False)
            except Exception as e:
                print(f"Error while writing results to file for {file}: {e}")
                input("Press enter to continue...")

########################################################################################################################




#### EXECUTION OF THE FUNCTIONS
########################################################################################################################
path_for_comments_scrapped = "C:/Users/Uygar TALU/Desktop/Comments_Scrapped"
output_path = "C:/Users/Uygar TALU/Desktop/Comments_Scrapped_NLP_PROCESSED"
process_comments(files_path=path_for_comments_scrapped, output_path=output_path, n_words=10)
########################################################################################################################




#FUNCTION TO COMBINE ALL THE PROCESSED NLP RESULTS CSV FILES
########################################################################################################################
#It takes a directory path as an input where your CSV files are located.
#For each CSV file in the provided directory:
#It reads the file into a DataFrame using pandas.
#It extracts a label from the filename. The label is the part of the filename starting from the first character up to
# the first underscore ('_').
#It adds a new column to the DataFrame called 'label' and assigns the extracted label to this column for all rows of
# the current DataFrame.
#It prints out a message indicating how many rows were loaded from this file and the label associated with these rows.
#It concatenates all the individual DataFrames into a single DataFrame called combined_data.
#After all CSV files have been processed, it prints out the total number of rows in the combined DataFrame.
#Finally, it returns the combined DataFrame.

def combine_csv_files(input_directory: str, output_file: str) -> pd.DataFrame:
    """
    Function to combine multiple CSV files into one dataframe, and create a new column based on the filename.

    Args:
    input_directory (str): The directory where CSV files are stored.
    output_file (str): The path and filename where the combined data will be saved as a CSV.

    Returns:
    combined_data (pd.DataFrame): A dataframe containing combined data from all CSV files.
    """

    combined_data = pd.DataFrame()  # Creates an empty dataframe

    for filename in os.listdir(input_directory):
        if filename.endswith(".csv"):  # Check for .csv files
            file_path = os.path.join(input_directory, filename)
            df = pd.read_csv(file_path)
            label = filename[0:filename.index("_")]  # Extract label from filename
            df['label'] = label  # Create new column with label
            combined_data = pd.concat([combined_data, df])  # Concatenate dataframes

            # Print out the number of rows for each file
            print(f'Added {df.shape[0]} rows from file {filename}')

    # Print out the total number of rows
    print(f'Total rows in combined data: {combined_data.shape[0]}')

    # Save the combined dataframe to a CSV file
    combined_data.to_csv(output_file, index=False)
    print(f'Combined data saved to {output_file}')

    # Return the combined data for use in Python environment
    return combined_data




#EXECUTION
########################################################################################################################
nlp_processed_comments_path = "C:/Users/Uygar TALU/Desktop/Comments_Scrapped_NLP_PROCESSED"
nlp_processed_comments_path_final_csv_path = "C:/Users/Uygar TALU/Desktop/Comments_NLP_Processed_RESULTS.csv"
comments_scrapped_NLP_processed_results_data = combine_csv_files(input_directory=nlp_processed_comments_path,
                                                                 output_file=nlp_processed_comments_path_final_csv_path)
########################################################################################################################


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################







################################## NLP ANALYSIS ON SPEECHES ANALYSIS LAYER 5 ###########################################

#TEXT PREPROCESSOR FUNCTION - ON RECOGNIZED SPEECH
########################################################################################################################
#It takes a text string as input.
#It checks if the input is a string and raises a ValueError if it's not.
#It downloads the Turkish stopwords.
#It adds additional custom words to the set of stopwords.
#It removes non-alphanumeric characters from the text using regular expressions.
#It removes punctuation marks from the text.
#It converts the text to lowercase to ensure consistent case handling.
#It tokenizes the text into individual words using the word_tokenize function.
#It removes stopwords from the list of tokens.
#It returns the preprocessed tokens as a list.


def text_preprocessor(text):
    """
    Preprocess a given text by removing non-alphanumeric characters, punctuation marks,
    converting the text to lowercase, tokenizing the text into words and removing stopwords.

    This function is specifically designed to handle Turkish texts.

    Args:
        text (str): The input text string to be preprocessed.

    Returns:
        list: A list of preprocessed tokenized words.

    Raises:
        ValueError: If the input is not a string.
        Exception: If an error occurs during text preprocessing.
    """
    if not isinstance(text, str):
        raise ValueError(f"Expected string but received {type(text)}")
    try:
        # Downloading Turkish stopwords
        stop_words = set(get_stop_words("turkish"))

        # Add your specific words to the stop words set
        custom_words = ["reis", "cumhur", "cumhurbaşkanı", "kazandı", "kazanmak", "oy", "kemal", "baykemal",
                        "sanasöz", "kemal kılıçdaroğlu", "Kemal Kılıçdaroğlu", "Recep Tayyip Erdoğan", "rte",
                        "reisicumhur", "secim", "seçim"]
        stop_words.update(custom_words)

        # Removing non-alphanumeric characters
        text_input = re.sub(r'\W+', ' ', text)

        # Removing punctuation marks
        text_input = text_input.translate(str.maketrans("", "", string.punctuation))

        # Converting the text to lowercase
        text_input = text_input.lower()

        # Tokenizing the text into words
        tokens = word_tokenize(text_input)

        # Removing stopwords
        tokens = [token for token in tokens if token not in stop_words]

        return tokens  # Return the tokens as a list instead of a string
    except Exception as e:
        print(f"Error during text preprocessing: {e}")
        input("Press enter to continue...")
########################################################################################################################




#EXECUTION FUNCTION FOR TEXT PROCESSING FOR EACH COLUMN - ON RECOGNIZED SPEECH
########################################################################################################################
#It initializes an empty list called processed_data_list to store the processed speeches.
#It iterates over each speech in the speech_data list.
#For each speech, it applies the text_preprocessor function to preprocess the comment and obtain
#a list of preprocessed tokens.
#After processing all speeches, the function returns the processed_data_list, which contains the preprocessed and
#tokenized speeches.
#In summary, the process_text_data function simplifies the process of preprocessing speech data by applying the
#text_preprocessor function to each speech and returning a list of processed speeches.


def process_text_data(speech_data):

    """
        Preprocesses the speech data by applying the text_preprocessor function to each speech.
        Returns a list of processed speeches where each speech is tokenized.

        Args:
            speech_data (list): List of speech data.

        Returns:
            list: List of processed speeches where each speech is tokenized.
        """

    processed_data_list = []
    for comment in speech_data:
        preprocessed_comment = text_preprocessor(comment)
        processed_data_list.append(preprocessed_comment)
    return processed_data_list
########################################################################################################################




#CORPUS CREATOR AND ID2WORD DICTIONARY CREATOR FUNCTION - ON RECOGNIZED SPEECH
########################################################################################################################
#It takes the processed_data_list as input, which is a combined list of processed data where each element
#is a list of tokens.
#It creates a dictionary (id2word) using the corpora.Dictionary method from the gensim library. This dictionary maps
#words to their integer ids.
#It creates the corpus by applying the doc2bow method on each list of tokens in the processed_data_list.
#This method converts each list of tokens into a bag-of-words representation.
#The corpus and id2word dictionary are returned as a tuple.
#In summary, the corpus_creator_doc2bow function takes a list of tokenized documents, creates a dictionary of words,
#and converts the documents into a document-term matrix (corpus) using the bag-of-words representation.


def corpus_creator_doc2bow(processed_data_list):

    """
        Creates a document-term matrix (corpus) and a dictionary (id2word) from a combined list of processed data.

        Args:
            processed_data_list_combined (list): Combined list of processed data where each element is a list of tokens.

        Returns:
            tuple: A tuple containing the corpus and id2word dictionary.
        """


    id2word = corpora.Dictionary(processed_data_list)
    corpus = [id2word.doc2bow(tokens) for tokens in processed_data_list]
    return corpus, id2word
########################################################################################################################



#COHERENCE VALUE OPTIMIZER FUNCTION FOR 50 TOPICS - ON RECOGNIZED SPEECH - Topic parameter 50 is the same for comments too
########################################################################################################################
#It initializes empty lists model_list and coherence_values to store the LDA models and coherence values, respectively.
#It iterates over a range of topic numbers from start to limit with a step size of step.
#For each topic number, it creates an LDA model using the LdaModel function from the gensim
#library. It sets the corpus, num_topics, and id2word parameters of the LDA model.
#It appends the created LDA model to the model_list.
#It calculates the coherence value of the LDA model using the CoherenceModel function from the gensim library.
#It sets the model, texts, dictionary, and coherence parameters of the coherence model.
#It appends the coherence value to the coherence_values list.
#It prints the coherence value for each topic number.
#Finally, it returns a tuple containing the model_list and coherence_values.
#In summary, the calculate_coherence_values function helps evaluate different LDA models by calculating their
#coherence values. It enables finding the optimal number of topics for LDA modeling.


def calculate_coherence_values(corpus, texts, dictionary, start, limit, step):

    """
            Calculate the coherence values for LDA models with different numbers of topics.

            This function iterates over a range of topic numbers (from start to limit, in steps of step),
            creates an LDA model for each number, calculates its coherence value, and stores it in a list.
            Each LDA model is also stored in a separate list. At the end, both lists are returned.

            Args:
                corpus (list of list of (int, float)): The document-term matrix (corpus) in bag-of-words format.
                texts (list of list of str): The tokenized, preprocessed texts used to create the corpus.
                dictionary (gensim.corpora.dictionary.Dictionary): The dictionary mapping words to their integer ids.
                start (int): The starting number of topics to consider.
                limit (int): The maximum number of topics to consider.
                step (int): The step size for the number of topics.

            Returns:
                tuple: A tuple of two lists: The first list contains the LDA models, the second contains the coherence values.
            """



    model_list = []
    coherence_values = []

    for num_topics in range(start, limit, step):
        print("Calculating for", num_topics, "Topics")

        lda_model = LdaModel(corpus=corpus,
                             num_topics=num_topics,
                             id2word=dictionary)

        model_list.append(lda_model)

        coherence_model = CoherenceModel(model=lda_model,
                                         texts=texts,
                                         dictionary=dictionary,
                                         coherence='c_v')

        coherence_values.append(coherence_model.get_coherence())
        print("Coherence Value:", coherence_model.get_coherence())

    return model_list, coherence_values
########################################################################################################################



#OPTIMAL LDA MODEL CREATOR FUNCTION - ON RECOGNIZED SPEECH (Alpha and Beta parameters are different)
########################################################################################################################
#It initializes empty lists model_list and coherence_values to store the LDA models and coherence values, respectively.
#It calls the calculate_coherence_values function to obtain the model_list and coherence_values lists
#based on the given inputs.
#It finds the optimal number of topics by finding the index of the maximum coherence value
#and adding it to the start parameter.
#It creates the optimal LDA model using the LdaModel function from the gensim library. It sets
#the corpus, num_topics, id2word, alpha, and beta parameters of the LDA model.
#The alpha and beta parameters are set to 5 which is different than the one we used for comments
#Finally, it returns the optimal LDA model.
#In case of any exceptions during the LDA modeling process, it prints an error message along with the
#model_list and coherence_values for debugging purposes.


def optimal_LDA_modeling(corpus, texts, dictionary, start=1, limit=50, step=1, alpha=5.0, beta=5.0):
    """
    Performs LDA (Latent Dirichlet Allocation) modeling to find the optimal number of topics.
    Returns the optimal LDA model.

    Args:
        corpus: The document-term matrix (corpus) in bag-of-words format.
        texts: The preprocessed texts used to create the corpus.
        dictionary: The dictionary mapping words to their integer ids.
        start (int): The starting number of topics to consider.
        limit (int): The maximum number of topics to consider.
        step (int): The step size for the number of topics.
        alpha (float): The alpha parameter for the LDA model. Higher values make documents more focused on few topics.
        beta (float): The beta parameter for the LDA model. Higher values make topics more focused on few words.

    Returns:
        LdaModel: The optimal LDA model.
    """
    model_list = []
    coherence_values = []

    try:
        model_list, coherence_values = calculate_coherence_values(corpus, texts, dictionary, start, limit, step)
        optimal_num_topics = coherence_values.index(max(coherence_values)) + start
        optimal_LDA_model = LdaModel(corpus=corpus, num_topics=optimal_num_topics, id2word=dictionary,
                                     alpha=alpha, eta=beta)
        return optimal_LDA_model
    except Exception as e:
        print(f"Error during LDA modeling: {str(e)}")
        print(f"Model list: {model_list}")
        print(f"Coherence values: {coherence_values}")
        input("Press enter to continue...")
########################################################################################################################



#SENTIMENT SCORE ANALYSIS FUNCTION - ON RECOGNIZED SPEECH
########################################################################################################################
#Inside a try-except block, it performs sentiment analysis using the TextBlob library.
#It calculates the sentiment polarity score for each recognized speech in the recognized speech list
#and stores them in the sentiments list.
#It returns the average sentiment score by summing up all the sentiment polarity scores and dividing
#by the total number of recognized speeches. If the sentiments list is empty, indicating no speech were provided,
#it returns a default sentiment score of 0.
#If any exception occurs during sentiment analysis, it prints an error message along with the exception details for
#debugging purposes.
#The sentiment_analysis_func function allows you to analyze the sentiment of a list of speeches and get
#an average sentiment score, which can provide insights into the overall sentiment of the speeches.



def sentiment_analysis_func(text):
    """
    Performs sentiment analysis on a list of comments.
    Returns the average sentiment score.

    Args:
        comments (list): List of comments to analyze.

    Returns:
        float: Average sentiment score.
    """

    try:
        sentiments = [TextBlob(text).sentiment.polarity for text in text]
        return sum(sentiments) / len(sentiments) if sentiments else 0
    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        input("Press enter to continue...")
########################################################################################################################




#ALL EXECUTION FUNCTON FOR NLP STEPS - ON RECOGNIZED SPEECH
########################################################################################################################
#It reads the CSV file specified by file_path using pd.read_csv, handling potential file reading errors.
#It creates the output directory specified by output_path if it doesn't exist already.
#It defines a dictionary, result_data, to store the results, with keys as column names and empty lists as values.
#For each column in the DataFrame, it performs the following steps:
#a. It retrieves the speeches from the column, drops any rows with missing values, and converts the comments to a list.
#b. It preprocesses the speeches using the process_text_data function to obtain tokenized and preprocessed speeches.
#c. It creates the document-term matrix (corpus) and dictionary (dictionary) using the corpus_creator_doc2bow function.
#d. It applies the optimal_LDA_modeling function to find the optimal number of topics and obtain the LDA model.
#e. It extracts the topics from the LDA model and stores them in the topics_words variable.
#f. It calculates the sentiment score for the speeches using the sentiment_analysis_func function.
#g. It appends the column name, topics_words, and sentiment score to the corresponding lists in the result_data dictionary.
#After processing all the columns, it creates a DataFrame, result_df, from the result_data dictionary.
#It transposes the DataFrame using the transpose() method to have the column names as rows.
#It saves the transposed DataFrame to a CSV file specified by results_file_path using to_csv.
#If any errors occur during saving to CSV, appropriate error messages are displayed.
#Finally, it returns the transposed DataFrame.
#The process_columns function allows you to process columns of a CSV file, perform topic modeling and sentiment
#analysis on the speeches, and save the results in a transposed format in a CSV file.


def process_columns(file_path, output_path):
    """
    Process the columns of a CSV file by performing text preprocessing, topic modeling (LDA), and sentiment analysis on the speeches.
    Save the results in a transposed format in a CSV file.

    Args:
        file_path (str): Path to the input CSV file.
        output_path (str): Path to the output directory where the processed results will be saved.

    Returns:
        pd.DataFrame: Transposed DataFrame containing the processed results.

    Raises:
        FileNotFoundError: If the input CSV file is not found.
        pd.errors.ParserError: If an error occurs while parsing the input CSV file.

    """
    try:
        df = pd.read_csv(file_path)
    except (FileNotFoundError, pd.errors.ParserError) as e:
        print(f"Error while reading the file: {e}")
        input("Press enter to continue...")
        return

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    results_file_path = os.path.join(output_path, "Speech_Recognition_NLP_PROCESSED.csv")

    result_data = {'Column Name': [], 'LDA Topics': [], 'Sentiment Score': []}

    for column in df.columns:
        try:
            comments = df[column].dropna().tolist()
            processed_comments = process_text_data(comments)
            corpus, dictionary = corpus_creator_doc2bow(processed_comments)
            lda_model = optimal_LDA_modeling(corpus, processed_comments, dictionary)
            topics = lda_model.show_topics(formatted=False)
            topics_words = [[word[0] for word in topic[1]] for topic in topics]
            sentiment = sentiment_analysis_func(comments)

            result_data['Column Name'].append(column)
            result_data['LDA Topics'].append(topics_words)
            result_data['Sentiment Score'].append(sentiment)

        except Exception as e:
            print(f"Error while processing comments for column {column}: {e}")
            input("Press enter to continue...")

    result_df = pd.DataFrame(result_data)
    result_df = result_df.transpose()

    try:
        result_df.to_csv(results_file_path, index=False)
    except Exception as e:
        print(f"Error while saving results to CSV: {e}")
        input("Press enter to continue...")
        return

    return result_df

########################################################################################################################



#EXECUTION OF THE NLP ANALYSIS ON RECOGNIZED SPEECHES
########################################################################################################################
file_path = "C:/Users/Uygar TALU/Desktop/Speech_Recognition_Results.csv"
output_path = "C:/Users/Uygar TALU/Desktop/"

results_df_speech_recognition_NLP_PROCESSED = process_columns(file_path, output_path)
########################################################################################################################


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################







################################## FEATURE ENGINEERING  ANALYSIS LAYER 6 ###############################################

#NEW FEATURES CREATOR FUNCTION
########################################################################################################################

#Replacing '-' with NaN: This step ensures that any occurrences of '-' in the dataframe are replaced with
#NaN (missing value).

#Creating 'Engagement_Rate_FE' feature: This feature is calculated by dividing the number of diggs (video_diggcount) by
#the number of plays (video_playcount). It represents the engagement rate of a video, indicating how many diggs a video
#receives per play.

#Creating 'Face_Detection_Rate_FE' feature: This feature is calculated by dividing the number of frames in which a face
#is detected (Number of Frames Face Detected) by the total number of frames in the video (Total Frame Of The Video).
#It represents the rate of face detection in a video, indicating how often faces appear in the frames.

#Creating 'Dominant_Emotion_Score_FE' feature: This feature is calculated by taking the maximum value among a set of
#emotion confidence level columns (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral). It represents the dominant
#emotion score in a video, indicating the most prevalent or intense emotion detected.

#Creating 'Emotional_Diversity_FE' feature: This feature is calculated by taking the standard deviation of the
#emotion confidence level columns and dividing it by the total number of frames in the video. It represents the emotional
#diversity in a video, indicating the variation or spread of emotions across the frames.

#Creating 'Sentiment_Disparity_FE' feature: This feature is calculated by subtracting the sentiment scores from
#recognized speeches (Sentiment_Scores_Recognized_Speeches) from the sentiment scores from comments
#(Sentiment_Scores_Comments). It represents the disparity or difference in sentiment between speeches and comments.

#Creating 'Engagement_Per_Second_FE' feature: This feature is calculated by dividing the overall engagement metric
#(engagement_metric_all) by the duration of the video (video_duration). It represents the engagement per second,
#indicating the level of engagement generated by the video over its duration.

#Saving the dataframe as a CSV file: The modified dataframe is saved as a CSV
#file named 'Final_Dataframe_With_Robust_Metrics.csv' on the desktop.
#Returning the modified dataframe: The function returns the modified dataframe to the caller.

all_features_daraframe = pd.read_csv("C:/Users/Uygar TALU/Desktop/ALL_RESULTS_COMBINED.csv")


def robust_metrics_feature_engineering(dataframe):
    """
    Perform feature engineering on the given dataframe to create robust metrics.

    Args:
        dataframe (pandas.DataFrame): The input dataframe containing the required columns.

    Returns:
        pandas.DataFrame: The modified dataframe with the newly created features.
    """

    # Replace '-' with NaN
    dataframe.replace("-", np.nan, inplace=True)

    print("Starting feature engineering...")

    # Engagement Rate
    print("Creating 'Engagement_Rate_FE' feature...")
    dataframe['Engagement_Rate_FE'] = dataframe['video_diggcount'] / dataframe['video_playcount']

    # Face Detection Rate
    print("Creating 'Face_Detection_Rate_FE' feature...")
    dataframe['Face_Detection_Rate_FE'] = dataframe['Number of Frames Face Detected'] / dataframe[
        'Total Frame Of The Video']

    # Dominant Emotion Score
    print("Creating 'Dominant_Emotion_Score_FE' feature...")
    emotion_cols = ['Angry (Confidence level)', 'Disgust (Confidence level)', 'Fear (Confidence level)',
                    'Happy (Confidence level)', 'Sad (Confidence level)', 'Surprise (Confidence level)',
                    'Neutral (Confidence level)']
    dataframe['Dominant_Emotion_Score_FE'] = dataframe[emotion_cols].max(axis=1)

    # Emotional Diversity
    print("Creating 'Emotional_Diversity_FE' feature...")
    dataframe['Emotional_Diversity_FE'] = dataframe[emotion_cols].std(axis=1) / dataframe['Total Frame Of The Video']

    # Comment Sentiment - Speech Sentiment
    print("Creating 'Sentiment_Disparity_FE' feature...")
    dataframe['Sentiment_Disparity_FE'] = dataframe['Sentiment_Scores_Comments'] - dataframe[
        'Sentiment_Scores_Recognized_Speeches']

    # Engagement per Second
    print("Creating 'Engagement_Per_Second_FE' feature...")
    dataframe['Engagement_Per_Second_FE'] = dataframe['engagement_metric_all'] / dataframe['video_duration']

    print("Finished feature engineering, now saving to CSV on your desktop...")

    # Save the dataframe as a csv file on your desktop
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    dataframe.to_csv(os.path.join(desktop, 'Final_Dataframe_With_Robust_Metrics.csv'), index=False)

    print("CSV file saved successfully on your desktop.")

    return dataframe

#EXECUTION
robust_metrics_feature_engineering(all_features_daraframe)
########################################################################################################################




#TOPIC LABELS SEMANTIC MEANING SIMILARITY FEATURE - PRETRAINED MODEL USED IS GOOGLENEWS
########################################################################################################################
########################################################################################################################
########################################################################################################################

#The preprocess_topic function takes a topic as input, converts it to lowercase, splits it into individual words, and
#removes stop words. It returns a preprocessed list of words representing the topic.
#The cosine_similarity_topics function calculates the cosine similarity between two topics represented as word vectors.
#It takes two topic lists as input and returns the cosine similarity score.
#The calculate_and_save_similarity function calculates the topic alignment scores between comments and speeches in the
#dataframe. It creates dictionaries to store the average similarities for each topic. It populates these dictionaries by
#iterating over the unique topic labels in the 'Topic_Labels_Comments' and 'Topic_Labels_Recognized_Speeches' columns,
#preprocessing the topics, and calculating the average similarities with other topics.
#The main function calculate_similarity calculates the topic alignment score for each row of the dataframe. It retrieves
#the comments and speeches topics, preprocesses them if they are not empty or "-", and calculates the similarity score
#using the cosine_similarity_topics function. If either topic is empty, it uses the average similarity from the
#corresponding dictionary. If both topics are empty or None, it returns NaN.
#Finally, the function adds a 'Topic_Alignment_Score' column to the dataframe and saves the modified
#dataframe as a CSV file.

#IMPORTING THE DATAFRAME WHICH HAS ROBUST FEATURES IN IT
final_df_with_robust_metrics = pd.read_csv("C:/Users/Uygar TALU/Desktop/Final_Dataframe_With_Robust_Metrics.csv")

#IMPORTING THE PRETRAINED MODEL
model = KeyedVectors.load_word2vec_format("C:/Users/Uygar TALU/Desktop/GoogleNews-vectors-negative300.bin.gz",
                                          binary=True)

#CALLING FOR STOPWORDS AS ENGLISH SINCE THE TOPC LABELS ARE IN ENGLISH
stop_words = set(stopwords.words('english'))


############################# TOPIC PREPROCESSER FUNCTION #############################

#The function first checks if the topic parameter (which is expected to be a string of text) is missing or not a
#number (NaN). If it is, the function returns an empty list.

#If the topic is not NaN, the function converts all the characters in the topic string to lowercase. This is done
#because in text analysis, words are often case-sensitive,

#After converting to lowercase, the function splits the topic string into a list of words.

#The function then filters out any "stop words" from the list of words.

#Finally, the function returns the preprocessed topic as a list of words, with each word being a separate string in the list.


def preprocess_topic(topic):
    """
    Preprocesses a topic by converting it to lowercase, splitting it into words,
    and removing stop words.

    Args:
        topic (str): The topic to be preprocessed.

    Returns:
        list: The preprocessed topic as a list of words.
    """
    if pd.isna(topic):
        return []
    topic = topic.lower().split()
    topic = [word for word in topic if word not in stop_words]
    return topic




######## COSINE SIMILARITY CALCULATOR FUNCTION IN BETWEEN TWO TOPICS(Topic of comment and topic of speech) #########

#The function takes in two topics (each a list of words) as arguments.

#For each topic, it calculates an average word vector. This is done by mapping each word in the topic to a word vector
# (using pre-trained model), and then averaging these vectors.

#If the word vectors for both topics are valid, it computes the cosine similarity between these two vectors.

#The function returns this cosine similarity value, which gives an indication of how similar the two topics are to
# each other. If the word vectors are not valid, the function returns 'NaN' (Not a Number).

def cosine_similarity_topics(topic1, topic2):
    """
    Calculates the cosine similarity between two topics represented as word vectors.

    Args:
        topic1 (list): The first topic as a list of words.
        topic2 (list): The second topic as a list of words.

    Returns:
        float: The cosine similarity between the two topics. Returns NaN if the word vectors are invalid.
    """
    topic1_vec = np.mean([model[word] for word in topic1 if word in model], axis=0)
    topic2_vec = np.mean([model[word] for word in topic2 if word in model], axis=0)

    if type(topic1_vec) == np.ndarray and type(topic2_vec) == np.ndarray:
        return cosine_similarity(topic1_vec.reshape(1, -1), topic2_vec.reshape(1, -1))[0][0]
    else:
        return np.nan


###################### EXECUTION FUNCTION FOR TOPIC ALIGNMENT SCORE FEATURE ######################

#The function first sets up two dictionaries, avg_similarities_comments and
#avg_similarities_speeches, to store the average similarity scores for each topic in the comments and
#speeches, respectively.

#After that function iterates through the unique topics found in the comments and speeches
#(ignoring any missing values or "-"). For each topic, it preprocesses the topic label
#(converts to lowercase, removes stopwords, etc.) and calculates the cosine similarity with every other topic.
#It averages these similarity scores and stores the result in the appropriate dictionary.

#After, the function calculate_similarity(row)  next. This function is used to compute
#the similarity score for a single row in the dataframe. If both comment and speech topics for the row are non-empty,
#it calculates the cosine similarity between them. If only one topic is available, it retrieves the average similarity
#for that topic from the appropriate dictionary. If both topics are empty or "-", it returns 'NaN'.

#Then consine similarity function is applied to every row in the dataframe to compute the
#'Topic_Alignment_Score'. The result is stored in a new column in the dataframe.

#Finally, the function saves the updated dataframe as a CSV file in the specified path.


def calculate_and_save_similarity(df, path, filename):
    """
    Calculates the topic alignment scores between comments and speeches in the dataframe and saves the modified dataframe.

    Args:
        df (pandas.DataFrame): The input dataframe containing columns 'Topic_Labels_Comments' and 'Topic_Labels_Recognized_Speeches'.
        path (str): The path to the directory where the CSV file will be saved.
        filename (str): The name of the CSV file.

    Returns:
        None
    """
    # Create a dictionary to store the average similarity for each topic
    avg_similarities_comments = {}
    avg_similarities_speeches = {}

    # Populate the dictionary for comments topics
    for topic_label in df['Topic_Labels_Comments'].dropna().unique():
        if topic_label != "-":
            topic = preprocess_topic(topic_label)
            similarities = [cosine_similarity_topics(topic, preprocess_topic(other_label))
                            for other_label in df['Topic_Labels_Comments'].dropna().unique() if other_label != topic_label and other_label != "-"]
            avg_similarities_comments[topic_label] = np.mean(similarities)

    # Populate the dictionary for speeches topics
    for topic_label in df['Topic_Labels_Recognized_Speeches'].dropna().unique():
        if topic_label != "-":
            topic = preprocess_topic(topic_label)
            similarities = [cosine_similarity_topics(topic, preprocess_topic(other_label))
                            for other_label in df['Topic_Labels_Recognized_Speeches'].dropna().unique() if other_label != topic_label and other_label != "-"]
            avg_similarities_speeches[topic_label] = np.mean(similarities)

    def calculate_similarity(row):
        comments_topic_label = row['Topic_Labels_Comments']
        speeches_topic_label = row['Topic_Labels_Recognized_Speeches']

        comments_topic = preprocess_topic(comments_topic_label) if comments_topic_label != "-" else None
        speeches_topic = preprocess_topic(speeches_topic_label) if speeches_topic_label != "-" else None

        if comments_topic and speeches_topic:  # both are not empty
            return cosine_similarity_topics(comments_topic, speeches_topic)
        elif comments_topic:  # comments topic is not empty
            print("Comments topic is not empty, using average similarity.")
            return avg_similarities_comments.get(comments_topic_label, np.nan)
        elif speeches_topic:  # speeches topic is not empty
            print("Speeches topic is not empty, using average similarity.")
            return avg_similarities_speeches.get(speeches_topic_label, np.nan)
        else:  # both are empty or None
            print("Both topics are empty.")
            return np.nan

    df['Topic_Alignment_Score'] = df.apply(calculate_similarity, axis=1)

    # save the dataframe
    df.to_csv(os.path.join(path, filename), index=False)



##EXECUTION OF THE TOPIC ALIGNMENT GENERATOR FUNCTION
########################################################################################################################

calculate_and_save_similarity(all_features_daraframe, "C:/Users/Uygar TALU/Desktop/", 'topic_alignment_added.csv')

general_data_correlation_and_clustering_analysis = pd.read_csv("C:/Users/Uygar TALU/Desktop/topic_alignment_added.csv")
########################################################################################################################

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################







################################## CLUSTER AND CORRELATION ANALYSIS  ANALYSIS LAYER 7 ##################################

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

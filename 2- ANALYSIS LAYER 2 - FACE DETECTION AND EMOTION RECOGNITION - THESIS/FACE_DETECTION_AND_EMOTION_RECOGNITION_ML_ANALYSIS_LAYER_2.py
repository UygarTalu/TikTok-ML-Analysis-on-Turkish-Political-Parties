########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################


############################## FACE DETECTION - EMOTION RECOGNITION - ANALYSIS LAYER 1 #################################

###INFORMATION ABOUT THE SCRIPT(ANALYSIS LAYER 1)
"""
IN THE ANALYSIS LAYER ONE WE HAVE 4 DIFFERENT COMPLEX FUNCTIONS.
1- "get_video_paths"
2- "get_all_video_paths_video_specific"
3- "extract_video_label"
4- "auto_emotion_recognition"

Results for Layer 2- At the end of execution of layer 2 we create the emotion detection features from the
processed videos. The results or the features are combined in a data frame. The resulting data frame has:

"Video Count", "Total Frame Of The Video", "Number of Frames Face Detected", "Number of Frames Face Is Not Detected",
"Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral", "Dominant Emotion", "Angry (Confidence level)",
"Disgust (Confidence level)", "Fear (Confidence level)", "Happy (Confidence level)", "Sad (Confidence level)",
"Surprise (Confidence level)", "Neutral (Confidence level)", "Label"

The details of the columns are,

"Video Count" is the sequential number of the video in the process.
"Total Frame Of The Video" is the total number of frames in the video.
"Number of Frames Face Detected" is the number of frames in which a face was detected.
"Number of Frames Face Is Not Detected" is the number of frames in which no face was detected.
"Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral" are the counts of frames in which
each respective emotion was dominant.
"Dominant Emotion" is the emotion that appeared most frequently as the dominant emotion in the video.
"Angry (Confidence level)", "Disgust (Confidence level)", "Fear (Confidence level)", "Happy (Confidence level)",
"Sad (Confidence level)", "Surprise (Confidence level)", "Neutral (Confidence level)" are the average confidence levels
for each respective emotion throughout the video where a face was detected.
"Label" is the label of the video extracted from the folder name.
"""

#### PRETAINED MODELS USED FOR ANALYSIS LAYER 2 ARE, CASCADE CLASSIFIER FOR FACE DETECTION AND DEEP FACE ALGORITHM
#### FOR EMOTION RECOGNITION
########################################################################################################################
########################################################################################################################


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

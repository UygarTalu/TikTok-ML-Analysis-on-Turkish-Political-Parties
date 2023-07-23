########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################


############################## AUDIO PROCESSING - SPEECH RECOGNITION - ANALYSIS LAYER 3 ################################

###INFORMATION ABOUT THE SCRIPT(ANALYSIS LAYER 3)
"""
IN THE ANALYSIS LAYER TWO WE HAVE 3 DIFFERENT FUNCTIONS.
1- "audio_extractor"
2- "speech_to_text_google"

Results for Layer 3- At the end of analysis layer 3 we will obtain a dataframe where number of the videos in one column
and the recognized speeches related to each video will in the other column. Labels for the videos will be specified
manually in the csv file which means manually I will modify the csv file in a way that it will be easier for NLP
analysis layer.

Video Number: Number of processed video.
Label: Hashtag or user profile label for the processed video.
Recognized Speech: Recognized speech for each video in text format.
"""

#### IN THIS ANALYSIS PRETRAIND ONLINE API BASED MODEL OF GOOGLE IS USED CALLED RECOGNIZE.GOOGLE
########################################################################################################################
########################################################################################################################

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

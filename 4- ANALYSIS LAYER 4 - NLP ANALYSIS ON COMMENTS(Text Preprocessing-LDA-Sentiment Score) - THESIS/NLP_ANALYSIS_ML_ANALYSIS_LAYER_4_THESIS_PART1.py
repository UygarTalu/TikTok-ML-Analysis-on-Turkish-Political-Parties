########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################


############### NLP - TEXT PREPROCESSING - LDA MODELING - SNETIMENT ANALYSIS - ANALYSIS LAYER 4 ON COMMENTS ############

###INFORMATION ABOUT THE SCRIPT(ANALYSIS LAYER 4)
"""
IN THE ANALYSIS LAYER FOUR WE HAVE 9 DIFFERENT FUNCTIONS.
1- "text_preprocessor"
2- "process_text_data"
3- "process_single_csv"
4- "process_all_csv_in_dir"
5- "corpus_creator_doc2bow"
6- "calculate_coherence_values"
7- "optimal_LDA_modeling"
8- "sentiment_analysis_func"
9- "process_comments"

Results for Layer 4- We will apply analysis layer 4 on both comments for the related videos and also on the recognized
speeches from the previous layer analysis with some specific difference for both parts. But most of the functions and
the operationalization is same Because we want to obtain topics for both what is being said in the videos
and what does users said for the related video. Sentiment score analysis will also be applied to recognized speeches
and the comments too. So that we can obtain the polarization score for what is being said in the video and what does
users said for the video in terms of comments. At the end of analysis layer 4 we will have a dataframe.
The dataframe will contain the results for the scrapped comments.
"""

#### MODELS OR ALGORITHMS USED IN ANALYSIS LAYER 4 - Text Preprocessing Methods - Coherence Score Calculations -
#LDA Modeling - Polarization Score
########################################################################################################################
########################################################################################################################




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
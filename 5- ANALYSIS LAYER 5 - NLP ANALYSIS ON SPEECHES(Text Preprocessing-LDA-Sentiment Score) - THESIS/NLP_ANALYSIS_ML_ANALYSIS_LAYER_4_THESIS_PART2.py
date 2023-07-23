########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################


############### NLP - TEXT PREPROCESSING - LDA MODELING - SNETIMENT ANALYSIS - ANALYSIS LAYER 5 ON SPEECHES ############

###INFORMATION ABOUT THE SCRIPT(ANALYSIS LAYER 5)
"""
IN THE ANALYSIS LAYER FOUR WE HAVE 7 DIFFERENT FUNCTIONS.
1- "text_preprocessor"
2- "process_text_data"
3- "corpus_creator_doc2bow"
4- "calculate_coherence_values"
5- "optimal_LDA_modeling"
6- "sentiment_analysis_func"
7- "process_columns"

Results for Layer 5- In the analysis layer 5 we apply the same models and follow up same procedure but this time we
specifically applying them on the one csv file that is the output of speech recognition layer. So that we do not use
two functions in the previous layer which are processing one single csv and also processing all csv files
Main reason is that we have one csv file obtained from speech recognition analysis layer. Depending on that at the
end of this layer we will obtain the LDA topics sentiment scores of the transcribed texts.

!!! IMPORTANT:  some of the functions are identical ith the previous layer so that they are used directly without changing
anything because of the in this layer normally we process the speeches but it is normal to see some arguments in the functions
 specified as "comments"
"""

#### MODELS OR ALGORITHMS USED IN ANALYSIS LAYER 5 - Text Preprocessing Methods - Coherence Score Calculations -
#LDA Modeling - Polarization Score
########################################################################################################################
########################################################################################################################


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
################## NLP ANALYSIS FOR RECOGNIZED SPEECHES FROM SCRAPPED VIDEOS  ##########################################


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
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

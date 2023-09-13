# A study in sentiment analysis
An experiment with sentiment analysis in Vader, Roberta, and transformers/pipeline 
Using the following Amazon reviews dataset: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
Followed the tutorial by Rob Mulla on Youtube.

Explanation:
(0)
Reading the data and playing around with tokenizing the text.

(1)
The first chunk of code runs Vader using the nltk library (SentimentIntensityAnalyzer) which uses a "bag of words" approach. This fails to recognize the relations between words and hence is very limited.
The code tokenizes the reviews into tokens and their tags, then calculates the negative, neutral and positive indicators for each review and appends the scores to the dataset. Feel free to uncomment line (60) to see the resulting dataset.

(2)
The second chunk uses the `transformers` library: using the `cardiffnlp/twitter-roberta-base-sentiment` model, it uses `AutoTokenizer` to tokenize the text, and then `AutoModelForSequenceClassification` to calculate the negative, neutral and positive indicators for each review then appends the score to the dataset. Uncomment line (124) to see the resulting dataset.

(3)
The last chunk uses the hugging face `transformers/pipeline` to analyze the text. https://huggingface.co/docs/transformers/main_classes/pipelines

#Use:
Just download archive.zip from the Kaggle link above, extract it inside the directory `sentiment_analysis`, and run `python3 sentiment_analyzer.py`.

#(0) Reading Data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")

import nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('maxent_ne_chunker')
#nltk.download('words')
#nltk.download('vader_lexicon')

df = pd.read_csv("Reviews.csv")
#downscaling the dataset for practical reasons
df = df.head(500)

#df["Score"].value_counts().sort_index().plot(kind="bar", title ="# reviews by stars", figsize=(10,5))
#plt.show()

#Basic NLTK

#tokenizing an example
example = df["Text"][50] 
tokens = nltk.word_tokenize(example)
#print(tokens[:10])

#getting parts of speech (tags) for the example
tagged = nltk.pos_tag(tokens)
#print(tagged[:10])

entities = nltk.chunk.ne_chunk(tagged)
#entities.pprint()

######################################################################################################
#(1) Vader: sentiment scoring per words. "Bag of words" approach
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

#sentiment analyzer object
sia = SentimentIntensityAnalyzer()

#print(sia.polarity_scores("I am so sad!"))
#print(sia.polarity_scores(example))

#now run polarity score on entire dataset
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row["Text"]
    myid = row["Id"]
    res[myid] = sia.polarity_scores(text)

vaders = pd.DataFrame(res).T
vaders = vaders.reset_index()
vaders = vaders.rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')
#print(vaders.head())

#seaborn visualisation confirming scores (compound, pos, neu, neg vs Score (in #stars))
ax = sns.barplot(data=vaders, x ="Score", y="compound")
ax.set_title("Compound Score by Amazon Stars Reviews")

fig,axs = plt.subplots(1,3,figsize=(15,5))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title("Neutral")
axs[2].set_title("Negative")
plt.tight_layout()
#plt.show()

#########################################################################################
#(2) Roberta pretrained model: picks up on relations between words
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

#print(example)
#print(sia.polarity_scores(example))

def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    #print(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2],
    }
    return scores_dict

res= {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    #some reviews have too long of a text, so we handle the exception and print the ID.
    try:
        text = row["Text"]
        myid = row["Id"]
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {}
        for k,v in vader_result.items():
            vader_result_rename[f"vader_{k}"] = v
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
    except RuntimeError:
        print(f'Broke for id {myid}')

#combining the results
results_df = pd.DataFrame(res).T
results_df = results_df.reset_index()
results_df = results_df.rename(columns={"index":"Id"})
results_df = results_df.merge(df, how="left")

#print(results_df.head())

#########################################################################################
#Now comparing the result of Vader and Roberta. Uncomment line (132) to see results
sns.pairplot(data=results_df,
             vars=["vader_neg", "vader_neu", "vader_pos", "roberta_neg", "roberta_neu", "roberta_pos"],
             hue="Score",
             palette="tab10")
#plt.show()
#########################################################################################
#Reviewing examples:
#Highest positive positivity score for rating being 1 and highest negativity score for rating being 5:
print(results_df.query("Score == 1").sort_values("roberta_pos", ascending=False)["Text"].values[0])
print(results_df.query("Score == 1").sort_values("vader_pos", ascending=False)["Text"].values[0])
print(results_df.query("Score == 5").sort_values("roberta_neg", ascending=False)["Text"].values[0])
print(results_df.query("Score == 5").sort_values("vader_neg", ascending=False)["Text"].values[0])
#########################################################################################
#(3) Using the hugging face transformers pipeline
from transformers import pipeline

sent_pipeline = pipeline("sentiment-analysis")
print(sent_pipeline("I love sentiment analysis!"))
print(sent_pipeline("BOOOOOOOO"))
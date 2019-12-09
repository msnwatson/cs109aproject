# Project Description

## Accompanying Materials
* All of the data used in our analyses can be found [here](https://github.com/msnwatson/cs109aproject/tree/master/data):
* The Jupyter notebook used to perform analysis can be found [here](https://github.com/msnwatson/cs109aproject/blob/master/trumptweetapproval.ipynb):

## Motivations

## Goal
Construct a predictive model which uses the the topics of Trump’s tweets, favorites, and retweets to predict approval/disapproval rating using poll data from fivethirtyeight as the ground truth.

# Data Cleaning

Before beginning any work to extract useful features from Trump tweet data, it was important to verify that data were consistent. The complete procedure for cleaning the data is detailed in the Jupyter notebook; however, a few notable findings are detailed here along with justifications:

*There was one entry with justy a bit.ly link and no other text or attributes. We dropped this entry because it provides us no additional information and consisted of mostly null features
*We also noticed that there were several null entries in the data. Since the missing categories included favorite_count and retweet_count, which are a huge part of our analysis, we dropped these entries from the main data frame and added them to a separate data frame. We also noticed several entries where the inputs for the features were mixed up. We also dropped these from the main data frame. Because there are only a few entries with missing data or mixed up data, we may try to find the tweets at a later date and manually re-enter the data for those tweets.
*Once we had dropped all of the NaN entries, we noticed that the favorite_count feature was the wrong data type. We converted the column’s data type from object to float. Then we split the main data frame into two separate data frames data_retweets and data_not_retweets so that we can look at trends in Trump’s retweets and Trump’s original tweets separately.


# Feature Engineering

There are two major challenges associated with our project, both associated with the use of tweet data across time.

## Extraction of Usable Features from Tweet Data

We are specifically interested in how topics of tweets from President Trump affects approval ratings. There are several ways of proceeding. Our initial thought after looking through sklearn documentation was to use the DictVectorizer tool, essentially creating a counter for each unique word in all of the tweets and using this as the feature. However, we quickly realized that this would result in at least approximate high-dimensional data introducing all of the problems associated with the curse of dimensionality and limiting the variety and utility of techniques for our future analyses.

In order to address these concerns, we decided to use word embeddings in the form of Word2Vec. [Here](https://machinelearningmastery.com/what-are-word-embeddings/) is a primer on Word2Vec and other word embedding techniques. Though, this introduced some other concerns. In particular, training our own embedding to produce high-quality embeddings would require sufficiently large amounts of text and take quite a long time to train even if we had access to GPU. Then, to sidestep this issue, we followed the advice of Cedric to use a pre-trained Word2Vec model provided by Google. Specific information on this model can be found [here](https://code.google.com/archive/p/word2vec/). For the purposes of this project, this choice is justified as unsupervised techniques are not covered in the course material, and Google has incentive to train a high-quality model. Vectors are averaged across strings, filtering out common English stop words. Sklearn's documentation describes that the built-in stop word list has some issues so a modified [external list](https://gist.github.com/sebleier/554280) which can be quickly scanned by the reader in the notebook was used. The goal of this aggregation is to attain a word embedding vector which represents the main topics of each tweet to "tag" tweets.

It is important to note that we take such great lengths to employ data science and machine learning techniques to classify tweets rather than manually constructing a list of keywords to avoid introducing our own biases into the classification, limiting the validation of any final conclusions we can make.


# Project Description

## Accompanying Materials
* All of the data used in our analyses can be found [here](http://google.com):
* The Jupyter notebook used to perform analysis can be found [here]():

## Motivations

## Goal
Construct a predictive model which uses the the topics of Trump’s tweets, favorites, and retweets to predict approval/disapproval rating using poll data from fivethirtyeight as the ground truth.

# Data Cleaning

Before beginning any work to extract useful features from Trump tweet data, it was important to verify that data were consistent. The complete procedure for cleaning the data is detailed in the Jupyter notebook; however, a few notable findings are detailed here along with justifications:

*There was one entry with justy a bit.ly link and no other text or attributes. We dropped this entry because it provides us no additional information and consisted of mostly null features

*We also noticed that there were several null entries in the data. Since the missing categories included favorite_count and retweet_count, which are a huge part of our analysis, we dropped these entries from the main data frame and added them to a separate data frame. We also noticed several entries where the inputs for the features were mixed up. We also dropped these from the main data frame. Because there are only a few entries with missing data or mixed up data, we may try to find the tweets at a later date and manually re-enter the data for those tweets.

*Once we had dropped all of the NaN entries, we noticed that the favorite_count feature was the wrong data type. We converted the column’s data type from object to float. Then we split the main data frame into two separate data frames data_retweets and data_not_retweets so that we can look at trends in Trump’s retweets and Trump’s original tweets separately.


# Feature Engineering



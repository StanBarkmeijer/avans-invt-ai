import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Importing the dataset
afinn_path = 'data/AFINN.json'
tweets_path = 'data/twitter_training.csv'

afinn = pd.read_json(afinn_path, typ='series')
tweets = pd.read_csv(tweets_path, sep=',')
tweets = tweets.dropna()

# Cleaning the texts
import string
import re

def clean_text(text):
    if isinstance(text, str):
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.encode('ascii', 'ignore').decode('ascii')
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'http\S+', '', text)
    return text


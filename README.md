# Corruption Prediction
 
 This is a repository for a data science project, titled "Prediction of Corruption Perception Index of a country
based on macro-economic features".

## Group Members:
1. Clarence Toh
2. Monica Saravana
3. Kang Yun Yi

## Project Flow

1. Data Collection: extract macroeconomic statistics from Quandl and historical CPI values from Transparency International. 
2. Feature Extraction: evaluating important indicators with high correlation to CPI and discovering new trends.
3. Machine Learning Predicition: To see if RNNs can predict CPI with historical Macroeconomic indicator data.

## Prerequisites

1. The script is tested on Python 3
2. Install and import the following libraries on your system, in order for the code to run. 

```
import pandas as pd
import numpy as np
from tensorflow import set_random_seed
from numpy.random import seed
import io
from sklearn.preprocessing import MinMaxScaler
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from google.colab import drive, files
import shutil
import time
from bs4 import BeautifulSoup
import urllib.request
from requests import get
from tika import parser
import quandl
from google.colab import drive, files
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import seaborn as sns
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from yellowbrick.model_selection import RFECV
from sklearn.feature_selection import VarianceThreshold
```
## Abstract and Motivation
Corruption in a country and its government is one of the main evils that hinder its progress. It is
common knowledge that most of the developed countries are perceived as corruption free and
the developing and under-developed countries have higher rates of corruption. Because of its
nature, corruption is a difficult measure to quantify and there a few estimates that manage to do
it. One of them is Corruption Perception Index, an estimate that quantifies corruption based on
the perceptions of people living in that country by conducting extensive surveys and statistically
interpreting the results. In our project, we would like to model these perceptions of people using
socio and macro-economic factors to predict the CPI of countries. The results that we obtain
would hence, attempt at quantifying and unifying varying perceptions of people from different
countries across different times
## Goals
We hope to be able to predict future CPI values and attain the specific indicators that induce a
poor perception of a country’s transparency level.
## Usage

#### CDS_project_time_series.ipynb
This script gets the time series for Year vs indicators for a single country: Dimensions: 22 X 1080 -
For 22 years
#### CDS_project_all_countries_2016.ipynb
This script gets the Countries vs Indicators dataset for every year : Dimensions: 180 X
1080 - For 180 countries

#### Feature Extraction.ipynb
This script explores:
1. Dimensionality reduction via Principal Component Analysis
2. Feature Selection with random forest
3. Feature Importance with Extra Trees Classifier
4. SelectKBest class to extract top 10 best features
5. Recursive feature elimination
6. Plotting a correlation graph

#### LSTM_CTY.ipynb
These scripts performs several things:
1. Interpolating data, from annual data to monthly data
2. Applying percentage changes to the data
3. Binarizing data in multiple ways
4. Applying LSTM model for prediction
5. Evaluating results by calculating percentage error

## Final Thoughts
We believe that corruption must be detected through prediction as soon as possible in order for
any government to take corrective and preventive measures. Due to the limited resources many
countries allocate for combating corruption, efforts should focus on areas most likely to be
involved in corruption cases. CPI values in general might not be the best measure for
Corruption because they are based on the perception of people and therefore, there is already a
certain bias in the data as people of a country get access to the information the government releases to them. Hence our research has provided us with a reasonable conclusion that LSTM
models are able to predict CPI values given major macro economic, social and environmental
factors and our feature selection models may allow us to round down to the most important
areas where the governments should look into. This can be considered as an early warning
system for corruption and can help narrow the governments’ focus and better implement
preventive and corrective policies.
The methods in this research have often been used to predict financial stock prices or to
forecast the weather, but there are not many studies that uses neural networks to predict public
corruption. Perhaps, the research in this paper can provide a naive model that could serve as a
proof of concept for other research aimed at fighting corruption with machine learning.

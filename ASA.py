import matplotlib
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

np.random.seed(42) 

#Exploratory Data Analysis

pos_data = pd.read_csv('data/pos.csv')   
neg_data = pd.read_csv('data/negative.csv')

# data = shuffle(data)
# data.head()

# give columns name to dataset
neg_data.columns = ['english','amharic']
pos_data.columns = ['english','amharic']
#print(neg_data)

pos = pos_data.amharic.values[:400].tolist()
neg = neg_data.amharic.values[:400].tolist()

data = pd.concat([pd.DataFrame(pos+neg),pd.DataFrame(np.ones(50).tolist()+np.zeros(50).tolist())], axis=1) 
#print(data)
data.columns = ['text','label']
#print(data.info())

#character level normalization
import re
#method to normalize character level missmatch such as ጸሀይ and ፀሐይ
def normalize_char_level_missmatch(input_token):
    rep1=re.sub('[ሃኅኃሐሓኻ]','ሀ',input_token)
    rep2=re.sub('[ሑኁዅ]','ሁ',rep1)
    rep3=re.sub('[ኂሒኺ]','ሂ',rep2)
    rep4=re.sub('[ኌሔዄ]','ሄ',rep3)
    rep5=re.sub('[ሕኅ]','ህ',rep4)
    rep6=re.sub('[ኆሖኾ]','ሆ',rep5)
    rep7=re.sub('[ሠ]','ሰ',rep6)
    rep8=re.sub('[ሡ]','ሱ',rep7)
    rep9=re.sub('[ሢ]','ሲ',rep8)
    rep10=re.sub('[ሣ]','ሳ',rep9)
    rep11=re.sub('[ሤ]','ሴ',rep10)
    rep12=re.sub('[ሥ]','ስ',rep11)
    rep13=re.sub('[ሦ]','ሶ',rep12)
    rep14=re.sub('[ዓኣዐ]','አ',rep13)
    rep15=re.sub('[ዑ]','ኡ',rep14)
    rep16=re.sub('[ዒ]','ኢ',rep15)
    rep17=re.sub('[ዔ]','ኤ',rep16)
    rep18=re.sub('[ዕ]','እ',rep17)
    rep19=re.sub('[ዖ]','ኦ',rep18)
    rep20=re.sub('[ጸ]','ፀ',rep19)
    rep21=re.sub('[ጹ]','ፁ',rep20)
    rep22=re.sub('[ጺ]','ፂ',rep21)
    rep23=re.sub('[ጻ]','ፃ',rep22)
    rep24=re.sub('[ጼ]','ፄ',rep23)
    rep25=re.sub('[ጽ]','ፅ',rep24)
    rep26=re.sub('[ጾ]','ፆ',rep25)
    #Normalizing words with Labialized Amharic characters such as በልቱዋል or  በልቱአል to  በልቷል  
    rep27=re.sub('(ሉ[ዋአ])','ሏ',rep26)
    rep28=re.sub('(ሙ[ዋአ])','ሟ',rep27)
    rep29=re.sub('(ቱ[ዋአ])','ቷ',rep28)
    rep30=re.sub('(ሩ[ዋአ])','ሯ',rep29)
    rep31=re.sub('(ሱ[ዋአ])','ሷ',rep30)
    rep32=re.sub('(ሹ[ዋአ])','ሿ',rep31)
    rep33=re.sub('(ቁ[ዋአ])','ቋ',rep32)
    rep34=re.sub('(ቡ[ዋአ])','ቧ',rep33)
    rep35=re.sub('(ቹ[ዋአ])','ቿ',rep34)
    rep36=re.sub('(ሁ[ዋአ])','ኋ',rep35)
    rep37=re.sub('(ኑ[ዋአ])','ኗ',rep36)
    rep38=re.sub('(ኙ[ዋአ])','ኟ',rep37)
    rep39=re.sub('(ኩ[ዋአ])','ኳ',rep38)
    rep40=re.sub('(ዙ[ዋአ])','ዟ',rep39)
    rep41=re.sub('(ጉ[ዋአ])','ጓ',rep40)
    rep42=re.sub('(ደ[ዋአ])','ዷ',rep41)
    rep43=re.sub('(ጡ[ዋአ])','ጧ',rep42)
    rep44=re.sub('(ጩ[ዋአ])','ጯ',rep43)
    rep45=re.sub('(ጹ[ዋአ])','ጿ',rep44)
    rep46=re.sub('(ፉ[ዋአ])','ፏ',rep45)
    rep47=re.sub('[ቊ]','ቁ',rep46) #ቁ can be written as ቊ
    rep48=re.sub('[ኵ]','ኩ',rep47) #ኩ can be also written as ኵ  
    return rep48

data['text'] = data['text'].apply(lambda x: normalize_char_level_missmatch(x))

#print(data)
text,label = data['text'].values,data['label'].values
#print(label)

#Naive Bays - CountVectorizer

#convert  a collection of text documents to a matrix of token counts.
from sklearn.feature_extraction.text import CountVectorizer 
matrix = CountVectorizer(analyzer='word') # countVectorizer analyze by word.
X = matrix.fit_transform(text).toarray() # Learn the vocabulary dictionary and return document-term matrix.
# print(X.shape)
#print(matrix.get_feature_names()) # Array mapping from feature integer indices to feature name.


unique_label = list(set(label))
Y= []
for i in label:
    Y.append(unique_label.index(i))

#print(unique_label)

from sklearn.model_selection import train_test_split # Split arrays or matrices into random train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.2)
#print(X_train.shape, X_test.shape)


# training Naive Bayes 

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predict Class
y_pred = classifier.predict(X_test)

# Accuracy 
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)

print(accuracy)


from pydantic import BaseModel

class Input(BaseModel):
    message: str

class Output(BaseModel):
    message: str

def hello_world(input: Input) -> Output:
    """Returns the `message` of the input data."""
    greeting  = "this works"
    return Output(message=greeting)


# opyrator launch-ui am_sent:hello_world

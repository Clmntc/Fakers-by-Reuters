###########################################
########################## IMPORT ###########################################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import string
# nltk.download('stopwords')
# nltk.download('punkt')
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stopwords_En = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()
# Modèle + Matrice de fonctionnalités TF-IDF
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english',max_df=0.7)
# Evaluation
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score



###########################################
########################## MODEL ###########################################
# Import csv
fake = pd.read_csv('C:/Users/Utilisateur/Desktop/Arturo/NLP_Fake_news/Hors_application/Data/Fake.csv')
true = pd.read_csv('C:/Users/Utilisateur/Desktop/Arturo/NLP_Fake_news/Hors_application/Data/True.csv')

fake['is_fake'] = 1
true['is_fake'] = 0

#Combining Title and Text
true["text"] = true["title"] + " " + true["text"]
fake["text"] = fake["title"] + " " + fake["text"]

# Subject is diffrent for real and fake thus dropping it
# Aldo dropping Date, title and Publication Info of real
true = true.drop(["subject", "date","title"], axis=1)
fake = fake.drop(["subject", "date", "title"], axis=1)

#Combining both into new dataframe and shuffle
data = true.append(fake, ignore_index=True).sample(frac=1).reset_index(drop=True)

### Create function to remove punctuation, tokenize, remove stopwords, and stem
def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = word_tokenize(text)
    text = " ".join([ps.stem(word) for word in tokens if word not in stopwords_En])
    return text

data['cleaned_text'] = data['text'].apply(lambda x: clean_text(x))
data.head()

# Preparing training and testing data using train_test_split
y = data.is_fake
X = data.cleaned_text
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state= 2022,stratify=y)

# Notre modèle
logreg_pipe = Pipeline([
 ('tvec', TfidfVectorizer()),
 ('logreg',LogisticRegression(C=1000.0, penalty='l1', solver='liblinear'))]) # tuned hpyerparameters gridsearch

logreg_pipe.fit(X_train,y_train)
y_pred_log = logreg_pipe.predict(X_test)


# Export pipeline as pickle file
with open("model_logreg.pkl", "wb") as file:
    pickle.dump(model, file)



# ###########################################
# ########################## EVALUTATION ###########################################

# print('Accuracy: ', accuracy_score(y_test, y_pred_log))
# print('Precision: ', precision_score(y_test,y_pred_log))
# print('Recall: ', recall_score(y_test, y_pred_log))
# print('f1-score: ', f1_score(y_test,y_pred_log))

# # Model evaluation
# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
# print(classification_report(y_test,y_pred_log))
# print(confusion_matrix(y_test,y_pred_log))

# ########################## EVALUTATION ###########################################
# ###########################################














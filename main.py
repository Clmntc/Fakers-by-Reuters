from flask import Flask, request, render_template
import os
import pickle
import nltk
stopwords_En = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()
# Modèle + Matrice de fonctionnalités TF-IDF
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english',max_df=0.7)

PEOPLE_FOLDER = os.path.join('static')

app = Flask(__name__)
model = pickle.load(open('model_final.pkl', 'rb'))

app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER

@app.route('/')
def index():
    return render_template('index.html') 

if __name__ == "__main__":
    app.run(debug=True)
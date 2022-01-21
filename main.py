from flask import Flask, request, render_template
import os
import pickle

PEOPLE_FOLDER = os.path.join('static')

app = Flask(__name__)
model = pickle.load(open('C:/Users/Utilisateur/Desktop/Arturo/NLP_Fake_news/model_final.pkl', 'rb'))

app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER

@app.route('/')
def index():
    return render_template('index.html') 

if __name__ == "__main__":
    app.run(debug=True)
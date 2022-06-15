from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# load the model from disk
filename = 'spamham.pkl'
clf = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('tranform.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
        my_prediction = ''.join(map(str, my_prediction))
        if(my_prediction == '0'):
            my_prediction = 'The above email is not a spam email'
        elif(my_prediction == '1'):
            my_prediction = 'The above email is a spam email'


        print(my_prediction);
    return render_template('index.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)
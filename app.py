from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    fever = 1 if request.form.get('fever') == 'yes' else 0
    cough = 1 if request.form.get('cough') == 'yes' else 0
    fatigue = 1 if request.form.get('fatigue') == 'yes' else 0
    breathing = 1 if request.form.get('breathing') == 'yes' else 0
    age = int(request.form.get('age', 25))
    gender = 1 if request.form.get('gender') == 'male' else 0
    bp = int(request.form.get('bp', 1))
    cholesterol = int(request.form.get('cholesterol', 1))

    features = np.array([[fever, cough, fatigue, breathing, age, gender, bp, cholesterol]])
    prediction = model.predict(features)[0]

    return render_template('result.html', disease=prediction)

if __name__ == '__main__':
    app.run(debug=True)
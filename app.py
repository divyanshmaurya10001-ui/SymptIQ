from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.form['symptoms']
    return render_template('result.html', symptoms=symptoms)

if __name__ == '__main__':
    app.run(debug=True)
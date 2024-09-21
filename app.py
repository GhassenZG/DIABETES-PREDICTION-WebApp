from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__, template_folder='template')

# Load the SVM model
with open('best_classifier.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve form data and convert to float
        features = [float(request.form[x]) for x in ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree_function', 'age']]
        final_features = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(final_features)

        # Interpret prediction
        if prediction[0] == 0:
            result = 'Non-Diabetic'
        else:
            result = 'Diabetic'

        return render_template('result.html', prediction_text=f'The patient is {result}')

if __name__ == '__main__':
    app.run(debug=True)

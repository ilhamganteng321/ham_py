from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Inisialisasi Flask app
app = Flask(__name__)

# Load dataset dan model
data = pd.read_csv('./dataset/diabetes_prediction_dataset.csv')

# Preprocessing data
categorical_cols = data.select_dtypes(include=['object']).columns

if 'gender' in categorical_cols:
    data['gender'] = data['gender'].map({'Female': 0, 'Male': 1})

if 'smoking_history' in categorical_cols:
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    data['smoking_history'] = label_encoder.fit_transform(data['smoking_history'])

# Pisahkan data menjadi fitur dan target
X = data.drop('diabetes', axis=1)
y = data['diabetes']

# Bagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisasi fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Inisialisasi dan latih model
model = RandomForestClassifier(random_state=100)
model.fit(X_train, y_train)

# Evaluasi model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


# Endpoint untuk prediksi dan akurasi
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data dari permintaan
        data = request.get_json()

        # Pastikan nama kolom sesuai dengan yang ada pada data training
        columns = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level',
                   'blood_glucose_level']

        # Membuat dataframe dari data input dengan nama kolom yang benar
        features = pd.DataFrame([[
            int(data['gender']),
            float(data['age']),
            int(data['hypertension']),
            int(data['heart_disease']),
            int(data['smoking_history']),
            float(data['bmi']),
            float(data['HbA1c_level']),
            float(data['blood_glucose_level'])
        ]], columns=columns)

        # Normalisasi data
        features = scaler.transform(features)

        # Prediksi
        prediction = model.predict(features)
        result = 'Risk Detected' if prediction[0] == 1 else 'No Risk'

        # Mengembalikan prediksi dan akurasi
        return jsonify({
            'prediction': result,
            'accuracy': accuracy
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# Jalankan server
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5070)

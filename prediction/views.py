import numpy as np
import pandas as pd
from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


df = pd.read_csv(r'H:/Dataset/diabetes_dataset.csv') 

le = LabelEncoder()
df['smoking_history'] = le.fit_transform(df['smoking_history'])

gender_encoder = OneHotEncoder(sparse_output=False)
gender_encoded = gender_encoder.fit_transform(df[['gender']])
gender_categories = gender_encoder.categories_[0]
gender_df = pd.DataFrame(gender_encoded, columns=[f'gender_{category}' for category in gender_categories])
df = pd.concat([df, gender_df], axis=1)

x = df.drop(['diabetes', 'gender'], axis=1)
y = df['diabetes']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

# Train model
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# Function to generate recommendations
def generate_recommendation(prediction):
    if prediction == 1:
        return "Consult a healthcare provider. Manage your diet, exercise regularly, and monitor your blood sugar."
    return "No immediate concerns, but continue to monitor your health regularly."

# Function to handle predictions and recommendations
def get_prediction_and_recommendation(data):
    gender = data.get('gender')
    age = int(data.get('age', 0))
    hypertension = int(data.get('hypertension', 0))
    heart_disease = int(data.get('heart_disease', 0))
    smoking_history = data.get('smoking_history', 'never')
    bmi = float(data.get('bmi', 0.0))
    hemoglobin_level = float(data.get('hemoglobin_level', 0.0))
    blood_glucose_level = int(data.get('blood_glucose_level', 0))

    smoking_history_encoded = le.transform([smoking_history])[0]
    input_data = np.array([[age, hypertension, heart_disease, smoking_history_encoded, bmi, hemoglobin_level, blood_glucose_level] + gender_encoder.transform([[gender]]).tolist()[0]])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Generate recommendation
    recommendation = generate_recommendation(prediction)

    return prediction, recommendation

# API view for predicting diabetes risk
@api_view(['POST'])
def predict_diabetes_api(request):
    try:
        # Get prediction and recommendation from the input data
        prediction, recommendation = get_prediction_and_recommendation(request.data)

        return JsonResponse({
            'prediction': 'Diabetes' if prediction == 1 else 'No Diabetes',
            'recommendation': recommendation
        })
    except ValueError as e:
        return JsonResponse({'error': str(e)}, status=400)

# Web View (Form Handling)
def predict_diabetes(request):
    if request.method == 'POST':
        try:
            # Get prediction and recommendation from the form data
            prediction, recommendation = get_prediction_and_recommendation(request.POST)

            return render(request, 'prediction/result.html', {
                'prediction': 'Diabetes' if prediction == 1 else 'No Diabetes',
                'recommendation': recommendation,
            })
        except ValueError as e:
            return render(request, 'prediction/form.html', {
                'error': 'Invalid input detected. Please enter valid numeric values.',
                'gender_categories': gender_categories,
                'smoking_history_options': ['never', 'No Info', 'current', 'former', 'ever', 'not current']
            })
    else:
        return render(request, 'prediction/form.html', {
            'gender_categories': gender_categories,
            'smoking_history_options': ['never', 'No Info', 'current', 'former', 'ever', 'not current']
        })

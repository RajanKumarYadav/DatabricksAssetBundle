import mlflow
import pandas as pd

# Specify the model URI (adjust this as needed)
model_uri = 'runs:/2778d98de5e64543a7e2c0ff866b42b3/model_diabetes_prediction'

# Load the model from the specified model URI
try:
    # For scikit-learn models, use mlflow.sklearn.load_model
    loaded_model = mlflow.sklearn.load_model(model_uri=model_uri)
    print(f"Model loaded successfully from {model_uri}")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    loaded_model = None

# Example data to predict on (replace with actual data)
data = [
    {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50
    },
    # Add more rows as needed
]

if loaded_model:
    try:
        # Convert the data into a DataFrame
        df = pd.DataFrame(data)

        # Ensure the DataFrame has the expected columns and data types
        expected_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        if not all(col in df.columns for col in expected_columns):
            raise ValueError('Missing columns in input data')

        # Reorder and cast columns to ensure correct data types
        df = df[expected_columns]
        df = df.astype({
            'Pregnancies': 'int64',
            'Glucose': 'int64',
            'BloodPressure': 'int64',
            'SkinThickness': 'int64',
            'Insulin': 'int64',
            'BMI': 'float64',
            'DiabetesPedigreeFunction': 'float64',
            'Age': 'int64'
        })

        # Make prediction
        predictions = loaded_model.predict(df)

        # Output predictions
        print("Predictions:", predictions)

        # Map prediction values to messages
        if predictions[0] == 0:
            result_message = "Negative"
        elif predictions[0] == 1:
            result_message = "Positive"
        else:
            result_message = "Unknown"  # Handle unexpected values if necessary

        # Format the result into the desired structure
        formatted_result = {
            "Based on your test report, the diabetes result is:": result_message
        }

        print(formatted_result)

    except Exception as e:
        print(f"Error making predictions: {str(e)}")
else:
    print("Model is not loaded. Cannot make predictions.")

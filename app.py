from flask import Flask, jsonify, render_template, redirect, url_for, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)


# Load the trained model and other necessary data
loaded_objects = joblib.load('model_and_data.joblib')

modelRFC = loaded_objects['modelRFC']
dataset = loaded_objects['dataset']
symptom_severity_dict = loaded_objects['symptom_severity_dict']
symptom_id_dict = loaded_objects['symptom_id_dict']

# Load the dataset with symptom severity
symptom_severity_dataset = pd.read_csv("Symptom-severity.csv")
# Load the dataset with symptom description
symptom_description_dataset = pd.read_csv("symptom_Description.csv")
# Load the dataset with symptom precaution
symptom_precaution_dataset = pd.read_csv("symptom_precaution.csv")

# Extract symptom options from the first column
symptom_options = list(symptom_severity_dataset.iloc[:, 0])

def make_prediction(selected_symptoms):
    # Convert selected symptoms to lowercase and replace spaces with underscores
    selected_symptoms = [symptom.lower().replace(' ', '_') for symptom in selected_symptoms]
    
    # Create a DataFrame for the input symptoms with NaN for additional symptoms
    input_data = pd.DataFrame([[None] + selected_symptoms + [np.nan] * (17 - len(selected_symptoms))], columns=dataset.columns[:-1])

    cols = input_data.columns
    data = input_data[cols].values.flatten()

    s = pd.Series(data)
    s = s.str.strip()
    s = s.values.reshape(input_data.shape)

    input_data = pd.DataFrame(s, columns=input_data.columns)

    # Create a new DataFrame with the same structure as inputdataset
    input_data_severity = pd.DataFrame(columns=input_data.columns)

    # input_data_severity keep the same columns as input_data, including diseases, symptoms, and severity
    input_data_severity['Disease'] = input_data['Disease']
    input_data_severity.iloc[:, 1:17] = input_data.iloc[:, 1:17]  


    # Encode symptoms in the data with the symptom id
    for column in input_data.columns[1:]:
        input_data[column] = input_data[column].str.strip().map(symptom_id_dict)


    # Encode symptoms in the data with the symptom severity
    for column in input_data_severity.columns[1:]:
        input_data_severity[column] = input_data_severity[column].str.strip().map(symptom_severity_dict)
        
    # NaN tukar zero
    input_data_severity = input_data_severity.fillna(0)

    # Sum up severity values for each row
    input_data_severity['totalseverity'] = input_data_severity.iloc[:, 1:].sum(axis=1)

    input_data['severity'] = input_data_severity['totalseverity']

    # NaN tukar zero
    input_data = input_data.fillna(0)

    # Convert the input data to NumPy array
    input_data_array = input_data.iloc[:, 1:].values

    # Make predictions using the trained model
    predicted_probabilities = modelRFC.predict_proba(input_data_array)[0]

    # Get the top 5 predicted diseases
    top5_indices = predicted_probabilities.argsort()[-5:][::-1]
    top5_diseases = modelRFC.classes_[top5_indices]

    # Display the top 5 predicted diseases with confidence levels and severity
    top5_predictions = []
    for i in range(len(top5_indices)):
        disease = top5_diseases[i]
        probability = predicted_probabilities[top5_indices[i]]
        severity_value = dataset[dataset['Disease'] == disease]['severity'].values[0]
        top5_predictions.append({
            'disease': disease,
            'probability': probability,
            'severity_value': severity_value
        })

    return top5_predictions

@app.route('/', methods=['GET', 'POST'])
def home():

    # If it's a GET request or initial rendering, render the index template with symptom options
    print("Rendering index.html")
    return render_template('index.html', symptom_options=symptom_options)

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        try:
            # Get the selected symptoms from the form data
            selected_symptoms = request.form.getlist('selected_symptoms')

            # Remove the word "Remove" from each symptom
            selected_symptoms = [symptom.replace('Remove', '') for symptom in selected_symptoms]

            # Print out the data retrieved from the client
            print("Selected Symptoms:", selected_symptoms)

            # Make predictions using the trained model
            top5_predictions = make_prediction(selected_symptoms)

            # If it's a POST request, render the result template
            return render_template('result.html', selected_symptoms=selected_symptoms,
                                   top5_predictions=top5_predictions)

        except Exception as e:
            # Handle exceptions and return an appropriate response
            print(f"Error processing the request: {e}")
            return "Invalid data or internal server error", 400

@app.route('/get_description')
def get_description():
    disease_name = request.args.get('disease')

    # Assuming symptom_description_dataset is the DataFrame containing descriptions
    description = symptom_description_dataset[symptom_description_dataset['Disease'] == disease_name]['Description'].values[0]

    return jsonify({'success': True, 'description': description})

@app.route('/get_precautions')
def get_precautions():
    disease_name = request.args.get('disease')

    # Assuming symptom_precaution_dataset is the DataFrame containing precautions
    precautions = symptom_precaution_dataset[symptom_precaution_dataset['Disease'] == disease_name].iloc[:, 1:].values.flatten()

    return jsonify({'success': True, 'precautions': precautions.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
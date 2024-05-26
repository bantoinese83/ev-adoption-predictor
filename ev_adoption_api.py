from flask import Flask, request, jsonify
from flasgger import Swagger
from ev_model import EVModel
from openai_api import TextGenerator
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
swagger = Swagger(app)

# Load the trained model
model_path = os.getenv('MODEL_PATH', 'model/ev_adoption_model.pkl')
model = EVModel(model_path)


# Root endpoint
@app.route('/')
def index():
    return '''
    <h1>Welcome to Electric Vehicle Adoption Predictor!</h1>
    <p>Use this API to predict the likelihood of individuals adopting electric vehicles based on various factors.</p>
    <p>Visit the <a href="/apidocs">API documentation</a> for more details.</p>
    
    <h2>Run Streamlit App</h2>
    <p>Run the Streamlit app by executing the following command:</p>
    <code>streamlit run ev_adoption_streamlit.py</code>
    
    <p>Developed by: <a href="Bryan Antoine">Bryan Antoine</a></p>
    
    '''


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict electric vehicle adoption.
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            Government_Incentives:
              type: number
              description: "Government incentives for electric vehicle adoption."
              Example: 5000
            Income:
              type: number
              description: "Income level of the population."
              Example: 60000
            Income_Population_Interact:
              type: number
              description: "Interaction between income and population density."
              Example: 120000000
            Population_Density:
              type: number
              description: "Population density of the area."
              Example: 1500
    responses:
      200:
        description: Predictions and insights on electric vehicle adoption
        schema:
          type: object
          properties:
            predictions:
              type: array
              items:
                type: number
            insights:
              type: string
      500:
        description: Error message
        schema:
          type: object
          properties:
            error:
              type: string
    """
    try:
        # Receive input data
        data = request.get_json()
        logging.info(f"Received input data: {data}")

        # Validate input data
        if not isinstance(data, dict):
            raise ValueError("Invalid input data format.")

        required_keys = ["Government_Incentives", "Income", "Income_Population_Interact", "Population_Density"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")

        # Make predictions
        predicted_labels = model.predict(data)

        # Log and return predicted labels
        logging.info(f"Predictions: {predicted_labels}")

        # Generate insights using TextGenerator
        text_generator = TextGenerator()
        insights = text_generator.generate_text(
            f"Insights on electric vehicle adoption based on {data} and prediction {predicted_labels}")
        logging.info(f"Insights: {insights}")

        return jsonify({'predictions': predicted_labels, 'insights': insights}), 200

    except Exception as exp:
        logging.error(f"Error: {exp}")
        return jsonify({'error': str(exp)}), 500


if __name__ == '__main__':
    app.run(debug=True)

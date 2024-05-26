import joblib
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define labels for predictions
prediction_labels = {
    0: 'Unlikely to adopt electric vehicles',
    1: 'Likely to adopt electric vehicles'
}


class EVModel:
    def __init__(self, model_path):
        try:
            self.model = joblib.load(model_path)
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            self.model = None

    def predict(self, data):
        if self.model is None:
            raise Exception("Model is not available.")

        # Create DataFrame from input data with correct column order
        input_df = pd.DataFrame([data], columns=['Income', 'Population_Density', 'Government_Incentives',
                                                 'Income_Population_Interact'])

        # Make predictions
        predictions = self.model.predict(input_df)
        # Map predictions to labels
        predicted_labels = [prediction_labels[prediction] for prediction in predictions]

        return predicted_labels

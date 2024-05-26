import os

import streamlit as st
from ev_model import EVModel
from openai_api import TextGenerator

# Load the trained model
model_path = "model/ev_adoption_model.pkl"
model = EVModel(model_path)

# Initialize the TextGenerator
text_generator = TextGenerator()


def main():
    st.title("🚗🔌 Electric Vehicle Adoption Predictor 🌱🌍")

    st.markdown("""
    Welcome to the Electric Vehicle Adoption Predictor! 🎉
    Use this app to predict the likelihood of individuals adopting electric vehicles based on various factors. 📊
    """)

    st.sidebar.header("⚙️ E.V.A.P. Settings")

    # Input for OpenAI API Key
    st.sidebar.header('🔑 OpenAI API Key')
    api_key = st.sidebar.text_input("Enter your OpenAI API Key", value=os.getenv("API_KEY", ""))

    # Initialize the TextGenerator with the provided API key
    openai_text_generator = TextGenerator()
    openai_text_generator.client.api_key = api_key

    # Input form with sliders
    st.sidebar.header('🎛️ Personalize Your Prediction')

    st.sidebar.markdown("### 💰 Government Incentives")
    st.sidebar.markdown("The government incentives for electric vehicle adoption.")
    government_incentives = st.sidebar.slider("Government Incentives", min_value=0, max_value=10000, value=5000)

    st.sidebar.markdown("### 💼 Income")
    st.sidebar.markdown("The income level of the population.")
    income = st.sidebar.slider("Income", min_value=10000, max_value=150000, value=60000)

    st.sidebar.markdown("### 🏙️ Income Population Interact")
    st.sidebar.markdown("This represents the interaction between income and population density.")
    income_population_interact = st.sidebar.slider("Income Population Interact", min_value=1000000, max_value=200000000,
                                                   value=120000000)

    st.sidebar.markdown("### 🌍 Population Density")
    st.sidebar.markdown("The population density of the area.")
    population_density = st.sidebar.slider("Population Density", min_value=100, max_value=5000, value=1500)

    input_data = {
        'Government_Incentives': government_incentives,
        'Income': income,
        'Income_Population_Interact': income_population_interact,
        'Population_Density': population_density
    }

    # Predict button with progress bar
    if st.sidebar.button('🔮 Predict'):
        with st.spinner('Predicting...'):
            try:
                predicted_labels = model.predict(input_data)

                # Display emoji based on prediction
                if predicted_labels[0] == 'Likely to adopt electric vehicles':
                    st.success("🎉👍 Likely to adopt electric vehicles")
                else:
                    st.success("😢👎 Unlikely to adopt electric vehicles")

                # Generate insights using TextGenerator
                insights = text_generator.generate_text(
                    f"Insights on electric vehicle adoption based on {input_data} and prediction {predicted_labels}")
                st.markdown(f"**Insights:** {insights}")

            except Exception as e:
                st.error(f"Error: {e}")


if __name__ == '__main__':
    main()

import streamlit as st
import pandas as pd
import warnings
from deep_translator import GoogleTranslator
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# Suppress warnings
warnings.filterwarnings("ignore")

# Load and preprocess data
data = pd.read_csv("fetal_health.csv")
data = data.dropna(axis=0, how='any')
data2 = pd.read_csv("Maternal Health Risk Data Set.csv")
data2 = data2.dropna(axis=0, how='any')

# PCA for fetal health to select features
pcaf = PCA(n_components=3)  # Adjusted to match the number of features expected by the model
x2 = data.drop('fetal_health', axis=1)
pcaf.fit(x2)
most_influential_columns_for_fetal_health = x2.columns[pcaf.components_.argmax(axis=1)]

# PCA for maternal health to select features
pcam = PCA(n_components=4)  # Adjusted to match the number of features expected by the model
x = data2.drop('RiskLevel', axis=1)
pcam.fit(x)
most_influential_columns_for_maternal_health = x.columns[pcam.components_.argmax(axis=1)]

# Filter the datasets to use only the selected features
x2_filtered = x2[most_influential_columns_for_fetal_health]
x_filtered = x[most_influential_columns_for_maternal_health]

# Train-test split
y2 = data['fetal_health']
x2_train, x2_test, y2_train, y2_test = train_test_split(x2_filtered, y2, random_state=0, test_size=0.2)
y = data2['RiskLevel']
x_train, x_test, y_train, y_test = train_test_split(x_filtered, y, random_state=0, test_size=0.2)

# Train models
fetal_model = GradientBoostingClassifier()
fetal_model.fit(x2_train, y2_train)
maternal_model = GradientBoostingClassifier()
maternal_model.fit(x_train, y_train)

# Define the translation function
def translate_text(text, target_language):
    try:
        translated_text = GoogleTranslator(source='en', target=target_language).translate(text)
        return translated_text
    except Exception as e:
        st.warning(f"Translation error: {e}. Displaying in English.")
        return text

# List of supported languages
LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese": "zh",
    "Hindi": "hi",
    # Add other languages as needed
}

# Sidebar for language selection
st.sidebar.title("Language Selection")
language_name = st.sidebar.selectbox("Choose your language", list(LANGUAGES.keys()))
language_code = LANGUAGES[language_name]

# Sidebar menu
with st.sidebar:
    selected = option_menu('Health Check', 
                           ['About us', 'Pregnancy Risk Prediction', 'Fetal Health Prediction', 'Suggestions', 'More..'],
                           icons=['chat-square-text', 'hospital', 'capsule-pill', 'clipboard-data'], 
                           default_index=0)

# Display content based on selected menu item
if selected == 'Pregnancy Risk Prediction':
    st.title(translate_text("Pregnancy Risk Prediction", language_code))
    input_features = {}

    for i, col in enumerate(most_influential_columns_for_maternal_health):
        input_features[col] = st.text_input(f'Enter {col}', key=f"maternal_{col}_{i}")

    if st.button('Predict Pregnancy Risk'):
        try:
            # Ensure inputs are not empty
            if all(input_features.values()):
                predicted_risk = maternal_model.predict([list(map(float, input_features.values()))])[0]
                risk_levels = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
                risk_level = risk_levels.get(predicted_risk, "Unknown Risk")
                st.write(f"Risk Level: {translate_text(risk_level, language_code)}")
            else:
                st.warning(translate_text("Please fill in all the required fields.", language_code))
        except Exception as e:
            st.warning(f"Error in prediction: {e}")

elif selected == 'Fetal Health Prediction':
    st.title(translate_text("Fetal Health Prediction", language_code))
    fetal_input_features = {}

    for i, col in enumerate(most_influential_columns_for_fetal_health):
        fetal_input_features[col] = st.text_input(f'Enter {col}', key=f"fetal_{col}_{i}")

    if st.button('Predict Fetal Health'):
        try:
            # Ensure inputs are not empty
            if all(fetal_input_features.values()):
                predicted_health = fetal_model.predict([list(map(float, fetal_input_features.values()))])[0]
                health_statuses = {1: "Normal Prediction", 2: "Suspect Prediction", 3: "Pathological Prediction"}
                health_status = health_statuses.get(predicted_health, "Unknown Status")
                st.write(f"Fetal Health Status: {translate_text(health_status, language_code)}")
            else:
                st.warning(translate_text("Please fill in all the required fields.", language_code))
        except Exception as e:
            st.warning(f"Error in prediction: {e}")

# Add 'About us', 'Suggestions', and 'More..' sections here
# ...

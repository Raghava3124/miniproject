import streamlit as st
import pandas as pd
import warnings
from deep_translator import GoogleTranslator
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import openai
import pyttsx3
from sklearn.decomposition import PCA



# Initialize pyttsx3 for text-to-speech
engine = pyttsx3.init()

def configure_voice_engine():
    """Configure pyttsx3 voice engine."""
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)  # Female voice
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 1.0)  # Max volume

configure_voice_engine()

def speak_text(text):
    """Speak the provided text using pyttsx3."""
    engine.say(text)
    engine.runAndWait()






# Suppress warnings
warnings.filterwarnings("ignore")

# Load and preprocess data
data = pd.read_csv("fetal_health.csv")
data = data.dropna(axis=0, how='any')
data2 = pd.read_csv("Maternal Health Risk Data Set.csv")
data2 = data2.dropna(axis=0, how='any')

# Split data for maternal health model
y = data2['RiskLevel']
x = data2.drop('RiskLevel', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)

# Split data for fetal health model
y2 = data['fetal_health']
x2 = data.drop('fetal_health', axis=1)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, random_state=0, test_size=0.2)


maternal_model = GradientBoostingClassifier()
maternal_model.fit(x_train, y_train)

# Train fetal health prediction model
fetal_model = GradientBoostingClassifier()
fetal_model.fit(x2_train, y2_train)

maternal_predictions = maternal_model.predict(x_test)
maternal_accuracy = accuracy_score(y_test, maternal_predictions)
print(f"Maternal Health Model Accuracy: {maternal_accuracy:.2f}")

# Calculate and print accuracy for fetal health model
fetal_predictions = fetal_model.predict(x2_test)
fetal_accuracy = accuracy_score(y2_test, fetal_predictions)
print(f"Fetal Health Model Accuracy: {fetal_accuracy:.2f}")








pcaf=PCA(n_components=10)
pcaf.fit(x2)

# Get the principal components
components = pcaf.components_

# Get the column names
column_names = x2.columns

# Create a DataFrame to see which columns contribute to each principal component
pca_df = pd.DataFrame(components, columns=column_names)

components = pcaf.components_

# Get the column names
column_names = x2.columns

# Find the column with the highest absolute value in each component
most_influential_columns_fetal = []
for component in components:
    index = component.argmax()  # Get the index of the highest absolute value
    most_influential_columns_fetal.append(column_names[index])








components = pcaf.components_
original_columns = x2.columns

# Extract top 10 contributing features for each principal component
top_features = {}
for i, component in enumerate(components):
    # Sort features by absolute contribution
    sorted_indices = component.argsort()[-10:][::-1]
    top_features[f'PC{i + 1}'] = original_columns[sorted_indices].tolist()

# Print top features for each principal component
for pc, features in top_features.items():
    print(f"Top 10 features for {pc}: {features}")
    most_influential_columns_fetal=features
    break

print("\n\n\n")
print(most_influential_columns_fetal)
print("\n\n\n")



pcam=PCA(n_components=4)

pcam.fit(x)

# Get the principal components
components = pcam.components_

# Get the column names
column_names = x.columns

# Create a DataFrame to see which columns contribute to each principal component
pca_df2 = pd.DataFrame(components, columns=column_names)

print(pca_df2)

components = pcam.components_

# Get the column names
column_names = x.columns

# Find the column with the highest absolute value in each component
most_influential_columns_maternal = []
for component in components:
    index = component.argmax()  # Get the index of the highest absolute value
    most_influential_columns_maternal.append(column_names[index])

print(most_influential_columns_maternal)


dffnew=x2[most_influential_columns_fetal]

print(dffnew.head())
print(y2.head())






components = pcam.components_
original_columns = x.columns

# Extract top 10 contributing features for each principal component
top_features = {}
for i, component in enumerate(components):
    # Sort features by absolute contribution
    sorted_indices = component.argsort()[-4:][::-1]
    top_features[f'PC{i + 1}'] = original_columns[sorted_indices].tolist()

# Print top features for each principal component
for pc, features in top_features.items():
    print(f"Top 10 features for {pc}: {features}")
    most_influential_columns_maternal=features
    break


dfmnew=x[most_influential_columns_maternal]
print(dfmnew.head())
print("for testing --",most_influential_columns_maternal)
print(x.head())
print(y.head())



xn_train, xn_test, yn_train, yn_test = train_test_split(dfmnew, y, random_state=0, test_size=0.2)


xn2_train, xn2_test, yn2_train, yn2_test = train_test_split(dffnew, y2, random_state=0, test_size=0.2)


# Train maternal health prediction model with Gradient Boosting Classifier
maternal_model_new = GradientBoostingClassifier()
maternal_model_new.fit(xn_train, yn_train)

# Train fetal health prediction model with Gradient Boosting Classifier
fetal_model_new = GradientBoostingClassifier()
fetal_model_new.fit(xn2_train, yn2_train)


maternal_accuracy_new= maternal_model_new.score(xn_test, yn_test)
print(f"3) Maternal Health Prediction Model Accuracy: {maternal_accuracy_new * 100:.2f}%")


fetal_accuracy_new = fetal_model_new.score(xn2_test, yn2_test)
print(f"4) Fetal Health Prediction Model Accuracy: {fetal_accuracy_new * 100:.2f}%")



















# Define the translation function
def translate_text(text, target_language):
    try:
        translated_text = GoogleTranslator(source='en', target=target_language).translate(text)
        return translated_text
    except Exception as e:
        st.warning(f"Translation error: {e}. Displaying in English.")
        return text

# List of supported languages including Indian languages
LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese": "zh",
    "Hindi": "hi",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Gujarati": "gu",
    "Marathi": "mr",
    "Punjabi": "pa",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Odia": "or",
    "Urdu": "ur",
    "Nepali": "ne"
}


# Sidebar for language selection
st.sidebar.title("Language Selection")
language_name = st.sidebar.selectbox("Choose your language", list(LANGUAGES.keys()))
language_code = LANGUAGES[language_name]

# Sidebar menu
with st.sidebar:
    selected = option_menu('Health Check', 
                           ['About us', 'Pregnancy Risk Prediction', 'Fetal Health Prediction', 'Suggestions', 'More..'],
                           icons=['chat-square-text', 'hospital', 'capsule-pill', 'clipboard-data', 'gear'], 
                           default_index=0)

title = translate_text("About Us", language_code)
about_text = translate_text(
    "At Health Checkers, we aim to revolutionize healthcare by providing predictive analysis for maternal and fetal health.", 
    language_code
)
pregnancy_risk_title = translate_text("Pregnancy Risk Prediction", language_code)
pregnancy_risk_text = translate_text(
    "This feature uses advanced algorithms to assess risks during pregnancy, analyzing parameters like age, blood pressure, and more.", 
    language_code
)
fetal_health_title = translate_text("Fetal Health Prediction", language_code)
fetal_health_text = translate_text(
    "This system assesses fetal health status by analyzing various factors, providing insights into the baby's well-being.", 
    language_code
)

suggestions = {
    "Maternal Health": {
        "Low Risk": [
            "Keep up with scheduled prenatal appointments.",
            "Maintain a balanced diet rich in essential nutrients.",
            "Stay physically active with exercises suitable for pregnant women."
        ],
        "Medium Risk": [
            "Consult your healthcare provider if you notice any unusual symptoms.",
            "Follow your healthcare provider's recommendations closely.",
            "Monitor for any signs of preterm labor, such as regular contractions."
        ],
        "High Risk": [
            "Seek immediate medical attention in case of emergencies.",
            "Follow your healthcare provider's advice rigorously.",
            "Prepare for closer monitoring and possible interventions if indicated."
        ]
    },
    "Fetal Health": {
        "Normal Prediction": [
            "Ensure regular prenatal care to monitor fetal growth.",
            "Maintain a healthy lifestyle, avoiding harmful substances.",
            "Take prenatal vitamins as prescribed."
        ],
        "Suspect Prediction": [
            "Contact your healthcare provider if you have concerns about fetal movement.",
            "Follow recommendations for fetal monitoring, such as kick counts.",
            "Monitor for any signs of preterm labor or fetal distress."
        ],
        "Pathological Prediction": [
            "Discuss options with your healthcare provider if complications are detected.",
            "Follow advice on specialized testing if recommended.",
            "Prepare for potential interventions to optimize fetal health outcomes."
        ]
    }
}



# Display content based on selected menu item
if selected == 'Pregnancy Risk Prediction':
    st.title(translate_text("Pregnancy Risk Prediction", language_code))

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input('Age of the Person', key="age")
    with col2:
        diastolicBP = st.text_input('Diastolic BP in mmHg')
    with col3:
        BS = st.text_input('Blood glucose in mmol/L')
    with col1:
        heartRate = st.text_input('Heart rate in bpm')

    if st.button('Predict Pregnancy Risk'):
        try:
            # Predicting risk level
            predicted_risk = maternal_model_new.predict([[diastolicBP,age,BS,heartRate]])[0]
            if predicted_risk == 0:
                risk_level = "Low Risk"
                st.markdown('<p style="font-weight: bold; font-size: 20px; color: green;">Low Risk</p>', unsafe_allow_html=True)
            elif predicted_risk == 1:
                risk_level = "Medium Risk"
                st.markdown('<p style="font-weight: bold; font-size: 20px; color: orange;">Medium Risk</p>', unsafe_allow_html=True)
            elif predicted_risk == 2:
                risk_level = "High Risk"
                st.markdown('<p style="font-weight: bold; font-size: 20px; color: red;">High Risk</p>', unsafe_allow_html=True)
            
            # Display translated suggestions
            st.subheader(translate_text("Suggestions:", language_code))
            for suggestion in suggestions["Maternal Health"][risk_level]:
                st.write(f"- {translate_text(suggestion, language_code)}")
                
        except Exception as e:
            st.warning(f"Error in prediction: {e}")


elif selected== 'About us':
    st.title(title)
    st.write(about_text)

    col1, col2 = st.columns(2)
    with col1:
        st.header(pregnancy_risk_title)
        st.write(pregnancy_risk_text)
        st.image("graphics/pregnancy_risk_image.jpg", caption="Pregnancy Risk Prediction", use_column_width=True)

    with col2:
        st.header(fetal_health_title)
        st.write(fetal_health_text)
        st.image("graphics/fetal_health_image.jpg", caption="Fetal Health Prediction", use_column_width=True)



elif selected == 'Fetal Health Prediction':
    st.title(translate_text("Fetal Health Prediction", language_code))

    col1, col2, col3 = st.columns(3)
    fetal_features = []
    fields = list(most_influential_columns_fetal)
    for i, field in enumerate(fields):
        col = [col1, col2, col3][i % 3]
        with col:
            fetal_features.append(st.text_input(field))

    if st.button('Predict Fetal Health'):
        try:
            predicted_health = fetal_model_new.predict([fetal_features])[0]
            if predicted_health == 1:
                health_status = "Normal Prediction"
                st.markdown('<p style="font-weight: bold; font-size: 20px; color: green;">Normal</p>', unsafe_allow_html=True)
            elif predicted_health == 2:
                health_status = "Suspect Prediction"
                st.markdown('<p style="font-weight: bold; font-size: 20px; color: orange;">Suspect</p>', unsafe_allow_html=True)
            elif predicted_health == 3:
                health_status = "Pathological Prediction"
                st.markdown('<p style="font-weight: bold; font-size: 20px; color: red;">Pathological</p>', unsafe_allow_html=True)

            # Display translated suggestions
            st.subheader(translate_text("Suggestions:", language_code))
            for suggestion in suggestions["Fetal Health"][health_status]:
                st.write(f"- {translate_text(suggestion, language_code)}")
                
        except Exception as e:
            st.warning(f"Error in prediction: {e}")



elif selected == 'Suggestions':
    st.header(translate_text("Suggestions for Maternal Health", language_code))
    for risk_level, tips in suggestions["Maternal Health"].items():
        st.subheader(translate_text(risk_level, language_code))
        for tip in tips:
            st.markdown(f"- {translate_text(tip, language_code)}")

    st.header(translate_text("Suggestions for Fetal Health", language_code))
    for prediction, tips in suggestions["Fetal Health"].items():
        st.subheader(translate_text(prediction, language_code))
        for tip in tips:
            st.markdown(f"- {translate_text(tip, language_code)}")

elif (selected == 'More..'):

    st.title(translate_text("For more information", language_code))
    st.write(translate_text("Have a look at these", language_code))
    st.markdown("""<iframe width="560" height="315" src="https://www.youtube.com/embed/0BrxCY89_uQ?si=QQR-EGp_gtmsE2nz" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>""", unsafe_allow_html=True)
    st.markdown("""<iframe width="560" height="315" src="https://www.youtube.com/embed/Km0CsOjF_Fw?si=mYY6bhzH0UfaW9vt" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe></iframe>""", unsafe_allow_html=True)
    st.markdown("""<iframe width="560" height="315" src="https://www.youtube.com/embed/_dVuHFdUN0c?si=2sJsA-ooIvaU7loK" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
""", unsafe_allow_html=True)
    st.markdown("""<iframe width="560" height="315" src="https://www.youtube.com/embed/aCx0Pgb8X0Q?si=TYEJRLU32U5ZeVcy" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
""", unsafe_allow_html=True)


    st.markdown("""<iframe width="560" height="315" src="https://www.youtube.com/embed/8BH7WFmRs-E" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>""",unsafe_allow_html=True)


    st.markdown("""<iframe width="560" height="315" src="https://www.youtube.com/embed/H1owis533Xw" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
""", unsafe_allow_html=True)



    st.markdown("""<iframe width="560" height="315" src="https://www.youtube.com/embed/DDb6mMIHtas" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>""", unsafe_allow_html=True)


    st.markdown("""<iframe width="560" height="315" src="https://www.youtube.com/embed/Q0u_-LKkWNk" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
""", unsafe_allow_html=True)

    st.markdown("""<iframe width="560" height="315" src="https://www.youtube.com/embed/y4HK5CTVkXM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
""", unsafe_allow_html=True)
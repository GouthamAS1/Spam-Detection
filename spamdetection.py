import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score
import streamlit as st

# Load the data (ensure the file path is correct for your environment)
data = pd.read_csv(r'''S:\Desktop\mini project\Final\spam.csv''')
#print(data.head())

# Preprocess the data
data.drop_duplicates(inplace=True)
data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])
#print(data.head())

# Split the dataset into messages and categories
mess = data['Message']
cat = data['Category']

# Train-test split
(mess_train, mess_test, cat_train, cat_test) = train_test_split(mess, cat, test_size=0.2)

# Initialize the CountVectorizer
cv = CountVectorizer(stop_words='english')
features = cv.fit_transform(mess_train)

# Create and train the model
model = MultinomialNB()
model.fit(features, cat_train)

# Test the model accuracy and precision
features_test = cv.transform(mess_test)
predictions = model.predict(features_test)

# Calculate accuracy and precision
accuracy = accuracy_score(cat_test, predictions)
precision = precision_score(cat_test, predictions, pos_label='Spam', average='binary')

#print(f'Model accuracy: {accuracy}')  # Optional: show model accuracy
#print(f'Model precision: {precision}')  # Optional: show model precision

# Prediction function

def predict(message):
    input_message = cv.transform([message]).toarray()
    result = model.predict(input_message)
    return result[0]  # Return the first result (single prediction)

# Streamlit UI setup

st.header('Spam Detection')

input_mess = st.text_input('Enter Text Here')

if st.button('Validate'):
    output = predict(input_mess)

    if output == 'Spam':
        st.markdown("**Result:** The message is **Spam**")
    else:
        st.markdown("**Result:** The message is **Not Spam**")

    # Display accuracy and precision metrics
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
    st.write(f"Model Precision: {precision * 100:.2f}%")

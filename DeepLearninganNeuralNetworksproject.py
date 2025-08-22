import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# -----------------------------------
# STEP 1: Data Loading and Exploration
# -----------------------------------
st.write("### Step 1: Data Loading and Exploration")
# Load the dataset
df = pd.read_csv('IMDB Dataset.csv')

# Display the first few rows
st.write("First 5 rows of the dataset:")
st.dataframe(df.head())

# Check the distribution of sentiments
st.write("Sentiment distribution:")
st.write(df['sentiment'].value_counts())

# -----------------------------------
# STEP 2: Data Preprocessing
# -----------------------------------
st.write("### Step 2: Data Preprocessing")
# Convert 'sentiment' column to numerical (0 for negative, 1 for positive)
df['sentiment'] = df['sentiment'].map({'negative': 0, 'positive': 1})

# Define a preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags
    clean = re.compile('<.*?>')
    text = re.sub(clean, ' ', text)
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize and remove stop words
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# Apply preprocessing to the reviews
df['review'] = df['review'].apply(preprocess_text)

# Split the data into training and test sets
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.2, random_state=42
)

# Use TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000) # Use top 5000 most frequent words
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_raw)
X_test_tfidf = tfidf_vectorizer.transform(X_test_raw)

# Convert TF-IDF sparse matrix to a dense numpy array for Keras
X_train_dense = X_train_tfidf.toarray()
X_test_dense = X_test_tfidf.toarray()

# -----------------------------------
# STEP 3: Model Building
# -----------------------------------
st.write("### Step 3: Model Building")
# Get the number of features from TF-IDF vectorizer
input_dim = X_train_dense.shape[1]

# Construct the Sequential model
model = Sequential()
# First layer: Dense layer with ReLU activation
model.add(Dense(128, activation='relu', input_dim=input_dim))
# Hidden layer: Dense layer with ReLU activation
model.add(Dense(64, activation='relu'))
# Output layer: Dense layer with Sigmoid activation for binary classification
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display model summary
model.summary()

# -----------------------------------
# STEP 4: Model Training
# -----------------------------------
st.write("### Step 4: Model Training")
# Train the model with a validation split
history = model.fit(
    X_train_dense, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# -----------------------------------
# STEP 5: Evaluation
# -----------------------------------
st.write("### Step 5: Evaluation")
# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test_dense, y_test, verbose=0)
st.write(f"Test Accuracy: {accuracy:.4f}")
st.write(f"Test Loss: {loss:.4f}")

# -----------------------------------
# STEP 6: Visualization
# -----------------------------------
st.write("### Step 6: Visualization")
# Plot training and validation loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# -----------------------------------
# STEP 7: Report
# -----------------------------------
st.write("### Step 7: Report & Insights")
st.write("This section discusses the findings from the project.")
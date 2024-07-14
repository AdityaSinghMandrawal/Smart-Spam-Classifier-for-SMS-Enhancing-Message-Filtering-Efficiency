# Importing the necessary libraries
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
import nltk

# Download NLTK resources (if not already downloaded)
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
messages = pd.read_csv(r'C:\Users\Acer\Documents\SMS spam classifier\SMSSpamCollection', sep='\t',
                       names=["label", "message"])

# Data cleaning and preprocessing
w = WordNetLemmatizer()
corpus = []
for i in range(0, len(messages)):
    review = messages['message'][i]
    review = review.lower()
    review = review.split()
    review = [w.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
cv = CountVectorizer(max_features=1000)
X = cv.fit_transform(corpus).toarray()

# Encoding the labels
y = messages['label'].map({'ham': 0, 'spam': 1}).values

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Training model using Naive Bayes classifier
spam_detect_model = MultinomialNB().fit(X_train, y_train)

# Predicting on test set
y_pred = spam_detect_model.predict(X_test)

# Evaluating the model
confusion_m = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Displaying results
print("Confusion Matrix:")
print(confusion_m)
print("\nAccuracy:", accuracy)

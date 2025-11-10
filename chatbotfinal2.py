import nltk
import random
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances

# Download NLTK data (if not already installed)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load intent file
with open("F:\\vscode\\samples\\New folder\\newintents.json") as file:
    intents = json.load(file)

# Create a list of stopwords and a lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Create a list of documents and their labels
documents = []
labels = []
responses = {}

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize, clean the text, and use lemmatization
        tokens = [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(pattern) if word.lower() not in stop_words]
        documents.append(" ".join(tokens))
        labels.append(intent['tag'])
    responses[intent['tag']] = intent['responses']

# Create a TF-IDF model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# Initialize an empty dictionary to store user input for training
training_data = {}

# Get user input and find the intent
while True:
    user_input = input("You: ")
    if (user_input.lower() == 'exit') or (user_input.lower() == 'bye') or user_input.lower() == 'good bye' or (user_input.lower() == 'good to see you'):
        print("Chatbot: Goodbye!")
        break

    if user_input.lower() == 'train':
        # Allow the user to provide training data
        user_input = input("Please enter your training question: ")
        training_question = user_input
        user_input = input("Please enter the intent for the training question: ")
        training_intent = user_input
        if training_intent in intents['intents']:
            user_input = input("Please enter the response for the training question: ")
            training_response = user_input
            if training_intent not in training_data:
                training_data[training_intent] = []
            training_data[training_intent].append({'question': training_question, 'response': training_response})
            print("Training data added.")
        else:
            print("Invalid intent. Training data not added.")
    else:
        user_tokens = [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(user_input) if word.lower() not in stop_words]
        user_vector = vectorizer.transform([" ".join(user_tokens)])

        # Calculate cosine similarity using Jaccard similarity
        cosine_similarities = 1 - pairwise_distances(user_vector, X, metric='cosine')
        intent_index = cosine_similarities.argmax()
        intent_label = labels[intent_index]

        # Check if there's training data for the intent
        if intent_label in training_data:
            for example in training_data[intent_label]:
                if example['question'] == user_input:
                    response = example['response']
                    print("Chatbot:", response)
                    break
            else:
                response = random.choice(responses[intent_label])
                print("Chatbot:", response)
        else:
            response = random.choice(responses[intent_label])
            print("Chatbot:", response)

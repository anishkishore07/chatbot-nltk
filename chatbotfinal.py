import keras
import nltk
import pickle
import json
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import random
import requests
from pycricbuzz import Cricbuzz
import time


from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

words = []
classes = []
documents = []
ignore = ['?', '!', ',', "'s"]

data_file = open('newintents.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))


training_data = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern = doc[0]
    pattern = [lemmatizer.lemmatize(word.lower()) for word in pattern]

    for word in words:
        if word in pattern:
            bag.append(1)
        else:
            bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training_data.append((bag, output_row))

random.shuffle(training_data)
X_train = np.array([item[0] for item in training_data])
y_train = np.array([item[1] for item in training_data])

# Model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(len(X_train[0]),)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='softmax'))


# Training the model
adam = keras.optimizers.Adam(0.001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
weights = model.fit(X_train, y_train, epochs=200, batch_size=10, verbose=1)
model.save('mymodel.keras', weights)


from keras.models import load_model

model = load_model('mymodel.keras')
intents = json.loads(open('newintents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


# Predict
def clean_up(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def create_bow(sentence, words):
    sentence_words = clean_up(sentence)
    bag = list(np.zeros(len(words)))

    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence, model):
        p = create_bow(sentence, words)
        res = model.predict(np.array([p]))[0]
        threshold = 0.8
        results = [[i, r] for i, r in enumerate(res) if r > threshold]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []

        for result in results:
            return_list.append({'intent': classes[result[0]], 'prob': str(result[1])})
        return return_list


def get_response(return_list, intents_json):
    if len(return_list) != 0:
        tag = return_list[0]['intent']

    else:
        tag = 'noanswer'

    if tag == 'datetime':
        print(time.strftime("%A"))
        print(time.strftime("%d %B %Y"))
        print(time.strftime("%H:%M:%S"))

    if tag == 'news':
        main_url = " http://newsapi.org/v2/top-headlines?country=in&apiKey=bc88c2e1ddd440d1be2cb0788d027ae2"
        open_news_page = requests.get(main_url).json()
        article = open_news_page["articles"]
        results = []

        for ar in article:
            results.append([ar["title"], ar["url"]])

        for i in range(10):
            print(i + 1, results[i][0])
            print(results[i][1], '\n')


    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if tag == i['tag']:
            result = random.choice(i['responses'])
    return result


def response(text):
    return_list = predict_class(text, model)
    response = get_response(return_list, intents)
    return response

while (1):
    x = input("you: ")
    print(response(x))
    if x.lower() in ['bye', 'goodbye', 'get lost', 'see you']:
        break

# Self learning
print('Help me Learn?')
tag = input('Please enter general category of your question  ')
flag = -1
for i in range(len(intents['intents'])):
    if tag.lower() in intents['intents'][i]['tag']:
        intents['intents'][i]['patterns'].append(input('Enter your message: '))
        intents['intents'][i]['responses'].append(input('Enter expected reply: '))
        flag = 1

if flag == -1:
    intents['intents'].append(
        {'tag': tag,
         'patterns': [input('Please enter your message')],
         'responses': [input('Enter expected reply')]})

with open('newintents.json', 'w') as outfile:
    outfile.write(json.dumps(intents, indent=4))


while True:
    user_input = input("You: ")
    if user_input.lower() in ['bye', 'goodbye', 'get lost', 'see you']:
        break
    response_text = response(user_input)
    print("Bot:", response_text)

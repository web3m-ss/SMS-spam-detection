import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report

data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/SMS-Spam-Detection/master/spam.csv", encoding= 'latin-1')
data.head()
print(data['class'].value_counts())

# clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' url ', text)
    text = re.sub(r'£|\$|\b(?:dollar|dollars|pound|pounds)\b', ' currency ', text)
    text = re.sub(r'\b\d{5,11}\b', ' phone ', text)
    text = re.sub(r'\b\d+\b', ' number ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

data['message_clean'] = data['message'].apply(clean_text)
x = np.array(data["message_clean"])
y = np.array(data["class"])

cv = TfidfVectorizer(stop_words='english', ngram_range=(1,2),sublinear_tf=True,max_features=5000)
X = cv.fit_transform(x)

X_train, X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.33,random_state=42
)

m1 = CalibratedClassifierCV(LinearSVC(C=1.5, random_state=42))
m2 = LogisticRegression(C=5.0, max_iter=1000, random_state=42)
m3 = MultinomialNB(alpha=0.1)

clf = VotingClassifier(
    estimators=[('svc', m1), ('lr', m2), ('nb', m3)],
    voting='soft'
)
clf.fit(X_train,y_train)

sample = input("Enter user input: ")
print("The user input: ",sample)
data = cv.transform([sample]).toarray()
print("Model predict as: ",clf.predict(data))

# y_pred = clf.predict(X_test)
# print("Classification Report:")
# print(classification_report(y_test, y_pred))

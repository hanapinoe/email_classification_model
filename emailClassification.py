import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

# Loading dataset
df = pd.read_csv("MLminiproject/spam_ham_dataset.csv")
df.info()

spam_harm_count = df['label_num'].value_counts()
print(spam_harm_count)

total = spam_harm_count.sum()

ratio_harm = (spam_harm_count[0]/total)*100
ratio_spam = (spam_harm_count[1]/total)*100                                                                                                                         

print("Ratio of harm:",ratio_harm)
print("Ratio of spam:",ratio_spam)

data = {
    'Ratio': [ratio_harm, ratio_spam],
    'Label': ['Harm', 'Spam']
}
df_ratio_spam_harm = pd.DataFrame(data)

"""plt.figure(figsize=(8, 6))
sns.barplot(x='Label', y='Ratio', data=df_ratio_spam_harm)

for index, row in df_ratio_spam_harm.iterrows():
    plt.annotate(f'{row["Ratio"]:f}', (index, row["Ratio"]), ha='center', va='bottom')

plt.xlabel('Label')
plt.ylabel('Ratio')
plt.title('Percentage Ratio between Spam and Harm of Dataset')
plt.show()"""

print()

# Training model
## Pre-processing data
X = df['text']
y = df['label_num']

X_train, X_, y_train, y_ = train_test_split(X, y, train_size=0.8, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_, y_, train_size=0.5, random_state=0)

vectorizer = TfidfVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

Cs_g = svc_param_selection(X_val_vec, y_val, 5)
model = svm.SVC(kernel='rbf', C=Cs_g['C'], gamma=Cs_g['gamma'])
model.fit(X_train_vec, y_train)

# Prediction
y_train_pred = model.predict(X_train_vec)
y_val_pred = model.predict(X_val_vec)
y_test_pred = model.predict(X_test_vec)

'''with open('classification_report_train.txt', 'w') as file:
    file.write(classification_report(y_train, y_train_pred))

with open('classification_report_validation.txt', 'w') as file:
    file.write(classification_report(y_val, y_val_pred))

with open('classification_report_test.txt', 'w') as file:
    file.write(classification_report(y_test, y_test_pred))'''


# Evaluate
## Train error
train_error = 1 - accuracy_score(y_train, y_train_pred)

## Validation error
val_error = 1 - accuracy_score(y_val, y_val_pred)

## Test error
test_error = 1 - accuracy_score(y_test, y_test_pred)

print("Train Error:", train_error)
print("Validation Error:", val_error)
print("Test Error:", test_error)



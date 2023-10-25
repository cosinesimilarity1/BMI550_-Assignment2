import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, make_scorer
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('vader_lexicon')

nltk.download('stopwords')
nltk.download('wordnet')

data = pd.read_csv("Dataset/fallreports_2023-9-21_train.csv")
# Pre-processing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    filtered = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalpha()]
    return ' '.join(filtered)

data = data.dropna(subset=['fall_description'])

senti_model = SentimentIntensityAnalyzer()
data['fall_sentiment'] = data['fall_description'].apply(lambda x: (senti_model.polarity_scores(x)['compound'] + 1) / 2)

data['fall_description'] = data['fall_description'].apply(preprocess)

'''
fall_location, mds_updrs_iii_hy_video: These columns have been binarized. So i dropped it
category, redcap_repeat_instrument: These columns have constant values, so dropped it
'''
data = data.drop(['category','mds_updrs_iii_hy_video','redcap_repeat_instrument','fall_location'],axis=1)  # This is of no use
data['aime2023_dataset'].replace({'Yes':1,'No':0},inplace=True)
data['fall_desc_repeat'].replace({'Yes':1,'No':0},inplace=True)
data['fog_yn'].replace({'Yes':1,'No':0},inplace=True)
data['location_binary'].replace({'Yes':1,'No':0},inplace=True)
data['mds_updrs_iii_binary'].replace({'mild':1,'severe':0},inplace=True)
data['previous_falls'].replace({'faller':1,'non-faller':0},inplace=True)
data['gender'].replace({'Male':1,'Female':0},inplace=True)
data['race'].replace({'White':1,'African American/Black':0},inplace=True)
data['ethnicity'].replace({'Hispanic or Latino':1,'Not Hispanic or Latino':0},inplace=True)
data['location_binary'].replace({'Yes':1,'No':0},inplace=True)  # YES: Home/ Indoor, NO: Other
data['education'].replace({'Completed graduate degree':1,'Completed college':2, "Completed junior college (associate's degree, technical training, etc...)":3, 'Completed high school':4},inplace=True)
data['num_falls_6_mo'].replace({'3 or more':3},inplace=True)  # YES: Home/ Indoor, NO: Other
data['fall_class'].replace({'CoM (self-induced or externally-applied)':1,'BoS (slips / trips)':2, 'Unclassifiable (falls from bed, sports-related, no data)':3},inplace=True)
# cols_with_nan = data.columns[data.isna().any(axis=0)]
# filtered_df = data[cols_with_nan]
# print(filtered_df)
data['num_falls_6_mo'].fillna(0, inplace=True)
data['fall_study_day'].fillna(0, inplace=True)
data['location_binary'].fillna(0, inplace=True)
data['fog_yn'].fillna(0, inplace=True)

'''
FEATURE ENGINEERING
Through correlation heatmap it was found that:
'age_at_enrollment', 'previous_falls' and 'fall_rate' are the three topmost positively correlated with target
'abc_total' and 'moca_total' are the topmost negatively correlated with target
'''
data['age_and_fall_rate_positive'] = data['age_at_enrollment'] * data['fall_rate']
data['abc_moca_total_negative'] = data['abc_total'] * data['moca_total']
data['unique_words'] = data['fall_description'].apply(lambda x: len(set(word_tokenize(x))))

numeric_features= ['previous_falls','age_and_fall_rate_positive','abc_moca_total_negative','fall_sentiment','unique_words']
# cols = numeric_features + ['fall_description','age_at_enrollment','fall_rate','abc_total','moca_total']
data[numeric_features].to_csv('final.csv',index=False)

X = data.drop('fog_q_class', axis=1)  # Input features
y = data['fog_q_class']   # Output or target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train['fall_description'])
X_test_tfidf = vectorizer.transform(X_test['fall_description'])
# print("Shape of train:", X_train.shape)
# print("Shape of test:", X_test.shape)
X_train = np.hstack((X_train_tfidf.toarray(), X_train[numeric_features].values))
X_test = np.hstack((X_test_tfidf.toarray(), X_test[numeric_features].values))
# print("Shape of train:", X_train.shape)
# print("Shape of test:", X_test.shape)

# 3. Set up classifiers and parameters for GridSearch
# models = {
#     'Naive Bayes': MultinomialNB(),
#     'Logistic Regression': LogisticRegression(max_iter=10000),
#     'SVM': SVC(),
#     'KNN': KNeighborsClassifier(),
#     'Decision Trees': DecisionTreeClassifier(),
#     'AdaBoost': AdaBoostClassifier(),
#     # 'Random Forest': RandomForestClassifier(),
#     #
#     # 'XGBoost': XGBClassifier(objective='binary:logistic', eval_metric="logloss"),
#     # 'GradientBoosting': GradientBoostingClassifier(),
# }
#
# param_grids = {
#     'Naive Bayes': { 'alpha': [0.01, 0.1, 1, 10, 100]},
#
#     'Logistic Regression': {'C': [0.001, 0.01, 0.1, 1, 10, 100],
#         'penalty': ['l1', 'l2'],
#         'solver': ['lbfgs', 'liblinear', 'sag']},
#
#     'SVM': {'C': [0.01, 0.1, 1, 10], 'kernel': ['linear', 'rbf']},
#     # 'SVM': {'C': [0.001, 0.01, 0.1, 1, 10],
#     #     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#     #     'gamma': ['scale', 'auto']},
#
#     'KNN': {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']},
#
#     'Decision Tree': {'max_depth': [None, 5, 10, 15], 'criterion': ['gini', 'entropy']},
#     # 'Decision Trees': {
#     #     'criterion': ['gini', 'entropy'],
#     #     'max_depth': [None,5, 10, 20, 30, 40, 50],
#     #     'min_samples_split': [2, 5, 10],
#     #     'min_samples_leaf': [1, 2, 4]
#     # },
#
#     'AdaBoost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.5]}
#     # 'Random Forest': {'n_estimators': [10, 50, 100, 200],
#     #     'criterion': ['gini', 'entropy'],
#     #     'max_depth': [None, 5,10, 20, 30, 40, 50],
#     #     'min_samples_split': [2, 5, 10],
#     #     'min_samples_leaf': [1, 2, 4]},
#     #
#     # 'XGBoost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.5]},
#     # 'GradientBoosting': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.5]},
#
# }
#
# best_models = {}
# for classifier_name, classifier in models.items():
#     grid_search = GridSearchCV(classifier, param_grids[classifier_name], cv=5)
#     grid_search.fit(X_train, y_train)
#     best_models[classifier_name] = grid_search.best_estimator_
#     models[classifier_name].set_params(**grid_search.best_params_)
#     predictions = grid_search.predict(X_test)
#     print(f"Model: {classifier_name}")
#     print("Best Parameters:", grid_search.best_params_)
#     print("Accuracy:", accuracy_score(y_test, predictions))
#     print("Micro F1:", f1_score(y_test, predictions, average='micro'))
#     print("Macro F1:", f1_score(y_test, predictions, average='macro'))
#     print("------------")
# print("\n\n Best models parameters")
# print(best_models)

classifiers= {'Naive Bayes': MultinomialNB(alpha=0.01),
              'Logistic Regression': LogisticRegression(C=10,penalty='l1', max_iter=50000, solver='liblinear'),
              'Decision Trees': DecisionTreeClassifier(criterion='gini', max_depth=15),
              'SVM': SVC(probability=True,C=1, kernel='linear'),
              'KNN': KNeighborsClassifier(n_neighbors=3, weights='distance'),
              'AdaBoost': AdaBoostClassifier(learning_rate=0.1, n_estimators=100)
              }

def create_real_test_set_texts():
    data = pd.read_csv('Dataset/fallreports_2023-9-21_test.csv')
    data = data.dropna(subset=['fall_description'])
    senti_model = SentimentIntensityAnalyzer()
    data['fall_sentiment'] = data['fall_description'].apply(
        lambda x: (senti_model.polarity_scores(x)['compound'] + 1) / 2)

    data['fall_description'] = data['fall_description'].apply(preprocess)
    data = data.drop(['category', 'mds_updrs_iii_hy_video', 'redcap_repeat_instrument', 'fall_location'],
                     axis=1)  # This is of no use
    data['aime2023_dataset'].replace({'Yes': 1, 'No': 0}, inplace=True)
    data['fall_desc_repeat'].replace({'Yes': 1, 'No': 0}, inplace=True)
    data['fog_yn'].replace({'Yes': 1, 'No': 0}, inplace=True)
    data['location_binary'].replace({'Yes': 1, 'No': 0}, inplace=True)
    data['mds_updrs_iii_binary'].replace({'mild': 1, 'severe': 0}, inplace=True)
    data['previous_falls'].replace({'faller': 1, 'non-faller': 0}, inplace=True)
    data['gender'].replace({'Male': 1, 'Female': 0}, inplace=True)
    data['race'].replace({'White': 1, 'African American/Black': 0}, inplace=True)
    data['ethnicity'].replace({'Hispanic or Latino': 1, 'Not Hispanic or Latino': 0}, inplace=True)
    data['location_binary'].replace({'Yes': 1, 'No': 0}, inplace=True)  # YES: Home/ Indoor, NO: Other
    data['education'].replace({'Completed graduate degree': 1, 'Completed college': 2,
                               "Completed junior college (associate's degree, technical training, etc...)": 3,
                               'Completed high school': 4}, inplace=True)
    data['num_falls_6_mo'].replace({'3 or more': 3}, inplace=True)  # YES: Home/ Indoor, NO: Other
    data['fall_class'].replace({'CoM (self-induced or externally-applied)': 1, 'BoS (slips / trips)': 2,
                                'Unclassifiable (falls from bed, sports-related, no data)': 3}, inplace=True)
    data['num_falls_6_mo'].fillna(0, inplace=True)
    data['fall_study_day'].fillna(0, inplace=True)
    data['location_binary'].fillna(0, inplace=True)
    data['fog_yn'].fillna(0, inplace=True)
    data['age_and_fall_rate_positive'] = data['age_at_enrollment'] * data['fall_rate']
    data['abc_moca_total_negative'] = data['abc_total'] * data['moca_total']
    data['unique_words'] = data['fall_description'].apply(lambda x: len(set(word_tokenize(x))))
    numeric_features = ['previous_falls', 'age_and_fall_rate_positive', 'abc_moca_total_negative', 'fall_sentiment',
                        'unique_words']
    X = data.drop('fog_q_class', axis=1)  # Input features
    y_real_test = data['fog_q_class']  # Output or target variable
    X_real_test = vectorizer.transform(X['fall_description'])
    X_real_test = np.hstack((X_real_test.toarray(), X[numeric_features].values))
    return X_real_test,y_real_test

def only_text_without_vector():
    data = pd.read_csv('Dataset/fallreports_2023-9-21_test.csv')
    data = data.dropna(subset=['fall_description'])
    senti_model = SentimentIntensityAnalyzer()
    data['fall_sentiment'] = data['fall_description'].apply(
        lambda x: (senti_model.polarity_scores(x)['compound'] + 1) / 2)

    data['fall_description'] = data['fall_description'].apply(preprocess)
    data = data.drop(['category', 'mds_updrs_iii_hy_video', 'redcap_repeat_instrument', 'fall_location'],
                     axis=1)  # This is of no use
    data['aime2023_dataset'].replace({'Yes': 1, 'No': 0}, inplace=True)
    data['fall_desc_repeat'].replace({'Yes': 1, 'No': 0}, inplace=True)
    data['fog_yn'].replace({'Yes': 1, 'No': 0}, inplace=True)
    data['location_binary'].replace({'Yes': 1, 'No': 0}, inplace=True)
    data['mds_updrs_iii_binary'].replace({'mild': 1, 'severe': 0}, inplace=True)
    data['previous_falls'].replace({'faller': 1, 'non-faller': 0}, inplace=True)
    data['gender'].replace({'Male': 1, 'Female': 0}, inplace=True)
    data['race'].replace({'White': 1, 'African American/Black': 0}, inplace=True)
    data['ethnicity'].replace({'Hispanic or Latino': 1, 'Not Hispanic or Latino': 0}, inplace=True)
    data['location_binary'].replace({'Yes': 1, 'No': 0}, inplace=True)  # YES: Home/ Indoor, NO: Other
    data['education'].replace({'Completed graduate degree': 1, 'Completed college': 2,
                               "Completed junior college (associate's degree, technical training, etc...)": 3,
                               'Completed high school': 4}, inplace=True)
    data['num_falls_6_mo'].replace({'3 or more': 3}, inplace=True)  # YES: Home/ Indoor, NO: Other
    data['fall_class'].replace({'CoM (self-induced or externally-applied)': 1, 'BoS (slips / trips)': 2,
                                'Unclassifiable (falls from bed, sports-related, no data)': 3}, inplace=True)
    data['num_falls_6_mo'].fillna(0, inplace=True)
    data['fall_study_day'].fillna(0, inplace=True)
    data['location_binary'].fillna(0, inplace=True)
    data['fog_yn'].fillna(0, inplace=True)
    data['age_and_fall_rate_positive'] = data['age_at_enrollment'] * data['fall_rate']
    data['abc_moca_total_negative'] = data['abc_total'] * data['moca_total']
    data['unique_words'] = data['fall_description'].apply(lambda x: len(set(word_tokenize(x))))
    numeric_features = ['previous_falls', 'age_and_fall_rate_positive', 'abc_moca_total_negative', 'fall_sentiment',
                        'unique_words']
    X = data.drop('fog_q_class', axis=1)  # Input features
    y_real_test = data['fog_q_class']  # Output or target variable
    return X,y_real_test


################## Best model selection based on provided test dataset ##########################
results = {}
for name, clf in classifiers.items():
    X_real_test, y_real_test = create_real_test_set_texts()
    clf.fit(X_train, y_train)  # With selected hyperparameters
    predictions = clf.predict(X_real_test)
    accuracy = accuracy_score(y_real_test, predictions)
    f1_micro = f1_score(y_real_test, predictions, average='micro')
    f1_macro = f1_score(y_real_test, predictions, average='macro')
    results[name] = (accuracy, f1_micro, f1_macro)
    print(f"{name} - Accuracy: {accuracy}, F1 Micro: {f1_micro}, F1 Macro: {f1_macro}")

############# Ensemble #######################
ensemble_clf = VotingClassifier(estimators=[(name, clf) for name, clf in classifiers.items()], voting='hard')
ensemble_clf.fit(X_train, y_train)
X_real_test, y_real_test = create_real_test_set_texts()
ensemble_predictions = ensemble_clf.predict(X_real_test)
ensemble_accuracy = accuracy_score(y_real_test, ensemble_predictions)
ensemble_f1_micro = f1_score(y_real_test, ensemble_predictions, average='micro')
ensemble_f1_macro = f1_score(y_real_test, ensemble_predictions, average='macro')
print(f"Ensemble - Accuracy: {ensemble_accuracy}, F1 Micro: {ensemble_f1_micro}, F1 Macro: {ensemble_f1_macro}")

ensemble_clf = VotingClassifier(estimators=[(name, clf) for name, clf in classifiers.items()], voting='soft')
ensemble_clf.fit(X_train, y_train)
X_real_test, y_real_test = create_real_test_set_texts()
ensemble_predictions = ensemble_clf.predict(X_real_test)
ensemble_accuracy = accuracy_score(y_real_test, ensemble_predictions)
ensemble_f1_micro = f1_score(y_real_test, ensemble_predictions, average='micro')
ensemble_f1_macro = f1_score(y_real_test, ensemble_predictions, average='macro')
print(f"Ensemble - Accuracy: {ensemble_accuracy}, F1 Micro: {ensemble_f1_micro}, F1 Macro: {ensemble_f1_macro}")


########################## Performance vs Training Set Size #################
train_sizes = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
train_sizes_results = {"train_size": [], "accuracy": [], "micro_f1": [], "macro_f1": []}
cols = ['previous_falls','age_and_fall_rate_positive','abc_moca_total_negative','fall_sentiment','unique_words','fall_description']
print("Cols :",cols)
for size in train_sizes:
    X_train, _, y_train, _ = train_test_split(X[cols], y, train_size=size, random_state=42)

    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    X_train_vector = vectorizer.fit_transform(X_train['fall_description'])
    x_real_test, y_real_test= only_text_without_vector()
    X_test_vector = vectorizer.transform(x_real_test['fall_description'])

    X_train = np.hstack((X_train_vector.toarray(), X_train[numeric_features].values))
    X_test = np.hstack((X_test_vector.toarray(), x_real_test[numeric_features].values))
    # X_test = np.hstack((X_test.toarray(), X_test[numeric_features].values))
    best_model = KNeighborsClassifier(n_neighbors=3, weights='distance')
    best_model.fit(X_train, y_train)
    predictions = best_model.predict(X_test)

    train_sizes_results["train_size"].append(size)
    train_sizes_results["accuracy"].append(accuracy_score(y_real_test, predictions))
    train_sizes_results["micro_f1"].append(f1_score(y_real_test, predictions, average='micro'))
    train_sizes_results["macro_f1"].append(f1_score(y_real_test, predictions, average='macro'))

plt.plot(train_sizes_results["train_size"], train_sizes_results["micro_f1"], '-o', label="Micro F1")
plt.xlabel("Training Set Size")
plt.ylabel("Micro F1 Score")
plt.legend()
plt.title("Training Size vs. Performance")
plt.show()

############################# Ablation study ##################################
features = ['previous_falls','age_and_fall_rate_positive','abc_moca_total_negative','fall_sentiment','unique_words']

for feature in features:
    columns_to_use = ['previous_falls', 'age_and_fall_rate_positive', 'abc_moca_total_negative', 'fall_sentiment',
                      'unique_words']

    print("feature to be removed: ",feature)
    accuracies = []
    columns_to_use.remove(feature)
    print("cols using now:", columns_to_use)

    X = data.drop('fog_q_class', axis=1)  # Input features
    y = data['fog_q_class']  # Output or target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train['fall_description'])
    X_test_tfidf = vectorizer.transform(X_test['fall_description'])
    print("Shape of train:", X_train.shape)
    print("Shape of test:", X_test.shape)
    X_train = np.hstack((X_train_tfidf.toarray(), X_train[columns_to_use].values))
    X_test = np.hstack((X_test_tfidf.toarray(), X_test[columns_to_use].values))

    model = AdaBoostClassifier(learning_rate=0.1, n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    micro_f1 = f1_score(y_test, y_pred, average='micro')
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    print("Accuracy:", accuracy)
    print("Micro F1:", micro_f1)
    print("Macro F1:", macro_f1)
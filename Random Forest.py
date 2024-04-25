import pathlib
import pandas as pd
import numpy as np
from spacy.lang.ru.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import  StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


base_path = pathlib.Path(__file__).parent
temp_path = base_path.joinpath('temp_dir')

# model parameters
state = np.random.RandomState(0)
seed = np.random.seed(22)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=state)

# load datasets that were used to train and test LLM
train_x = pd.read_csv(temp_path.joinpath('train_LLM_outputs.csv'), index_col='Unnamed: 0')
test_x = pd.read_csv(temp_path.joinpath('test_LLM_outputs.csv'), index_col='Unnamed: 0')
train_y = pd.DataFrame()
test_y = pd.DataFrame()
train_y['mark'] = train_x['mark']
test_y['mark'] = test_x['mark']
train = train_x[['text_clean', 'lemmas', 'text_len_log', 'time_read_log', 'log_punct_num']]
test = test_x[['text_clean', 'lemmas', 'text_len_log', 'time_read_log', 'log_punct_num']]

# parameters for Grid Search
parameters = {
    'forest__n_estimators': [int(x) for x in np.linspace(start=10, stop=400, num=100)],
    'forest__max_depth': [int(x) for x in np.linspace(24, 28, num=2)],
}

# Do grid search over k, n_components and C:
numeric_features = ["text_len_log", "time_read_log", "log_punct_num"]
numeric_transformer = Pipeline(
    steps=[("scaler", StandardScaler())]
)

column_trans = ColumnTransformer(
    transformers=[
        ('text_cl', TfidfVectorizer(analyzer=u'word',
                                    stop_words=list(STOP_WORDS),
                                    encoding='utf-8',
                                    max_df=600,
                                    ngram_range=(1,1)),
         'text_clean'),
        ('lem', TfidfVectorizer(analyzer=u'word',
                                stop_words=list(STOP_WORDS),
                                encoding='utf-8',
                                ngram_range=(1, 1)),
         'lemmas'),
        ("num",
         numeric_transformer,
         numeric_features)])

# Include the classifier in the main pipeline
pipeline = Pipeline([
    ('features', column_trans),
    ('forest', RandomForestClassifier(class_weight="balanced", random_state=12345))
])

# Perform GridSearch
rf_model = GridSearchCV(pipeline,
                           param_grid=parameters,
                           cv=skf,
                           n_jobs=-1,
                           error_score='raise',
                           scoring='f1_weighted',
                           return_train_score=True,
                           verbose=1)

rf_model.fit(train, train_y)
best = rf_model.best_estimator_
print(rf_model.best_estimator_)
print(rf_model.best_score_)

test_pred = rf_model.predict(test)

# Print the precision and recall, among other metrics
print(classification_report(test_y, test_pred, digits=3))
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(
    test_y, test_pred, normalize="true",
    labels=rf_model.classes_),
                       display_labels=rf_model.classes_).plot()
plt.show()





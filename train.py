import pandas as pd
import pickle as pl

from sklearn.model_selection import train_test_split as tts
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score


# Training model
def train():

    df = pd.read_csv('coords.csv')

    # Get data from the file
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Prepating train/test
    X_train, X_test, y_train, y_test = tts(
        X, y, test_size=0.3, random_state=1234)

    # Training models
    pipelines = {
        'lr': make_pipeline(StandardScaler(), LogisticRegression()),
        'rc': make_pipeline(StandardScaler(), RidgeClassifier()),
        'rf': make_pipeline(StandardScaler(), RandomForestClassifier()),
        'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier()),
    }

    fit_data = {}

    # Iterating through pipelines
    for al, pipeline in pipelines.items():

        model = pipeline.fit(X_train, y_train)
        fit_data[al] = model

    # Evaluate model
    for al, model in fit_data.items():

        prediction = model.predict(X_test)
        print(al, accuracy_score(y_test, prediction))

    print(fit_data['rf'].predict(X_test))

    # Exporting trained model (RandomForest in this case) to emotions.pkl file
    #with open('emotions.pkl', 'wb') as f:
    #
    #    pl.dump(fit_data['rf'], f)

train()

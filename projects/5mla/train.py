#!/opt/conda/envs/dsenv/bin/python
import sys, os


import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
#
# Import model definition
#
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
import mlflow
import mlflow.sklearn
#
# Dataset fields
#
try:
  param = sys.argv[2]
  train_path = sys.argv[1]
except:
  
  sys.exit(1)
numeric_features = ["if"+str(i) for i in range(1,14)]
categorical_features = ["cf"+str(i) for i in range(1,27)]+ ["day_number"]

fields = ["id", "label"] + numeric_features + categorical_features
remove_cat_features = ['cf20', 'cf10', 'cf1', 'cf22', 'cf11', 'cf12', 'cf21', 'cf23']
categorical_features_new = list(categorical_features)
for feat in remove_cat_features:
    categorical_features_new.remove(feat)
new_fields = ["id", "label"] + numeric_features + categorical_features_new
#
# Model pipeline
#

# We create the preprocessing pipelines for both numeric and categorical data.

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
#    ('scaler', StandardScaler())
])


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features_new)
    ]
)
logreg_opts = dict(
  #'lbfgs'
    solver=param,
    class_weight='balanced',
)

# Now we have a full prediction pipeline.
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('logreg', LogisticRegression(**logreg_opts))
])


#
# Logging initialization
#


# Read dataset
#
#fields = """doc_id,hotel_name,hotel_url,street,city,state,country,zip,class,price,
#num_reviews,CLEANLINESS,ROOM,SERVICE,LOCATION,VALUE,COMFORT,overall_ratingsource""".replace("\n",'').split(",")

read_table_opts = dict(sep="\t", names=fields, index_col=False)
df = pd.read_table(train_path, **read_table_opts)

#split train/test
names = [fields[0]] + fields[2:]
X_train, X_test, y_train, y_test = train_test_split(
        df[names], df[fields[1]], test_size=0.33, random_state=42
)

#
# Train the model
#
with mlflow.start_run():
    estimator = model.fit(X_train, y_train)
    mlflow.log_params(estimator['logreg'].get_params())
    y_pred = estimator.predict_proba(X_test)[:,1]
    log_loss = sklearn.metrics.log_loss(y_test, y_pred)
    mlflow.log_metric("log_loss", log_loss)
    mlflow.sklearn.log_model(estimator, "reg_model")

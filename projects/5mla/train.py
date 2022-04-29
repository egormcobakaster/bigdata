#!/opt/conda/envs/dsenv/bin/python
import sys, os


import pandas as pd
from sklearn.model_selection import train_test_split

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
import sklearn
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
categorical_features_new = ['cf2', 'cf3', 'cf4', 'cf5', 'cf6', 'cf15', 'cf9', 'cf26']

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

mlflow.log_param("param2", "This is a param2")
read_table_opts = dict(sep="\t", names=fields, index_col=False)
df = pd.read_table(train_path, **read_table_opts)
mlflow.log_param("param0", "This is a param0")

names = [fields[0]] + fields[2:]
mlflow.log_param("param3", "This is a param3")
mlflow.log_param("param1", "This is a param1")
#
# Train the model
X = df[names]
mlflow.log_param("param5", "This is a param5")
y = df[fields[1]]

estimator = model.fit(X[1000000], y[1000000])
mlflow.log_param("param6", "This is a param6")
mlflow.log_params(estimator['logreg'].get_params())
y_pred = estimator.predict_proba(df[names][:10000])[:,1]
log_loss = sklearn.metrics.log_loss(df[fields[1]][:10000], y_pred)
mlflow.log_metric("log_loss", log_loss)
mlflow.sklearn.log_model(estimator, artifact_path="model")

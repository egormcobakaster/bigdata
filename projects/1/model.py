from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV

#
# Dataset fields
#

numeric_features = ["if"+str(i) for i in range(1,14)]
categorical_features = ["cf"+str(i) for i in range(1,27)]+ ["day_number"]

fields = ["id", "label"] + numeric_features + categorical_features
remove_cat_features = ['cf20', 'cf10', 'cf1', 'cf22', 'cf11', 'cf12', 'cf21', 'cf23', 'day_number']
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
    solver='lbfgs',
    class_weight='balanced',
)

# Now we have a full prediction pipeline.
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('logreg', LogisticRegression(**logreg_opts))
])

#!/opt/conda/envs/dsenv/bin/python
import sys, os
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
#
# Import model definition
#
from model import model, fields


#
# Logging initialization
#
logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))

#
# Read script arguments
#
try:
  proj_id = sys.argv[1]
  train_path = sys.argv[2]
except:
  logging.critical("Need to pass both project_id and train dataset path")
  sys.exit(1)


logging.info(f"TRAIN_ID {proj_id}")
logging.info(f"TRAIN_PATH {train_path}")

#
# Read dataset
#
#fields = """doc_id,hotel_name,hotel_url,street,city,state,country,zip,class,price,
#num_reviews,CLEANLINESS,ROOM,SERVICE,LOCATION,VALUE,COMFORT,overall_ratingsource""".replace("\n",'').split(",")

read_table_opts = dict(sep="\t", names=fields, index_col=False)
df = pd.read_table(train_path, **read_table_opts)

#split train/test
X_train, X_test, y_train, y_test = train_test_split(
        df[[fields[0] + fields[2:]]], df[fields[1]], test_size=0.33, random_state=42
)

#
# Train the model
#
model1 =model.fit(X_train, y_train)

model_score = model1.score(X_test, y_test)

logging.info(f"model score: {model_score:.3f}")

# save the model
joblib.dump(model1, "{}.joblib".format(proj_id))
joblib_model = joblib.load("1.joblib")
model_score = joblib_model.score(X_test, y_test)

logging.info(f"model score: {model_score:.3f}")

#!/opt/conda/envs/dsenv/bin/python

import sys, os
import logging
from joblib import load
import pandas as pd
import numpy as np

sys.path.append('.')
from model import fields, numeric_features, categorical_features, features

#
# Init the logger
#
logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))

#load the model
model = load("1.joblib")
logging.info("num {}".format(11))
#fields = """doc_id,hotel_name,hotel_url,street,city,state,country,zip,class,price,
#num_reviews,CLEANLINESS,ROOM,SERVICE,LOCATION,VALUE,COMFORT,overall_ratingsource""".replace("\n",'').split(",")

#read and infere
read_opts=dict(
        sep='\t', names=[fields[0] + fields[2:]], index_col='id',
        iterator=True, chunksize=100000
)
for df in pd.read_table(sys.stdin, **read_opts):
    if len(df) == 0:
        continue
    if df.size == 0:
        continue
    pred = model.predict(df)
    out = zip(df.index, pred)
    print("\n".join(["{0}\t{1}".format(*i) for i in out]))

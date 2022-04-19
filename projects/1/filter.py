#!/opt/conda/envs/dsenv/bin/python
import numpy as np
import sys
import os
from glob import glob
import logging
import pandas as pd
sys.path.append('.')
from model import fields

#
# Init the logger
#
logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))

#
# import the filter
#
filter_cond_files = glob('filter_cond*.py')
logging.info(f"FILTERS {filter_cond_files}")

if len(filter_cond_files) != 1:
    logging.critical("Must supply exactly one filter")
    sys.exit(1)

exec(open(filter_cond_files[0]).read())
for line in sys.stdin:
    if line.startswith(fields[0]):
        continue
    #unpack into a tuple/dict
    values = line.rstrip().split('\t')
    record = dict(zip(fields, values))
    if filter_cond(record):
        output = "\t".join([data_line[x] for x in fields])
        print(output)

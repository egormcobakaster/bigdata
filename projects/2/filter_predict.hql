add file projects/2/model.py;
add file projects/2/predict.py;
add file 2.joblib;
insert into hw2_pred select * from(select transform(*) using 'predict.py' as (id, pred) from hw2_test where if1>20 and if1<40) temp_name;

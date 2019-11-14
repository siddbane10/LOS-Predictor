import firebase_admin
import tensorflow as tf
import csv
import pandas as pd

from firebase_admin import credentials
from firebase_admin import db
# Fetch the service account key JSON file contents
#firebase_admin.initialize_app(cred)
# Initialize the app with a service account, granting admin privileges

cred = credentials.Certificate('hospital-db-a5b87-firebase-adminsdk-ij0wt-abe72ad69a.json')
#firebase_admin.initialize_app(cred, {'databaseURL': 'https://hospital-db-a5b87.firebaseio.com/'})
LOAD = 'Loading prediction. Please Wait.'
users_ref =  db.reference('pred/res')
users_ref.set(  { 'Deep':str(LOAD) }  )
#RUN ONCE
#firebase_admin.initialize_app(cred, {'databaseURL': 'https://hospital-db-a5b87.firebaseio.com/'})

data = pd.read_csv('clean-db/TRAINING_SET_W_DIAGNOSIS.csv')

data['DIAGNOSIS'].value_counts()

admin_type = tf.feature_column.categorical_column_with_hash_bucket('ADMISSION_TYPE',hash_bucket_size=50000)
admin_type_em = tf.feature_column.embedding_column(admin_type,dimension=10)
admin_loc = tf.feature_column.categorical_column_with_hash_bucket('ADMISSION_LOCATION',hash_bucket_size=50000)
admin_loc_em = tf.feature_column.embedding_column(admin_loc,dimension=10)
disc_loc = tf.feature_column.categorical_column_with_hash_bucket('DISCHARGE_LOCATION',hash_bucket_size=50000)
disc_loc_em = tf.feature_column.embedding_column(disc_loc,dimension=10)
insurance = tf.feature_column.categorical_column_with_hash_bucket('INSURANCE',hash_bucket_size=10000)
insurance_em = tf.feature_column.embedding_column(insurance,dimension=10)
eth = tf.feature_column.categorical_column_with_hash_bucket('ETHNICITY', hash_bucket_size=50000)
eth_em = tf.feature_column.embedding_column(eth,dimension=10)
gen = tf.feature_column.categorical_column_with_hash_bucket('GENDER',hash_bucket_size=5)
gen_em = tf.feature_column.embedding_column(gen,dimension=5)
ed_los = tf.feature_column.numeric_column('EDLOS', dtype=tf.float64)
diag = tf.feature_column.categorical_column_with_hash_bucket('DIAGNOSIS',hash_bucket_size=1600)
diag_em = tf.feature_column.embedding_column(diag,dimension=1600)

emfeat_cols = [admin_type_em,admin_loc_em,disc_loc_em,insurance_em,eth_em,ed_los,gen_em,diag_em]
feat_cols = [admin_type,admin_loc,disc_loc,insurance,eth,ed_los,gen,diag]

#x_data = pd.DataFrame(columns=[['ADMISSION_TYPE', 'ADMISSION_LOCATION','DISCHARGE_LOCATION' , 'INSURANCE', 'ETHNICITY','EDLOS','GENDER','DIAGNOSIS']])
#x_data[['ADMISSION_TYPE', 'ADMISSION_LOCATION', 'DISCHARGE_LOCATION', 'INSURANCE', 'ETHNICITY', 'EDLOS','GENDER','DIAGNOSIS']] = data[['ADMISSION_TYPE', 'ADMISSION_LOCATION', 'DISCHARGE_LOCATION','INSURANCE', 'ETHNICITY', 'EDLOS','GENDER','DIAGNOSIS']]

from sklearn.model_selection import train_test_split
x_data = pd.DataFrame(columns=[['ADMISSION_TYPE', 'ADMISSION_LOCATION','DISCHARGE_LOCATION' , 'INSURANCE', 'ETHNICITY','EDLOS','GENDER','DIAGNOSIS']])
x_data[['ADMISSION_TYPE', 'ADMISSION_LOCATION','DISCHARGE_LOCATION' , 'INSURANCE', 'ETHNICITY', 'EDLOS','GENDER','DIAGNOSIS']] = data[['ADMISSION_TYPE', 'ADMISSION_LOCATION','DISCHARGE_LOCATION' , 'INSURANCE', 'ETHNICITY', 'EDLOS','GENDER','DIAGNOSIS']]
	#Show corr for EDLOS
bucket_LOS = pd.DataFrame(columns=['LOS_cat'])
cat_series = []
split_day = 6

for x in data['LOS']:
    if x<=split_day:
        cat_series.append(0)
    elif x>=split_day+1:
        cat_series.append(1)
bucket_LOS['LOS_cat'] = cat_series
#cat_series
labels = data['LOS']
bucket_Label = bucket_LOS['LOS_cat']

X_train, X_test, y_train, y_test = train_test_split(x_data,bucket_Label,test_size=0.3, random_state=101)

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=1000,num_epochs=100,shuffle=True)

#ClassLinModel = tf.estimator.LinearClassifier(feature_columns=feat_cols,n_classes=2)
ClassDNNModel = tf.estimator.DNNClassifier(feature_columns=emfeat_cols,hidden_units=[100,100,100,100,100],n_classes=2)
#ClassLinModel.train(input_fn=input_func,steps=5000)
ClassDNNModel.train(input_fn=input_func,steps=2000)

#x_data['DISCHARGE_LOCATION'].value_counts()
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,y=y_test,batch_size=500,num_epochs=1,shuffle=False)

ClassDNNresults = ClassDNNModel.evaluate(eval_input_func)

ClassDNNresults

bucket_LOS = pd.DataFrame(columns=['LOS_cat'])
cat_series = []
split_day = 5
for x in data['LOS']:
    if x<=split_day:
        cat_series.append(0)
    elif x>=split_day+1:
        cat_series.append(1)
bucket_LOS['LOS_cat'] = cat_series
#cat_series
labels = data['LOS']
bucket_Label = bucket_LOS['LOS_cat']

X_train, X_test, y_train, y_test = train_test_split(x_data,bucket_Label,test_size=0.3, random_state=101)

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=1000,num_epochs=100,shuffle=True)

#ClassLinModel = tf.estimator.LinearClassifier(feature_columns=feat_cols,n_classes=2)
ClassDNNModel5 = tf.estimator.DNNClassifier(feature_columns=emfeat_cols,hidden_units=[100,100,100,100,100],n_classes=2)
#ClassLinModel.train(input_fn=input_func,steps=5000)
ClassDNNModel5.train(input_fn=input_func,steps=2000)

#x_data['DISCHARGE_LOCATION'].value_counts()
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,y=y_test,batch_size=500,num_epochs=1,shuffle=False)

ClassDNNresults5 = ClassDNNModel.evaluate(eval_input_func)

ClassDNNresults5

bucket_LOS = pd.DataFrame(columns=['LOS_cat'])
cat_series = []
split_day = 4
for x in data['LOS']:
    if x<=split_day:
        cat_series.append(0)
    elif x>=split_day+1:
        cat_series.append(1)
bucket_LOS['LOS_cat'] = cat_series
#cat_series
labels = data['LOS']
bucket_Label = bucket_LOS['LOS_cat']

X_train, X_test, y_train, y_test = train_test_split(x_data,bucket_Label,test_size=0.3, random_state=101)

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=1000,num_epochs=100,shuffle=True)

#ClassLinModel = tf.estimator.LinearClassifier(feature_columns=feat_cols,n_classes=2)
ClassDNNModel4 = tf.estimator.DNNClassifier(feature_columns=emfeat_cols,hidden_units=[100,100,100,100,100],n_classes=2)
#ClassLinModel.train(input_fn=input_func,steps=5000)
ClassDNNModel4.train(input_fn=input_func,steps=2000)

#x_data['DISCHARGE_LOCATION'].value_counts()
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,y=y_test,batch_size=500,num_epochs=1,shuffle=False)

ClassDNNresults4 = ClassDNNModel.evaluate(eval_input_func)

ClassDNNresults4

bucket_LOS = pd.DataFrame(columns=['LOS_cat'])
cat_series = []
split_day = 3

for x in data['LOS']:
    if x<=split_day:
        cat_series.append(0)
    elif x>=split_day+1:
        cat_series.append(1)
bucket_LOS['LOS_cat'] = cat_series
#cat_series
labels = data['LOS']
bucket_Label = bucket_LOS['LOS_cat']

X_train, X_test, y_train, y_test = train_test_split(x_data,bucket_Label,test_size=0.3, random_state=101)

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=1000,num_epochs=100,shuffle=True)

#ClassLinModel = tf.estimator.LinearClassifier(feature_columns=feat_cols,n_classes=2)
ClassDNNModel3 = tf.estimator.DNNClassifier(feature_columns=emfeat_cols,hidden_units=[100,100,100,100,100],n_classes=2)
#ClassLinModel.train(input_fn=input_func,steps=5000)
ClassDNNModel3.train(input_fn=input_func,steps=2000)

#x_data['DISCHARGE_LOCATION'].value_counts()
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,y=y_test,batch_size=500,num_epochs=1,shuffle=False)

ClassDNNresults3 = ClassDNNModel.evaluate(eval_input_func)

ClassDNNresults3

#For 2 days

bucket_LOS = pd.DataFrame(columns=['LOS_cat'])
cat_series = []
split_day = 2

for x in data['LOS']:
    if x<=split_day:
        cat_series.append(0)
    elif x>=split_day+1:
        cat_series.append(1)
bucket_LOS['LOS_cat'] = cat_series
#cat_series
labels = data['LOS']
bucket_Label = bucket_LOS['LOS_cat']

X_train, X_test, y_train, y_test = train_test_split(x_data,bucket_Label,test_size=0.3, random_state=101)

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=1000,num_epochs=100,shuffle=True)

#ClassLinModel = tf.estimator.LinearClassifier(feature_columns=feat_cols,n_classes=2)
ClassDNNModel2 = tf.estimator.DNNClassifier(feature_columns=emfeat_cols,hidden_units=[100,100,100,100,100],n_classes=2)
#ClassLinModel.train(input_fn=input_func,steps=5000)
ClassDNNModel2.train(input_fn=input_func,steps=2000)

#x_data['DISCHARGE_LOCATION'].value_counts()
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,y=y_test,batch_size=500,num_epochs=1,shuffle=False)

ClassDNNresults2 = ClassDNNModel.evaluate(eval_input_func)

ClassDNNresults2

accuracy = ClassDNNresults['accuracy']
accuracy = round(accuracy*100)
accuracy
probability = ClassDNNresults['auc']
probability5 = ClassDNNresults5['auc']
probability4 = ClassDNNresults4['auc']
probability3 = ClassDNNresults3['auc']
probability2 = ClassDNNresults2['auc']

probability = str(probability)
probability2 = str(probability2)
probability3 = str(probability3)
probability4 = str(probability4)
probability5 = str(probability5)
probability
#firebase_admin.initialize_app(cred)
# Initialize the app with a service account, granting admin privileges

#RUN ONCE
#firebase_admin.initialize_app(cred, {'databaseURL': 'https://hospital-db-a5b87.firebaseio.com/'})

# As an admin, the app has access to read and write all data, regradless of Security Rules

#spl = db.reference('/users/entry')
#print(spl.get()['n1'])

#firebase = firebase_admin.FirebaseApplication('https://hospital-db-a5b87.firebaseio.com/', None)

#while True:
	# result = firebase.get('/users', None)



users_ref =  db.reference('pred/res')
users_ref.set(  { 'Deep':str(LOAD) }  )

result2 = db.reference('/patients/entry')
result2
	#spl = spl.get()

pat_id = result2.get()['id']
pat_id

df = pd.DataFrame(columns=[['ADMISSION_TYPE', 'ADMISSION_LOCATION','DISCHARGE_LOCATION','INSURANCE', 'ETHNICITY', 'EDLOS','GENDER','DIAGNOSIS']])
		#UPLOAD PRED


#row = [result2.get()['id'],result2.get()['a_day'],result2.get()['a_month'],result2.get()['ad_type'],result2.get()['ad_loc'], result2.get()['discharge_loc'],result2.get()['insurance'],result2.get()['ethnicity'],result2.get()['ed_min'],result2.get()['gender'],1,'']
#row
row2 = [result2.get()['ad_type'],result2.get()['ad_loc'],'HOME',result2.get()['insurance'],result2.get()['ethnicity'],float(result2.get()['ed_min']),result2.get()['gender'],result2.get()['diagnosis']]
#row2

row2 = pd.DataFrame(row2)
row2 = row2.transpose()
#row3 = pd.DataFrame(row2,columns=[['ADMISSION_TYPE', 'ADMISSION_LOCATION', 'DISCHARGE_LOCATION','INSURANCE', 'ETHNICITY', 'EDLOS','GENDER']])

#row3
row2.columns=['ADMISSION_TYPE', 'ADMISSION_LOCATION','DISCHARGE_LOCATION', 'INSURANCE', 'ETHNICITY', 'EDLOS','GENDER','DIAGNOSIS']
row2['EDLOS'] = float(row2['EDLOS'])
#row2.info()

row2

pred_input_func = tf.estimator.inputs.pandas_input_fn(x=row2,batch_size=1,num_epochs=1,shuffle=False)

predictions = ClassDNNModel.predict(pred_input_func)

predictions
my_pred = list(predictions)

		#my_pred

pred_ids = [d['class_ids'] for d in my_pred]

pred_ids = str(pred_ids)
pred_ids
CLASS6 = pred_ids[8]
CLASS6

predictions5 = ClassDNNModel5.predict(pred_input_func)

predictions5
my_pred5 = list(predictions5)

		#my_pred

pred_ids5 = [d['class_ids'] for d in my_pred5]

pred_ids5 = str(pred_ids5)
pred_ids5
CLASS5 = pred_ids5[8]
CLASS5

predictions4 = ClassDNNModel4.predict(pred_input_func)

predictions4
my_pred4 = list(predictions4)

		#my_pred

pred_ids4 = [d['class_ids'] for d in my_pred4]

pred_ids4 = str(pred_ids4)
pred_ids4
CLASS4 = pred_ids4[8]
CLASS4

predictions3 = ClassDNNModel3.predict(pred_input_func)

predictions3
my_pred3 = list(predictions3)

		#my_pred

pred_ids3 = [d['class_ids'] for d in my_pred3]

pred_ids3 = str(pred_ids3)
pred_ids3
CLASS3 = pred_ids3[8]
CLASS3

predictions2 = ClassDNNModel2.predict(pred_input_func)

predictions2
my_pred2 = list(predictions2)

		#my_pred

pred_ids2 = [d['class_ids'] for d in my_pred2]

pred_ids2 = str(pred_ids2)
pred_ids2
CLASS2 = pred_ids2[8]
CLASS2

str2=''
str3=''
str4=''
str5=''
str6=''

str2 = 'Likely to stay more than 2 days'
if CLASS2 == '0':
    str2 = 'Likely to not stay more than 2 days'

str3 = 'Likely to stay more than 3 days'
if CLASS3 == '0':
    str3 = 'Likely to not stay more than 3 days'
str4 = 'Likely to stay more than 4 days'
if CLASS4 == '0':
    str4 = 'Likely to not stay more than 4 days'

str5 = 'Likely to stay more than 5 days'
if CLASS5 == '0':
    str5 = 'Likely to not stay more than 5 days'




SEND = ''
str6 = 'The patient is likely to stay for more than 6 days. \n\nThe probability for this situation is: ' + str(probability) +  '\nThe accuracy of model is ' + str(accuracy) + '%'

if CLASS6 == '0':
	str6 = 'The patient is likely to not stay for more than 6 days. \n\nThe probability for this situation is: ' + str(probability) +  '\nThe accuracy of model is ' + str(accuracy) + '%'



users_ref =  db.reference('pred/res')
users_ref.set(  { 'Deep':str(str6) }  )

users_ref =  db.reference('pred/table/s1')
users_ref.set(  { 'Deep':str(str5) }  )

users_ref =  db.reference('pred/table/s2')
users_ref.set(  { 'Deep':str(str4) }  )

users_ref =  db.reference('pred/table/s3')
users_ref.set(  { 'Deep':str(str3) }  )

users_ref =  db.reference('pred/table/s4')
users_ref.set(  { 'Deep':str(str2) }  )

users_ref =  db.reference('pred/table/s5')
users_ref.set(  { 'Deep':str(str6) }  )

users_ref =  db.reference('pred/table/p1')
users_ref.set(  { 'Deep':str(probability5) }  )

users_ref =  db.reference('pred/table/p2')
users_ref.set(  { 'Deep':str(probability4) }  )

users_ref =  db.reference('pred/table/p3')
users_ref.set(  { 'Deep':str(probability3) }  )

users_ref =  db.reference('pred/table/p4')
users_ref.set(  { 'Deep':str(probability2) }  )

users_ref =  db.reference('pred/table/p5')
users_ref.set(  { 'Deep':str(probability) }  )

print('OUTPUT SENT')



#data =  { 'Deep':pred_ids }

# x = df['SUBJECT_ID'].value_counts(sort=False)
# print(x)



# df.to_csv('INPUT_DATABASE.csv',index=False)

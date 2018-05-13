import tensorflow as tf
import numpy as np
import analysis
from sklearn.model_selection import train_test_split
import datetime


print("{}: started model run".format(str(datetime.datetime.now())))
# Various File Stuff
CSV_PATH = 'data/vector.csv'
# This file shows the improvement bump when neutral reviews are removed from the corpus.
# CSV_PATH = 'data/vector_full_no_neutral_no_tokens.csv'
TRAIN_DATA = 'data/train.csv'
TEST_DATA = 'data/test.csv'


TEST_SIZE = 0.33 # Percentage of the data to test vs. train on

RANDOM_STATE = True # Change this if you want a truly random split between test and training. Setting it to an integer will result in the same split each time.

# Various Models Tweaks
# No significant variation in metrics was observed by changing these values.
BATCH_SIZE=100
STEPS=1000
SHUFFLE_SIZE=1000
CLASSES=6

'''
slightly cumbersome but split out the csv into test and training for a little more visibility into the created artifacts.
'''

print("Splitting master data set...")
dataset = np.loadtxt(CSV_PATH, delimiter=",", skiprows=1)
testing_data, training_data = train_test_split(dataset, test_size=TEST_SIZE, random_state=RANDOM_STATE)

formatter = "%1.0f,%1.0f,%1.8f,%1.8f,%1.8f,%1.8f,%1.8f,%1.8f,%1.8f,%1.8f,%1.0f"
np.savetxt(TRAIN_DATA, training_data, fmt=formatter, delimiter=",")
np.savetxt(TEST_DATA, testing_data, fmt=formatter, delimiter=",")

'''
Feeding the model predictions isn't implemented but here would be one way to slice up the existing set to do it
PREDICT_DATA = 'data/predict.csv'
PREDICT_SIZE = 0.33 # Percentage of the data (of the testing sample) you want to run predictions against.
predict_data, testing_data = train_test_split(testing_data, test_size=PREDICT_SIZE, random_state=RANDOM_STATE)
np.savetxt(PREDICT_DATA, predict_data, fmt=formatter, delimiter=",")
'''

# Metadata describing the text columns
COLUMNS = analysis.headers()
FIELD_DEFAULTS = [
	[0],
	[0],
	[0.0], 
	[0.0], 
	[0.0], 
	[0.0], 
	[0.0], 
	[0.0], 
	[0.0], 
	[0.0], 
	[0],
]

# (Some of this is based on existing guidelines from the tensorflow web site)
def _parse_line(line):
	# Decode the line into its fields
	fields = tf.decode_csv(line, FIELD_DEFAULTS)

	# Pack the result into a dictionary
	features = dict(zip(COLUMNS,fields))

	# Separate the label from the features
	label = features.pop('RATING')

	return features, label

def csv_input_fn(csv_path, batch_size):
	# Create a dataset containing the text lines.
	dataset = tf.data.TextLineDataset(csv_path)

	# Parse each line.
	dataset = dataset.map(_parse_line)

	# Shuffle, repeat, and batch the examples.
	dataset = dataset.shuffle(SHUFFLE_SIZE).repeat().batch(batch_size)

	# Return the dataset.
	return dataset

feature_columns = [
	tf.feature_column.numeric_column(name)
	#    for name in COLUMNS[:-1]
	# We explicitly name columns so that we can be a little looser with the upstream data pipeline.
	for name in [
		'TEXT_NEGATIVE',
		'TEXT_MIXED',
		'TEXT_POSITIVE',
		'TEXT_NEUTRAL',
		'SUMMARY_NEGATIVE',
		'SUMMARY_MIXED',
		'SUMMARY_POSITIVE',
		'SUMMARY_NEUTRAL',		
	]
]

# Build the estimator
est = tf.estimator.LinearClassifier(
	feature_columns=feature_columns,
	weight_column='UPVOTE',
	n_classes=CLASSES,
)

# Some other options for classifiers
'''
est = tf.estimator.DNNClassifier(	
	feature_columns=feature_columns,
	weight_column='UPVOTE',
	n_classes=10,
	hidden_units=[1024, 512, 256]    ,

)
'''
'''
estimator = tf.estimator.DNNLinearCombinedClassifier(
	# wide settings
	linear_feature_columns=feature_columns,
	# deep settings
	dnn_feature_columns=[
		feature_columns],
	dnn_hidden_units=[1024, 512, 256],
)

estimator = tf.estimator.BaselineClassifier(
	n_classes=CLASSES
)
'''

# Train the estimator
print("{}: Training Estimator...".format(str(datetime.datetime.now())))
est.train(
	steps=STEPS,
	input_fn=lambda : csv_input_fn(TRAIN_DATA, BATCH_SIZE)
)

print("{}: Evaluating Model...".format(str(datetime.datetime.now())))
metrics = est.evaluate(
	input_fn=lambda : csv_input_fn(TEST_DATA, BATCH_SIZE),
	steps=STEPS
)

'''
If you were going to feed the model predictions here is where you would do it:
print("{}: Running Predictions again sample data...".format(str(datetime.datetime.now())))
predictions = est.predict(
	input_fn=lambda : csv_input_fn(PREDICT_DATA, BATCH_SIZE),
	predict_keys=['classes']
)
'''

print("--------")
print("RESULTS:")
print("--------")
print(metrics)
'''
# One way to see the results of the predictions:
for i in predictions:
	print(i)
'''

print("{}: Completed model run.".format(str(datetime.datetime.now())))

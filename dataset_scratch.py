import tensorflow as tf
import numpy as np
import analysis
from sklearn.model_selection import train_test_split
import datetime

print("{}: started model run".format(str(datetime.datetime.now())))
# Various File Stuff
CSV_PATH = 'data/vector.csv'
TRAIN_DATA = 'data/train.csv'
TEST_DATA = 'data/test.csv'

TEST_SIZE = 0.20 # Percentage of the data to test vs. train on
RANDOM_STATE = True # Change this if you want a truly random split between test and training.

# Various Models Tweaks
BATCH_SIZE=100
STEPS=2000
SHUFFLE_SIZE=1000

'''slightly cumbersome but split out the csv into test and training for a little more visibility'''

print("Splitting master data set...")
dataset = np.loadtxt(CSV_PATH, delimiter=",", skiprows=1)
testing_data, training_data = train_test_split(dataset, test_size=TEST_SIZE, random_state=RANDOM_STATE)
formatter = "%1.0f,%1.0f,%1.8f,%1.8f,%1.8f,%1.8f,%1.8f,%1.8f,%1.8f,%1.8f,%1.0f"
np.savetxt(TRAIN_DATA, training_data, fmt=formatter, delimiter=",")
np.savetxt(TEST_DATA, testing_data, fmt=formatter, delimiter=",")



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



# All the inputs are numeric
feature_columns = [
    tf.feature_column.numeric_column(name)
    for name in COLUMNS[:-1]]

# Build the estimator
est = tf.estimator.LinearClassifier(feature_columns,
                                    n_classes=10)


# Train the estimator
print("Training Estimator...")
est.train(
    steps=STEPS,
    input_fn=lambda : csv_input_fn(TRAIN_DATA, BATCH_SIZE))

print("Evaluating Model...")
answer = est.evaluate(
	input_fn=lambda : csv_input_fn(TEST_DATA, BATCH_SIZE),
	steps=STEPS
)

print("--------")
print("RESULTS:")
print("--------")
print(answer)

'''
train_spec = tf.estimator.TrainSpec(
	input_fn=lambda : csv_input_fn(TRAIN_DATA, BATCH_SIZE),
	max_steps=STEPS
)
eval_spec = tf.estimator.EvalSpec(
	input_fn=lambda : csv_input_fn(TEST_DATA, BATCH_SIZE)
)
tf.estimator.train_and_evaluate(
	est, 
	train_spec, 
	eval_spec
)
'''
print("{}: Completed model run.".format(str(datetime.datetime.now())))

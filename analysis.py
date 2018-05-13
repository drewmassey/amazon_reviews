import click
import json
import gzip
import boto3
from collections import Counter, defaultdict
import pandas as pd
import numpy 

"""
Command line script for various data preparation and scrubbing tasks.

I didn't comment this in a whole lot of detail.
"""


# Locations of various artifacts
data_remote = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz'
data_local = 'data/reviews_Movies_and_TV_5.json.gz'
data_enriched = 'data/enriched.json'
data_vectorized = 'data/vector.json'
data_vectorized_csv = 'data/vector_full.csv'
data_text_summary_delta = 'data/text_summary_delta.csv'
pickles = {
	'text_token_count': 'data/text_token_count.pickle',
	'summary_token_count': 'data/summary_token_count.pickle'
}

language_code = 'en'
comprehend = boto3.client('comprehend')

# Specify the keys in the vector for easy lookup.
vector_map = [
	'upvotes',
	'downvotes',
	'text_sentiment_negative',
	'text_sentiment_mixed',
	'text_sentiment_positive',
	'text_sentiment_neutral',
	'text_sentiment_token',
	'summary_sentiment_negative',
	'summary_sentiment_mixed',
	'summary_sentiment_positive',
	'summary_sentiment_neutral',
	'summary_sentiment_token',
	'rating'
]

@click.group()
def cli():
    pass # No default command

@cli.command()
def extract():
	"""Loads basic data"""
	click.echo('Run the following command to pull data from remote:')
	click.echo('# curl {} > {} '.format(data_remote, data_local))

@cli.command()
@click.option('--data_source', default=data_local)
def enrich(data_source):

	"""Runs sentiment analysis on data set"""
	click.echo('Running sentiment analysis (this costs money don\'t do it too much)...')
	g = gzip.open(data_source, 'r')
	# This could run faster if you  batched up the requests
	for l in g:
		try: 
			j = eval(l)
			j['textSentiment'] = comprehend.detect_sentiment(
				Text=j['reviewText'],
				LanguageCode=language_code
			)
			j['summarySentiment'] = comprehend.detect_sentiment(
				Text=j['summary'],
				LanguageCode=language_code
			)
			print(json.dumps(j))
		except Exception as e:
			print("Something went wrong on {}".format(l))


def enriched2vector(j):
	return [
		j['helpful'][0],
		j['helpful'][1],
		j['textSentiment']['SentimentScore']['Negative'],
		j['textSentiment']['SentimentScore']['Mixed'],		
		j['textSentiment']['SentimentScore']['Positive'],		
		j['textSentiment']['SentimentScore']['Neutral'],				
#		j['textSentiment']['Sentiment'],						
		j['summarySentiment']['SentimentScore']['Negative'],
		j['summarySentiment']['SentimentScore']['Mixed'],		
		j['summarySentiment']['SentimentScore']['Positive'],		
		j['summarySentiment']['SentimentScore']['Neutral'],				
#		j['summarySentiment']['Sentiment'],								
		j['overall'],
	]

def headers():
	return [
		'UPVOTE',
		'DOWNVOTE',
		'TEXT_NEGATIVE',
		'TEXT_MIXED',
		'TEXT_POSITIVE',
		'TEXT_NEUTRAL',
#		'TEXT_TOKEN',
		'SUMMARY_NEGATIVE',
		'SUMMARY_MIXED',
		'SUMMARY_POSITIVE',
		'SUMMARY_NEUTRAL',		
#		'SUMMARY_TOKEN',
		'RATING',
	]

@cli.command()
def transform():
	"""Convert enriched data to a JSON file for analysis"""
	with open(data_vectorized, 'w') as of:
		with open(data_enriched, 'r') as f:
			for l in f:
				try:
					j = json.loads(l)
					of.write(json.dumps(enriched2vector(j)))
					of.write("\n")
				except Exception as e:
					click.echo("ERROR: {}".format(e))

@cli.command()
@click.option('--input_file')
@click.option('--output_file')
def transform2csv(input_file, output_file):
	"""Convert enriched data to a CSV for analysis"""
	with open(output_file, 'w') as of:
		of.write(','.join(headers()))
		of.write("\n")
		with open(input_file, 'r') as f:
			for l in f:
				try:
					j = json.loads(l)
					j['overall'] = int(j['overall'])
					of.write(",".join(str(x) for x in enriched2vector(j)))
					of.write("\n")
				except Exception as e:
					click.echo("ERROR: {}".format(e))


@cli.command()
def token_counts():
	"""Do tokens correlate to ratings?"""
	grid = defaultdict(int)
	with open(data_vectorized, 'r') as f:
		for l in f:
			j = json.loads(l)
			key = "TEXT-{}-{}".format(
				j[vector_map.index("text_sentiment_token")],
				j[vector_map.index("rating")],				
			)
			grid[key] += 1
			key = "SUMMARY-{}-{}".format(
				j[vector_map.index("summary_sentiment_token")],
				j[vector_map.index("rating")],				
			)
			grid[key] += 1

	out = Counter(grid)
	rows = [
		'POSITIVE',
		'MIXED',
		'NEUTRAL',
		'NEGATIVE',
		'NULL',
	]
	d = {
		'TEXT': [list() for i in range(0,5)],
		'SUMMARY': [list() for i in range(0,5)],
	}
	for k in ['TEXT','SUMMARY']:
		for i in range(0,5):
			d[k][i] = [0, 0, 0, 0, 0, 0]

	for k,v in out.items():
		frame, row, col = k.split('-')
		d[frame][rows.index(row)][int(float(col))] = v

	text_df = pd.DataFrame(d['TEXT'], index=rows)
	text_df = text_df.drop([0], axis=1)
	text_df = text_df.drop(['NULL'], axis=0)

	summary_df = pd.DataFrame(d['SUMMARY'], index=rows)
	summary_df = summary_df.drop([0], axis=1)
	summary_df = summary_df.drop(['NULL'], axis=0)	

	print("TEXT")
	print(text_df.head())
	text_df.to_pickle(pickles['text_token_count'])
	print("------")
	print("SUMMARY")
	print(summary_df.head())
	summary_df.to_pickle(pickles['summary_token_count'])	

@cli.command()
def sample_breakdown():
	"""Get histogram of sampled data that was passed through AWS Comprehend"""
	counts = []
	with open(data_vectorized, 'r') as f:
			for l in f:
				j = json.loads(l)
				counts.append(j[vector_map.index("rating")])
	b = Counter(counts)
	print(b)

@cli.command()
def text_summary_delta():
	"""Calculate covariance of text versus summary scores"""
	with open(data_text_summary_delta,'w') as of:
		with open(data_vectorized, 'r') as f:
				for l in f:
					j = json.loads(l)
					v1 = numpy.array(
						(
							j[vector_map.index("text_sentiment_negative")],
							j[vector_map.index("text_sentiment_mixed")],
							j[vector_map.index("text_sentiment_positive")],
							j[vector_map.index("text_sentiment_neutral")],
						)															
					)
					v2 = numpy.array(
						(
							j[vector_map.index("summary_sentiment_negative")],
							j[vector_map.index("summary_sentiment_mixed")],
							j[vector_map.index("summary_sentiment_positive")],
							j[vector_map.index("summary_sentiment_neutral")],
						)															
					)
					delta = numpy.linalg.norm(v1 - v2)
					of.write("{},{}\n".format(
						delta,
						int(float(j[vector_map.index("rating")]))
					))
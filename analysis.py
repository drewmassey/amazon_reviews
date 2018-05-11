import click
import json
import gzip
import boto3


data_remote = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz'
data_local = 'data/reviews_Movies_and_TV_5.json.gz'
language_code = 'en'

comprehend = boto3.client('comprehend')

'''
@click.command()
def cli():
    """Example script."""
    click.echo('Hello World!')
'''

@click.group()
def cli():
    pass

@cli.command()
def extract():
	"""Loads basic data"""
	click.echo('Run the following command to pull data from remote:')
	click.echo('# curl {} > {} '.format(data_remote, data_local))

@cli.command()
def enrich():

	"""Runs sentiment analysis on data set"""
	# click.echo('Running sentiment analysis (this costs money don\'t do it too much)...')
	g = gzip.open(data_local, 'r')
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



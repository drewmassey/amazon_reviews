# Amazon Reviews
Predict product ratings on Amazon using customer reviews


## Narrative

### Methodology & Thought Process.
The task is this:

We have a corpus of ~1.7 million reviews from Amazon which include the reviewer id, the product id, the number of upvotes and downvotes for that particular review, the title and body of the review (called 'summary' and 'text' respectively in the data, and then the overall rating, which ranges from 1 to 5).

_Build a model to rate the products in the list using the reviews_

The approach that I have taken is to enrich the existing corpus of data by using `AWS Comprehend,` a service that provides sentiment analysis for English language text. This allows us to express the data as a 13-dimensional vector, which includes 11 number fields and two token fields.

Since a 13 dimensional vector space is rather cumbersome to work with, I conducted some preliminary analysis to see if there were some quick wins:

- Is our sample representative?
- How frequently to title and body return similar tokens?
- Do tokens reliably map to rating?
	It turns out they do, but only in one direction (i.e., positive reviews cluster towards high ratings)
- Can we reduce the dimensionality by finding a predictable relationship between text and summary?
	Nope, see jupyter notebook
- How should upvotes and downvotes weight the prediction model?


### Assumptions & Other Considerations
Since this model was built from scratch in just a few days, there are some considerable sharp edges. At a minimum here are obstacles and further issues that would need to be addressed before applying this model to any business problem:

- _Naive Data Model_. Probably the most obvious limitation of this solution is that it doesn't actually look at the product, or the user writing, in an effort to predict the rating. In a real world system this would basically be a show-stopping limitation.
- _The Gospel of AWS_. We haven't done serious analysis of `AWS Comprehend` is enriching our data the way we would expect, we kind of take their sentiment analysis for granted.
- _Cost_. Because this relies on an external pay-as-you-go service there was a cost associated with building this model.
- _Sampling_. In order to control cost, this analysis was conducted on 173,645 records that were passed through the sentiment analysis services. A full analysis of the entire data set would reveal a different set; moreover in the interest of time and cost savings we just read the top of the original data file; even at a 10% sample a better approach would be to shuffle and randomly select a larger percentage of the original data.

We have some sense of the sampling problem through a few simple analysis steps:

```
# Counts of ratings in the raw data
gzcat data/reviews_Movies_and_TV_5.json.gz | jq .overall | sort | uniq -c

1: 104219 	(6.14%)
2: 102410 	(6.03%)
3: 201302 	(11.86%)
4: 382994 	(22.56%)
5: 906608 	(53.41%)

# Counts of ratings in the data sampled here (see `analysis sample_breakdown`)
1: 8342 	(4.74%)
2: 7973 	(4.53%)
3: 16044 	(9.12%)
4: 36582 	(20.79%)
5: 104704 	(59.50%)
```



## Technical Implementation

### Environment & Setup
These scripts are written on a box with python 3.5. This repo hasn't been tested on a lot of setups but typically you should be able to do this:

```
virtualenv venv/
source venv/bin/activate
pip install -r requirements.txt
# This last line will install the setuptools for the click python framework to make it a little more usable.
pip install --editable .
```

### Data Retrieval & Enrichment

### Correlation Analysis

### Regression

### Training

### Shuffle and Sample

### Confusion Matrix

### Cost Analysis


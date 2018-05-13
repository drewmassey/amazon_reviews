# Amazon Reviews
Predict product ratings on Amazon using customer reviews.

## The challenge
This repo draws on a corpus of ~1.7 million reviews from Amazon for Movies and TV shows (http://jmcauley.ucsd.edu/data/amazon/) that includes the reviewer id, the product id, the number of upvotes and downvotes for that particular review, the title and body of the review (called 'summary' and 'text' respectively in the data), and then the overall rating, which ranges from 1 to 5.

This task is to _build a model to rate the products in the list using the review._

## TL;DR

The approach in this repository uses a Linear Classifier in combination with a external sentiment analysis service to quickly produce a model with ~64% on sample data and ~69% accuracy on filtered data. Other classification options were explored without a significant change in performance.

A literature review of existing solutions to this problem found that some authors were able to achieve accuracies as high as 78% for a similar problem by changing the labels from 1-5 to a three target space of high / medium / low. Other similar solutions expressed their accuracy somewhat differently, with one regression-driven approach achieving a MSE of 1.112 and a group of three student projects having a mean average error of 0.494-0.511.

## Narrative

### Methodology & Thought Process.

The approach that I have taken is to enrich the existing corpus of data by using AWS Comprehend (https://aws.amazon.com/comprehend/), a service that provides sentiment analysis for English language text. This allows us to quickly express each review as a set of numerical values. As with other AWS sentiment analysis products, Comprehend provides a series of values for general categories: "Positive", "Neutral", "Negative", and "Mixed," as well as a token indicating the prevailing sentiment for any given text.

### Preliminary analysis  
I conducted some preliminary analysis on the raw data to see if there were some quick wins:

- Is our sample representative? _This depends on how you define representative. One of the challenges of building a prediction model that spans ratings from 1-5 is that  reviews tend to cluster between 4 and 5. As discussed below, the data sample passed through AWS Comprehend is distributed roughly the same as the original set._

- Do sentiment tokens ("Positive", "Negative," etc.) reliably map to ratings?
	_It turns out they do, but only in one direction (i.e., positive reviews correlate with high ratings)_

- How frequently to title and body return similar sentiments? _A correlation analysis did not indicate this was a promising way to do a dimensional reduction_. See the discussion in the Jupyter notebook in this repo.

- How should upvotes and downvotes weigh the prediction model? _This question was not exhaustively explored but the model performed comparable with or without weighting upvoted reviews more heavily or by excluding reviews without votes._


### Assumptions & Other Considerations
Since this model was built in just a few days, there are some considerable sharp edges. At a minimum here are obstacles and further issues that would need to be addressed before applying this model to any business problem:

- _Narrow Data Model_. Probably the most obvious limitation of this solution is that it doesn't actually look at the product, or who is actually writing, in an effort to predict the rating. So we don't have any insight as to whether a particular person, for example, tends towards enthusiasm or negativity. Similarly, we can't weight our model based on the rating performance of other similar products In a real world system we would definitely want to at least contemplate such data sources.
- _The Gospel of AWS_. We haven't done serious analysis of whether `AWS Comprehend` is enriching our data the way we would expect, we kind of take their sentiment analysis for granted. Further analysis would be required to determine whether this was a cost-effective approach compared to building a model in house.
- _Cost_. Because this relies on an external pay-as-you-go service there was a cost associated with building this model.
- _Sampling_. In order to control cost, this analysis was conducted on 173,645 records that were passed through the sentiment analysis service. A full analysis of the entire data set would reveal a different set; moreover in the interest of time and cost savings we just read the top of the original data file; even at a 10% sample a better approach would be to shuffle and randomly select a larger percentage of the original data.

We have some sense of the sampling problem through a few simple analysis steps. First of all, ratings in this category tend to to cluster in the 4s and 5s:

####Counts of ratings in the raw data
```
gzcat data/reviews_Movies_and_TV_5.json.gz | jq .overall | sort | uniq -c
1: 104219 	(6.14%)
2: 102410 	(6.03%)
3: 201302 	(11.86%)
4: 382994 	(22.56%)
5: 906608 	(53.41%)
```

#### Counts of ratings in the data sampled from AWS Comprehend
```
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

### Further Reading

Peter Brydon and Kevin Groyake, _Amazon Rating Prediction_ (https://pdfs.semanticscholar.org/b71b/fe0fbe009991dc52ac5b03b75b8b44be5aac.pdf)

Josh Bencina, _Amazon Review Python Notebook_ (https://gist.github.com/jbencina/03b2673a6fc27e2717650686b379eeca)

Amazon Review Raters Production Competition (https://www.kaggle.com/c/ugentml16-3/leaderboard)

* I couldn't figure out what a circle is suppose to do. Are they created by teach or students, are they suppose to lead guided meditation or what?

* there was no filter, of categorisation of circles.



* based on the line that "without creating circles that feel weird (i.e. only a group of loosely connected people), I think you need to create the circle rather than the rule?





Hey Anders,

just a few questions that I would like to clarify about the case study.

First of all, I didn't quite get the meaning of the question.


I probably misunderstood the question, but do you want me to list all 10,000 circles and their members?  like:


circle 1:

* user a
* user b
* user c

circle 2: 

* user b 
* user d
* user z

* what is the problem? is it to increase once you have created the circle, you would recommend the circle to the users right?


* what's the reason for 10,000 circles?


* need to assum one way relationship. I assume that the relationship is teacher -> multiple student

* the simple way to think about clique would be that everyone knows
  everyone in the group, or the network is complete (all nodes are
  connected to all nodes).

* the clustering coefficient is computed on the connectivity of nodes
  that are directly connected to the node of interst.

For example if A is connected to B, C D, then the clustering
coefficient is computed on how tight are B, C and D. The tightness is
the number of actual edge between the nodes divided by the number of
possible nodes. (k * (k - 1))/2 or lower/upper triangle.




I assume you want to know how I would create circles based on the analysis.

for friendship, it's an undirected graph.

something doesn't feel quite right, but not sure how to verify it.


the number of nodes remain the same that means 

########################################################################

Social Network Analysis:

* social network analysis, what is the goal here?


first solution that comes to my mind would basically be node2vec than cluster them.


* so weird is defined as loosely connected? what if it is a tight private group?

which circle I would create? so is this an optimisation? I don't even have features.



1.26M nodes, not sure how many edges there are. We can also take a look at the sparsity.

There is also the time dimension, although people from different era are less likely to connect. For the sake of analysis, I will assume that the date is the date they signed up to the app.


########################################################################

Hotel Review:

This sentiment analysis with clustering.

there are so many ways I can think of to cluster this. The simplest would be to create a tfidf matrix and then cluster

there are two steps, encode the text and then cluster

encoding type:

* tfidf
* embedding - word2vec from gensim or simply use the pretrained model.

clustering:

I think even the basic k-means would do, but happy to explore other options.



I don't think there is a need to parse the text with Spacy.

* we can also predict the sentiments since we want to know what kind of keyword lead to low sentiment.




# let's do a quick EDA, we will look at the freuency of the words and build n-gram to see the words that occur together.

Another interesting way to visualise it would be to use catboost to explain the texts.


################################

* tokenize the data

* another way is to do topic modelling which is a way of clustering as well.


* transformer for sentiment prediction, this like catboost will show which words are the activator.

################################



other fun ideas:

transformers

########################################################################

I would really target on the topics that are associated with low
sentiments. Group the data by high and low score, and the take the
average of the topics to see which area has the largest difference.


########################################################################


* There are too many adjectives in the description.

########################################################################


you can use nlp.pipe!!!! and disable "ner" when loading the model.


rule based matching allows matching over multiple tokens.
from spacy.matcher import Matcher

matcher = Matcher(nlp.vocab, validate=False)

* interesting that if you write a comprehension in tuple, it becomes a generator! 


########################################################################

there are certainly signal in the text for bad reviews, I suspect they
are negative adjectives. However, this is not really the point.

what are user needs? The review doesn't have any information about user need. It's a feedback on their experience. You can probably say the review is a reflection of whether the experience relates to their expectation but most would not have responded.

I assume the needs are something along the line:

* have breakfast
* good location
* clean room
* good for kids and family
* value for money (price related)
* extra activities and tours

etc.....


One, I belive that what you should focus on are what results in the
worst experience. And this is what customer success do as well, find
the worst offender and then fix them. Looking at those with the lowest
review, we found .... absolute nothing, since there are multiple
hotels in question.......

The question would be more suitable for a single hotel.


what if you can't find any clusters? What akind of cluster are you expect to find?



so in this case, I would go back and clarify what the stakeholder is
interested in finding out and do.

*****

at least I couldn't find a single consistent pattern in a particular
area that is associated with the negative reviews. This might change
if we have the hotel ID.

*****


you can get better topic by removing the adjectives, but that also
means you won't have the sentiments.

########################################################################

* reviews are more about experience rather than function. So not
  really segmenting the review in terms of niche.



* what are we clustering on, we are trying to cluster so we can
  understand what customers are talking about.
  

########################################################################


* I guess I've created one that's with 10k circle using undirected graph.
* need to create another one that's based on directed graph.
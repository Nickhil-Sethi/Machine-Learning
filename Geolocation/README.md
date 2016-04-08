# Geolocation
@author: Nickhil-Sethi

Code I wrote when I was at NYU's SMaPP Lab for our project on tweet geolocation. I'm hoping to apply this code on some othe language data sets. 

Investigators: Richard Bonneau, Nickhil Sethi, Yvan Scher, Duncan Penfold-Brown

##Stage 1: {New York, San Francisco}

At this point, we are just aiming to explore the problem, draft some preliminary code and perform basic feature selection. *geolocation_sketch.py*, *geolocation_sketch2.py*, *geolocation_sketch3.py* are under stage 1.

The label to predict is coordinates from {New York , San Francisco}, and the features are presence or absence of a predictive word. Tweets are streamed from the following geoboxes (west, south, east, north):

NewYorkGeo: [-74.2624761611,40.4927807345,-73.6862848252,40.9063810756]  
SanFranciscoGeo: [-122.5318908691,37.235795328,-121.7930603027,37.9301172688] 

This caputres, roughly, the Tri-State area around Manhattan (i.e. the five boroughs, New Jersey, Yonkers, and some of Long Island), and the Bay Area of San Francisco. It should be noted that this is very awkward to capture areas of interest. The geobox for NY captures for more of New Jersey than it does of Long Island, even though people commute from far out into LI. 

Each tweets in the MongoDB instance is given a random number generated from *U[0,1]* using numpy.random.rand(), which can then be used to randomly sample from the collection.

Computing perplexity is difficult because hypergeometric requires large factorials to be calculated. Special method was developed to compute perplexities stably and efficiently through log transform.

##Stage 2: {New York, San Francisco, Midwest}

*geolocation_sketch4.py* is under stage 2.

Old geoboxes were paused and new collections __NewYorkGeo_2__ and __SanFrancsicoGeo_2__ were set up to capture potentially seasonal variation in tweet features. A third geobox, __MidwestGeo__, was setup to capture the region, just northeast of Chicago to southwest of San Antonio. 

The basic idea is to incorporate a large outgroup population, so that the model can be trained on arbitraty tweets to predict the output __{NY, SF, Other}__; without an outgroup, the false positive rates would be high. 

NY:  [-74.313357,40.468728,-73.640899,40.946894]  
SF:  [-122.499149,37.119352,-121.545733,38.1566]  
Midwest: [-100.187830,28.270533,-85.402366,42.689952]

Issues to be cognizant of: 
1) Numerical stability of computing perplexities.
2) Tunnel to hades breaks at some point during runtime. Must be checked and restarted if down, in real time.
scipy compared to my code 

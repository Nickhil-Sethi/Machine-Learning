import pymongo
import pprint

client = pymongo.MongoClient('localhost',49999)
#code runs without this step!
client['admin'].authenticate('smapp_readOnly','')

NY = client.NewYorkGeo
NY_T1 = NY.tweets_1

SF = client.SanFranciscoGeo
SF_T1 = SF.tweets_1

ny = 0
print "New York...","\n"
for doc in NY_T1.find():
	if doc['random_number'] <= .1:
		print pprint.pprint(doc['text'])
		#store data
		ny += 1
	if ny == 5:
		break
sf=0
print "San Fran...","\n"
for doc in SF_T1.find():
	if doc['random_number'] <= .01:
		print pprint.pprint(doc['text'])
		#store data
		sf += 1
	if sf == 5:
		break
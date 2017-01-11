import pandas as pd 
import matplotlib.pyplot as plt


data 			= pd.read_csv("/Users/Nickhil_Sethi/Documents/Datasets/dow_jones_index/dow_jones_index.data")
grouped_data 	= data.groupby('stock')

# let's take a look at the percent changes in price
percent_changes = pd.DataFrame()
for key,frame in grouped_data:
	frame.set_index('date',inplace=True)
	percent_changes[key] = frame['percent_change_price']

threshold	= 1.0
cov			= percent_changes.corr()
# print(cov)
print(len(percent_changes))
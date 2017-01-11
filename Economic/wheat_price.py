import os 
import pandas as pd 
import tensorflow as tf
import matplotlib.pyplot as plt

from collections import OrderedDict

FILE 		 = "/Users/Nickhil_Sethi/Documents/Datasets/actual-wheatprice-index-european.csv"
DATA 	  	 = OrderedDict()

wheat_prices = pd.read_csv(FILE)
wheat_prices.set_index('Year',inplace=True)
wheat_prices.plot()
plt.show()


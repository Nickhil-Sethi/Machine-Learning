import os
import pandas as pd

from collections import OrderedDict

DIRECTORY = "/Users/Nickhil_Sethi/Documents/Datasets/WDI_csv/"
DATA 	  = OrderedDict()
for file in os.listdir(DIRECTORY):
	print('reading in %s ' % file)
	data = pd.read_csv(os.path.join(DIRECTORY,file))
	file = file.replace(".csv","")
	DATA[file] = data

for key,value in DATA.items():
	print(key)
	print(value.columns.tolist())

print(DATA['WDI_Data'])
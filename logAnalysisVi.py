#import plotly.express as px
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import time

def isfloat(num): 
	try:
		float(num)
		return True
	except ValueError:
		return False

def printCurves():
	df = pd.read_csv("logVI.csv")
	nx = int(df.iloc[0]["episode"])
	ny = int(df.iloc[0]["value"])
	n = len(df.index)
	for i in range(n-1):
		V=df.iloc[i+1]["value"].replace("[","").replace("]","").split(" ")
		V= [float(x) for x in V if isfloat(x)]
		V = np.abs(np.asarray(np.reshape(np.asarray(list(map(float,V))),(int(nx),int(ny)))))
		fig, ax = plt.subplots()
		print("V : ", V)
		ax.matshow(V, cmap='ocean')
		for i in range(nx):
			for j in range(ny):
				c = abs(V[j,i])
				ax.text(i, j, str(c), va='center', ha='center')
		plt.show()
		plt.close()

	#ax.matshow()

printCurves();
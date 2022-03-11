#import plotly.express as px
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import time
from matplotlib import animation

def isfloat(num): 
	try:
		float(num)
		return True
	except ValueError:
		return False
'''
def printCurves():
	df = pd.read_csv("logVI.csv")
	nx = int(df.iloc[0]["episode"])
	ny = int(df.iloc[0]["value"])
	n = len(df.index)
	im = plt.imshow([[]])
	for i in range(n-1):
		V=df.iloc[i+1]["value"].replace("[","").replace("]","").split(" ")
		V= [float(x) for x in V if isfloat(x)]
		V = np.abs(np.asarray(np.reshape(np.asarray(list(map(float,V))),(int(nx),int(ny)))))
		fig, ax = plt.subplots()
		print("V : ", V)
		ax.matshow(V, cmap='gray')
		for i in range(nx):
			for j in range(ny):
				c = abs(V[j,i])
				ax.text(i, j, str(c), va='center', ha='center')
		plt.imshow(V)
		plt.colorbar()
		plt.set_cmap('viridis')
		plt.show()

	#ax.matshow()
'''
def init():
    im.set_data(np.zeros((nx, ny)))

def animate(i):
	V=df.iloc[i+1]["value"].replace("[","").replace("]","").split(" ")
	V= [float(x) for x in V if isfloat(x)]
	V = np.abs(np.asarray(np.reshape(np.asarray(list(map(float,V))),(int(nx),int(ny)))))
	im.set_data(V)
	return im

df = pd.read_csv("logVI.csv")
nx = int(df.iloc[0]["episode"])
ny = int(df.iloc[0]["value"])
n = len(df.index)


V=df.iloc[n-1]["value"].replace("[","").replace("]","").split(" ")
V= [float(x) for x in V if isfloat(x)]
V = np.abs(np.asarray(np.reshape(np.asarray(list(map(float,V))),(int(nx),int(ny)))))
vmax = np.amax(V)
fig = plt.figure()
data = np.zeros((nx, ny))
im = plt.imshow(data, cmap='gist_gray_r', vmin=0, vmax=vmax)


anim = animation.FuncAnimation(fig, animate, init_func=init, frames=n-1,
                               interval=500,repeat=False)
plt.show()
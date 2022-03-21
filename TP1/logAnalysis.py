import plotly.express as px
import pandas as pd


class logAnalysis:

	def __init__(self, file : str):
		self.file = file
		self.values = []

	def printCurves(self):
		df = pd.read_csv("TP1/partie_3/visualisation/logQ.csv")
		fig = px.scatter(x=df["episode"], y=df["value"])
		fig.show()

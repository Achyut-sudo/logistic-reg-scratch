# %%
import numpy as np 
import pandas as pd 
import math
import time
import matplotlib.pyplot as plt
import decimal
import warnings
import os 
# %%
warnings.filterwarnings("ignore")
# 555
class Model:

	def __init__(self,dataframe,Y,directory,thetaTranspose = None,precision = 7):
		self.X = self.createX(self.dropColumns(dataframe))
		self.precision = precision
		self.valuesUB = 10**self.precision
		self.valuesLB = -1*self.valuesUB
		decimal.getcontext().prec = 10**self.precision
		self.directory = directory
		(self.sigmoidLB,self.sigmoidUB) = (0.000001,0.999999)
		self.Y = Y
		self.directory = directory
		self.vectorizedSigmoid = np.vectorize(self.sigmoid)
		self.nanFilter = np.vectorize(lambda val: val if np.isnan(val) == False else 0)
		self.thetaT = self.initializeThetaT() if thetaTranspose == None else thetaTranspose
		self.vectorizedPrediction = np.vectorize(self.predict)
		self.vectorizedLimit = np.vectorize(self.limit)

	def createX(self,dataframe):
		retArr = np.array([])
		for i,row in dataframe.iterrows():
			temp = np.array(row)
			temp = temp.reshape((self.featureSize,1))
			try:
				retArr = np.hstack([retArr,temp])
			except ValueError:
				retArr = temp
		return retArr

	def dropColumns(self,dataframe):
		self.dropedColumns = None
		print("filtered datframe:\n",dataframe.head(5))
		columns = dataframe.columns
		self.dropedColumns =[]
		for i,column in enumerate(columns) :
			print(i,column)
		while True:
			try:
				index = int(input("enter a column index to drop or any string to exit : "))
				try:
					try:
						self.dropedColumns.append((index,columns[index]))
					except AttributeError :
						self.dropedColumns = [(index,columns[index])]
				except IndexError:
					index = int(input("enter avlid index:  "))
			except ValueError:
				break
		retDataFrame =dataframe.drop(columns=[columns for index,columns in self.dropedColumns])
		(self.TrainingSize,self.featureSize) = retDataFrame.shape
		return retDataFrame

	def initializeThetaT(self):
		choice = int(input("1. load from file\n2. np.ones\n3. np.zeros\n4. np.random.uniform\n5. slected range np.random.uniform\n"))
		if choice == 1:
			fileName = input("fileName: ")
			index = int(input("index: "))
			return self.readFile(fileName,index)
		if choice == 2:
			return np.ones((1,self.featureSize))
		if choice == 3:
			return np.zeros((1,self.featureSize))
		if choice == 4:
			(a,b) = (float(input("lower limit: ")),float(input("upper limit: ")))
			return np.random.uniform(min(a,b),max(a,b),(1,self.featureSize))
		if choice == 5:
			thetaTs = []
			counter = 0
			while True :
				print(f"index: {counter}")
				(a,b) = (float(input("lower limit: ")),float(input("upper limit: ")))
				temp = np.random.uniform(min(a,b),max(a,b),(1,self.featureSize))
				print(f"theta.T: {temp}")
				thetaTs.append(temp)
				h = np.matmul(temp,self.X)
				h = self.vectorizedSigmoid(h)
				accuracy = sum([1 for i in range(self.TrainingSize) if self.predict(h[0][i]) == self.Y[0][i]])
				accuracy = (accuracy/self.TrainingSize)*100 
				print(f"accuracy for this parameter matrix(theta.T): {accuracy} ")
				choice = input("enter a index or enter any string for more iteration: ")
				try:
					choice = int(choice)
					while True :
						try:
							return thetaTs[choice]
						except IndexError:
							choice = int(input("enter a valid index: "))
				except ValueError:
					counter += 1

	startToken = lambda self,index: f"{index}START\n"
	endToken = lambda self,index: f"{index}END\n"

	def writeToFile(self,array,fileName,index,overwrite = False):
		tokens= [self.startToken(index),self.endToken(index)]
		with open(fileName ,"a+" if overwrite == False else "w+") as wf:
			wf.write(tokens[0])
			for row in array:
				for element in row[:-1] :
					wf.write(f"{element},")
				wf.write(f"{row[-1]}\n")
			wf.write(tokens[1])

	def readFile(self,fileName,index):
		tokens = [self.startToken(index),self.endToken(index)]
		write = False
		retArr = np.array([])
		with open(fileName,"r") as rf :
			for line in rf.readlines() :
				if line in tokens :
					if line == tokens[0] :
						write = True
					else:
						break
				else:
					if write == True :
						temp = np.fromstring(line[:-1],sep=",")
						temp = np.array([temp])
						try:
							retArr = np.vstack([retArr,temp])
						except ValueError :
							retArr = temp
		return retArr

	limit = lambda self,value: min(self.valuesUB,max(self.valuesLB,value))
	sigmoid = lambda self,hypo: min(0.999999999999,max(0.000000000001,1/(1+np.exp(-hypo))))
	predict = lambda sel,hypo : 1 if hypo >= 0.5 else 0

	def hypothesis(self):
		self.linearHypo = np.matmul(self.thetaT,self.X)
		self.hypo = self.vectorizedSigmoid(self.linearHypo)
		self.predictions = self.vectorizedPrediction(self.hypo)
		self.accuracy = (sum(list(map(lambda tup:1 if tup[0] == tup[1] else 0,zip(self.predictions[0],self.Y[0]))))/self.TrainingSize)*100

	def gradientDescent(self,alpha):
		temp = np.zeros(self.thetaT.shape)
		for i in range(self.TrainingSize):
			coefficient = self.Y[0][i] - self.hypo[0][i]
			temp1 = coefficient*np.array([self.X[:,i]])
			temp = self.vectorizedLimit(temp + temp1)
		self.thetaT = self.vectorizedLimit(self.thetaT + (alpha/self.TrainingSize)*temp)

	def j(self):
		cost = 0
		for i in range(self.TrainingSize):
			(y,h) = (self.Y[0][i],self.hypo[0][i])
			cost += y*np.log(h)
			cost += (1-y)*np.log(1-h)
		self.cost = -1*self.limit(cost/self.TrainingSize)

	dropedColumnsIndices = lambda self: f"dc{','.join([str(index) for index,columns in self.dropedColumns])}"

	def createFile(self,directory,fileName,extension):
		file = f"{directory}\\{fileName}.{extension}"
		nxt = 1
		while True:
			if os.path.exists(file) == False:
				break
			else:
				file = f"{directory}\\{fileName}_{nxt}.{extension}"
				nxt += 1
		return file
		
	def batchTraining(self,alpha,size=100):
		DCInitials = self.dropedColumnsIndices()
		thetaFile = self.createFile(directory=self.directory,fileName=f"{DCInitials}alpha{alpha}size{size}",extension="txt")
		figsFile = self.createFile(directory=f"{self.directory}\\Figs",fileName=f"{DCInitials}alpha{alpha}size{size}",extension="png")
		accuracies =[]
		costs = []
		startTime = time.time()
		for i in range(size):
			self.writeToFile(array=self.thetaT,fileName=thetaFile,index=i,overwrite=True if i == 0 else False)
			print("------------------------------------------------------------------")
			print(f"iteration {i+1}")
			print(f"theta.T: {self.thetaT}")
			self.hypothesis()
			accuracies.append(self.accuracy)
			self.j()
			costs.append(self.cost)
			print(f"accuracy: {self.accuracy} cost: {self.cost}")
			self.gradientDescent(alpha)
			print(f"theta.T: {self.thetaT}")
			print("------------------------------------------------------------------")
		runTime = time.time()--startTime
		print(f"runTime: {runTime}")
		maxAccuracy=max(accuracies)
		maxArr = np.array([[i  for i,accuracy in enumerate(accuracies) if accuracy == maxAccuracy]])
		self.writeToFile(index="MaxAccuracyIndices",array= maxArr,fileName=thetaFile,overwrite = False)
		with open("log.txt","a+") as af:
			af.write(f"directory: {self.directory} alpha: {alpha} maximum accuracy {maxAccuracy} size: {size} dropedColumns: {[columns for cindex,columns in self.dropedColumns]}\n")
		figure = plt.figure()
		plot1 = figure.add_subplot(2,1,1)
		plot1.plot(range(size),accuracies)
		plot1.set_xlabel("no. of iterations")
		plot1.set_ylabel("accuracy in percentage")
		plot2 = figure.add_subplot(2,1,2)
		plot2.plot(range(size),costs)
		plot2.set_xlabel("no. of iterations")
		plot2.set_ylabel(f"j(\u03F4)",rotation=0)
		figure.suptitle(f"alpha: {alpha} maximum accuracy: {maxAccuracy} size: {size}\ndropedColumns: {self.dropedColumns}")
		figure.savefig(figsFile,bbox_inches='tight')
		figure.show()
		self.maxThetaTs = maxArr[0]
		self.file = thetaFile
		choice = input("enter \"y\" to write Max accuracies index to a file: ")
		if "y" in choice  :
			fileName = input("file Name: ")
			while True:
				try:
					index = int(input("enter start index: "))
					break
				except ValueError:
					index = int(input("enter a integer: "))
			overwrite = False if os.path.exists(fileName) == True else True
			self.writeMaxes(fileName =fileName,startIndex = index,overwrite=overwrite)


	def writeMaxes(self,fileName,startIndex,overwrite=False):
		print(self.maxThetaTs)
		for i,j in enumerate(self.maxThetaTs):
			temp = self.readFile(fileName=self.file,index=int(j))
			if i == 0 :
				self.writeToFile(fileName=fileName,index=startIndex + i,array=temp,overwrite=overwrite)
			else:
				self.writeToFile(fileName=fileName,index=startIndex + i,array=temp,overwrite=False)

				
#%%

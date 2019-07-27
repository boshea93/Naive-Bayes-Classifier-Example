import math
import numpy as np
import csv
import pandas as pd



#Function to count instances of a class. Function inputs are data for the dataset, labelColInd for the
#index of the label field in the dataset, and labelValue for the class we are counting instances of.
def countInstances(data,labelColInd,labelValue):

	numOfInstances = 0
	for i in range(data.shape[0]):
		if data[i][labelColInd] == labelValue:
			numOfInstances+=1

	return numOfInstances

#Function to estimate P(C) inputs are the dataset, index of the label field, and the
#number of instances of the class C
def estimatePrior(data, labelColInd, labelValue, numOfClassInstances):

	return (numOfClassInstances/data.shape[0])

#Function to estimate the mean of the gaussian for each attribute in a dataset
def estimateMeans(data, labelColInd, labelValue,numOfClassInstances):

	means = np.zeros(data.shape[1])

	for j in range(data.shape[1]):
		#print("j = ", j)
		if j == labelColInd:
			means[j] = None

		else:
			sumOfValues = 0

			for i in range(data.shape[0]):
				if data[i][labelColInd] == labelValue:
					sumOfValues += data[i][j]
			#print("Sum: ", sumOfValues)
			means[j] = sumOfValues/numOfClassInstances

	return means

#Function to estimate the variance of the gaussian for each attribute in a dataset
def estimateVariances(data,means,labelColInd,labelValue, numOfClassInstances):

	
	variances = np.zeros(data.shape[1])


	for j in range(data.shape[1]):

		if j == labelColInd:
			variances[j] = None

		else:
			sumOfSquareDiff = 0
			for i in range(data.shape[0]):
				if data[i][labelColInd] == labelValue:
					sumOfSquareDiff += (data[i][j] - means[j])**2

			variances[j] = sumOfSquareDiff/(numOfClassInstances-1)


	return variances

def gaussianPDF(mean,variance,x):

	temp = ((x-mean)**2)/(2*variance)
	return (1/math.sqrt(2*math.pi*variance))*math.exp(-1*temp)

def binaryNaiveBayesClassify(data,labelColInd,class0Means,class0Variances,class1Means,class1Variances,class0Prior,class1Prior):


	predictions = np.zeros(data.shape[0])

	for i in range(data.shape[0]):

		class0Temp = np.zeros(data.shape[1])
		class1Temp = np.zeros(data.shape[1])

		for j in range(data.shape[1]):
			if j == labelColInd:
				class0Temp[j] = None
				class1Temp[j] = None

			else:
				class0Temp[j] = np.log(gaussianPDF(class0Means[j],class0Variances[j],data[i][j]))
				class1Temp[j] = np.log(gaussianPDF(class1Means[j],class1Variances[j],data[i][j]))

		logXiGivenClass0 = 0
		logXiGivenClass1 = 0

		for j in range(data.shape[1]):
			if j != labelColInd:
				logXiGivenClass0 += class0Temp[j]
				logXiGivenClass1 += class1Temp[j]

		logXiGivenClass0 += np.log(class0Prior)
		logXiGivenClass1 += np.log(class1Prior)

		#If there is a tie give the label 1
		if(logXiGivenClass0 > logXiGivenClass1):
			predictions[i] = 0
		else:
			predictions[i] = 1
	
	return predictions



if __name__ == '__main__':

	#Read in Data using Pandas read_csv function
	testData = pd.read_csv('spambasetest.csv', header = None)
	testData = testData.values
	trainData = pd.read_csv('spambasetrain.csv', header = None)
	trainData = trainData.values

	#Column Index for Label Value in data sets 
	labelColInd = 9

	#Field Names for our data sets 
	attributeNames = [  "char_freq_;",
						"char_freq_(",
						"char_freq_[",
						"char_freq_!",
						"char_freq_$",
						"char_freq_#",
						"capital_run_length_average",
						"capital_run_length_longest",
						"capital_run_length_total"]

	#Frequencies for Each Class in Training Data Set
	class0Instances = countInstances(trainData,labelColInd,0)
	class1Instances = countInstances(trainData,labelColInd,1)
	

	#P(C) Values Estimated from Training Data Set
	priorClass0 = estimatePrior(trainData,labelColInd,0,class0Instances)
	priorClass1 = estimatePrior(trainData,labelColInd,1,class1Instances)

	#Mean Values Corresponding to (xi| Class) from Training Data Set
	class0Means = estimateMeans(trainData,labelColInd,0,class0Instances)
	class1Means = estimateMeans(trainData,labelColInd,1,class1Instances)

	#Variance Values Corresponding to (xi| Class) from Training Data Set
	class0Variances = estimateVariances(trainData,class0Means,labelColInd,0,class0Instances)
	class1Variances = estimateVariances(trainData,class1Means,labelColInd,1,class1Instances)

	#Predictions on Test Data Generated 
	predictions = binaryNaiveBayesClassify(testData,labelColInd,class0Means,class0Variances,class1Means,class1Variances,priorClass0,priorClass1)

	#Routine to calculate number of records in test data predicted correctly
	numClassifiedCorrect = 0
	numClassifiedIncorrect = 0

	for i in range(testData.shape[0]):
		if predictions[i] == testData[i][labelColInd]:
			numClassifiedCorrect +=1
		else:
			numClassifiedIncorrect +=1

	percentageError = numClassifiedIncorrect/testData.shape[0]


	print("Class 0 Instances: ", class0Instances)
	print("Class 1 Instances: ", class1Instances)
	print("P(0) = ",priorClass0)
	print("P(1) = ",priorClass1)


	print("")
	print("(Means, Variance) Pairs Corresponding to (xi| Class 0):","\n")

	for i in range(trainData.shape[1]):
		if i != labelColInd:
			print(attributeNames[i]," : (",class0Means[i]," , ", class0Variances[i], " )")
			#print("")

	print("")
	print("(Means, Variance) Pairs Corresponding to (xi| Class 1):","\n")

	for i in range(trainData.shape[1]):
		if i != labelColInd:
			print(attributeNames[i]," : (",class1Means[i]," , ", class1Variances[i], " )")
			#print("")

	print("")
	print("Testing Results:")

	print("Total Number Classified Correctly: ", numClassifiedCorrect)
	print("Total Number Classified Incorrectly: ",numClassifiedIncorrect)
	print("Percentage Error: ", percentageError*100, "%")



	print("Predictions:")
	print(predictions)

	#for i in range(len(predictions)):
		#print(predictions[i])



